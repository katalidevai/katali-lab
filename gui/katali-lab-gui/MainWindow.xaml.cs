using System.Collections.ObjectModel;
using Microsoft.Win32;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using KataliLabGui.Models;

namespace KataliLabGui;

public partial class MainWindow : Window
{
    private const string ModelsRoot = @"C:\models";

    private static readonly (string Path, string Label)[] KnownDefaults =
    [
        (@"C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf", "Qwen3.6-35B-A3B Q4_K_M"),
        (@"C:\models\Qwen_Qwen3.5-122B-A10B-Q4_K_M\Qwen_Qwen3.5-122B-A10B-Q4_K_M", "Qwen3.5-122B-A10B Q4_K_M"),
    ];

    // stderr phase keywords katali-lab may emit (case-insensitive substring match)
    private static readonly (string Needle, string Status)[] PhaseHints =
    [
        ("loading model", "loading model…"),
        ("loading", "loading model…"),
        ("prefill", "prefill…"),
        ("prefilling", "prefill…"),
        ("generating", "generating…"),
        ("decode", "generating…"),
        ("done", "done"),
        ("finished", "done"),
    ];

    private readonly ObservableCollection<ModelEntry> _models = new();
    private readonly GuiSettings _settings;
    private Process? _runningProcess;
    private CancellationTokenSource? _cts;
    private TextBlock? _streamingTextBlock;
    private Border? _streamingBubble;
    private bool _isGenerating;
    private bool _stopRequested;
    private int _runId;
    private bool _suppressModelSave;
    private string _lastSpeed = "";
    private string? _screenshotPath;

    // Generation wall-clock timer (compare with/without think)
    private readonly DispatcherTimer _genTimer = new() { Interval = TimeSpan.FromMilliseconds(100) };
    private DateTime _genStartedUtc;
    private double _lastReplySec;
    private string _lastToks = "";
    private bool _seenOutput;

    // Residual <think>...</think> strip across stdout chunks (band-aid; engine
    // already disables thinking by default). Honors KATALI_THINK=1.
    private bool _stripThink = true;
    private bool _inThinkBlock;
    private readonly StringBuilder _thinkCarry = new();

    private static readonly Brush UserBubbleBrush = new SolidColorBrush(Color.FromRgb(0x1E, 0x3A, 0x4C));
    private static readonly Brush AssistantBubbleBrush = new SolidColorBrush(Color.FromRgb(0x1C, 0x23, 0x30));
    private static readonly Brush TextPrimaryBrush = new SolidColorBrush(Color.FromRgb(0xE6, 0xED, 0xF3));
    private static readonly Brush TextSecondaryBrush = new SolidColorBrush(Color.FromRgb(0x8B, 0x94, 0x9E));
    private static readonly Brush AccentBrush = new SolidColorBrush(Color.FromRgb(0x2D, 0xD4, 0xBF));
    private static readonly Brush GeneratingBrush = new SolidColorBrush(Color.FromRgb(0x6E, 0x76, 0x81));

    public MainWindow()
    {
        InitializeComponent();
        _settings = GuiSettings.Load();
        ModelCombo.ItemsSource = _models;
        ParseScreenshotArg();
        Loaded += OnLoaded;
        Closed += OnClosed;
        _genTimer.Tick += GenTimer_Tick;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        RefreshModels();
        PromptBox.Focus();
        if (!string.IsNullOrEmpty(_screenshotPath))
        {
            // Defer one layout pass so the window is actually painted.
            Dispatcher.BeginInvoke(async () =>
            {
                await Task.Delay(400);
                try { CaptureScreenshot(_screenshotPath!); }
                catch { /* best-effort */ }
                Application.Current.Shutdown();
            }, DispatcherPriority.ApplicationIdle);
        }
    }

    private void OnClosed(object? sender, EventArgs e)
    {
        PersistSelectedModel();
        KillRunningProcess();
    }

    private void ParseScreenshotArg()
    {
        var args = Environment.GetCommandLineArgs();
        for (var i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], "--screenshot", StringComparison.OrdinalIgnoreCase))
            {
                _screenshotPath = args[i + 1];
                break;
            }
        }
    }

    /// <summary>
    /// Capture the main window via RenderTargetBitmap and write a PNG.
    /// Invoked when started with: katali-lab-gui.exe --screenshot &lt;path.png&gt;
    /// </summary>
    private void CaptureScreenshot(string path)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        var dpi = VisualTreeHelper.GetDpi(this);
        var w = Math.Max(1, (int)ActualWidth);
        var h = Math.Max(1, (int)ActualHeight);
        var rtb = new RenderTargetBitmap(
            (int)(w * dpi.DpiScaleX),
            (int)(h * dpi.DpiScaleY),
            dpi.PixelsPerInchX,
            dpi.PixelsPerInchY,
            PixelFormats.Pbgra32);
        rtb.Render(this);

        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(rtb));
        using var fs = File.Create(path);
        encoder.Save(fs);
    }

    // ── Model discovery / persistence ────────────────────────────────────

    private void RefreshModels_Click(object sender, RoutedEventArgs e) => RefreshModels();

    private void ModelCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_suppressModelSave) return;
        if (ModelCombo.SelectedItem is ModelEntry)
            PersistSelectedModel();
    }

    private void PersistSelectedModel()
    {
        if (ModelCombo.SelectedItem is not ModelEntry m) return;
        _settings.RememberModelPath(m.Path);
        _settings.Save();
    }

    private void RefreshModels()
    {
        var previous = (ModelCombo.SelectedItem as ModelEntry)?.Path
                       ?? _settings.LastModelPath;
        _suppressModelSave = true;
        try
        {
            _models.Clear();

            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            if (Directory.Exists(ModelsRoot))
            {
                foreach (var file in Directory.EnumerateFiles(ModelsRoot, "*.gguf", SearchOption.TopDirectoryOnly))
                    AddModel(file, LabelFromGguf(file), isDir: false, seen);

                foreach (var dir in Directory.EnumerateDirectories(ModelsRoot))
                    TryAddModelDirectory(dir, seen);
            }

            foreach (var (path, label) in KnownDefaults)
            {
                if (seen.Contains(path)) continue;
                if (File.Exists(path))
                    AddModel(path, label, isDir: false, seen);
                else if (Directory.Exists(path) && DirectoryContainsGguf(path))
                    AddModel(path, label, isDir: true, seen);
            }

            // Recents / browsed paths (any drive) — Refresh must NOT wipe these.
            foreach (var recent in _settings.RecentModelPaths.ToList())
                TryAddExistingPath(recent, seen);

            if (!string.IsNullOrEmpty(_settings.LastModelPath))
                TryAddExistingPath(_settings.LastModelPath!, seen);

            if (_models.Count == 0)
            {
                SetStatus("idle — no models found (try Browse…)");
                return;
            }

            var match = previous != null
                ? _models.FirstOrDefault(m => string.Equals(m.Path, previous, StringComparison.OrdinalIgnoreCase))
                : null;
            ModelCombo.SelectedItem = match ?? _models[0];
            SetStatus($"idle — {_models.Count} model(s)");
        }
        finally
        {
            _suppressModelSave = false;
            // Persist restored / default selection so next launch remembers it.
            PersistSelectedModel();
        }
    }
    private void TryAddModelDirectory(string dir, HashSet<string> seen)
    {
        var directGgufs = Directory.EnumerateFiles(dir, "*.gguf", SearchOption.TopDirectoryOnly).Any();
        if (directGgufs)
        {
            AddModel(dir, LabelFromDir(dir), isDir: true, seen);
            return;
        }

        foreach (var sub in Directory.EnumerateDirectories(dir))
        {
            if (Directory.EnumerateFiles(sub, "*.gguf", SearchOption.TopDirectoryOnly).Any())
                AddModel(sub, LabelFromDir(sub), isDir: true, seen);
        }
    }

    private static bool DirectoryContainsGguf(string dir)
    {
        try
        {
            return Directory.EnumerateFiles(dir, "*.gguf", SearchOption.TopDirectoryOnly).Any()
                || Directory.EnumerateDirectories(dir)
                    .Any(s => Directory.EnumerateFiles(s, "*.gguf", SearchOption.TopDirectoryOnly).Any());
        }
        catch
        {
            return false;
        }
    }

    private void AddModel(string path, string label, bool isDir, HashSet<string> seen)
    {
        if (!seen.Add(path)) return;
        _models.Add(new ModelEntry(label, path, isDir));
    }

    private static string LabelFromGguf(string filePath) =>
        BeautifyLabel(Path.GetFileNameWithoutExtension(filePath));

    private static string LabelFromDir(string dirPath)
    {
        var name = Path.GetFileName(dirPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        return BeautifyLabel(name);
    }

    private static string BeautifyLabel(string raw)
    {
        var s = raw;
        if (s.StartsWith("Qwen_", StringComparison.OrdinalIgnoreCase))
            s = s.Substring("Qwen_".Length);
        s = s.Replace('_', ' ');
        return s;
    }


    // ── Browse any drive ─────────────────────────────────────────────────

    private void BrowseModel_Click(object sender, RoutedEventArgs e)
    {
        // Shift+Browse… → folder picker for multi-shard models
        if ((Keyboard.Modifiers & ModifierKeys.Shift) != 0)
        {
            BrowseFolder_Click(sender, e);
            return;
        }

        var dlg = new OpenFileDialog
        {
            Title = "Select GGUF model",
            Filter = "GGUF models (*.gguf)|*.gguf|All files (*.*)|*.*",
            CheckFileExists = true,
            Multiselect = false,
        };
        var initial = ResolveBrowseInitialDir();
        if (!string.IsNullOrEmpty(initial) && Directory.Exists(initial))
            dlg.InitialDirectory = initial;

        if (dlg.ShowDialog(this) != true) return;

        var chosen = dlg.FileName;
        RememberBrowseDir(Path.GetDirectoryName(chosen));

        // Multi-shard: parent folder has multiple .gguf siblings → use the directory
        var parent = Path.GetDirectoryName(chosen);
        if (!string.IsNullOrEmpty(parent) && CountGgufSiblings(parent) > 1)
        {
            AcceptBrowsedPath(parent, isDir: true);
            SetStatus($"idle — multi-shard folder: {Path.GetFileName(parent.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))}");
            return;
        }

        AcceptBrowsedPath(chosen, isDir: false);
    }

    private void BrowseFolder_Click(object sender, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog
        {
            Title = "Select multi-shard model folder",
            Multiselect = false,
        };
        var initial = ResolveBrowseInitialDir();
        if (!string.IsNullOrEmpty(initial) && Directory.Exists(initial))
            dlg.InitialDirectory = initial;

        if (dlg.ShowDialog(this) != true) return;

        var folder = dlg.FolderName;
        if (string.IsNullOrEmpty(folder)) return;
        RememberBrowseDir(folder);

        if (!DirectoryContainsGguf(folder))
        {
            MessageBox.Show(
                "That folder has no .gguf files.\nPick the folder that contains the shard files.",
                "Katali Lab", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        AcceptBrowsedPath(folder, isDir: true);
        SetStatus($"idle — folder: {Path.GetFileName(folder.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))}");
    }

    private string ResolveBrowseInitialDir()
    {
        if (!string.IsNullOrEmpty(_settings.LastBrowseDirectory)
            && Directory.Exists(_settings.LastBrowseDirectory))
            return _settings.LastBrowseDirectory!;

        if (ModelCombo.SelectedItem is ModelEntry cur)
        {
            try
            {
                if (cur.IsDirectory && Directory.Exists(cur.Path))
                    return cur.Path;
                var parent = Path.GetDirectoryName(cur.Path);
                if (!string.IsNullOrEmpty(parent) && Directory.Exists(parent))
                    return parent;
            }
            catch { /* ignore */ }
        }

        if (Directory.Exists(ModelsRoot))
            return ModelsRoot;
        return Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory);
    }

    private void RememberBrowseDir(string? dir)
    {
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return;
        _settings.LastBrowseDirectory = dir;
        _settings.Save();
    }

    private static int CountGgufSiblings(string dir)
    {
        try
        {
            return Directory.EnumerateFiles(dir, "*.gguf", SearchOption.TopDirectoryOnly).Count();
        }
        catch
        {
            return 0;
        }
    }

    private void TryAddExistingPath(string path, HashSet<string> seen)
    {
        if (string.IsNullOrWhiteSpace(path) || seen.Contains(path)) return;
        try
        {
            if (Directory.Exists(path) && DirectoryContainsGguf(path))
                AddModel(path, LabelFromDir(path), isDir: true, seen);
            else if (File.Exists(path) && path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                AddModel(path, LabelFromGguf(path), isDir: false, seen);
        }
        catch
        {
            /* ignore missing / inaccessible paths */
        }
    }

    private static string NormalizeModelPath(string path, bool isDir)
    {
        if (string.IsNullOrWhiteSpace(path)) return path;
        try
        {
            var full = Path.GetFullPath(path.Trim());
            if (isDir)
                full = full.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return full;
        }
        catch
        {
            return path.Trim();
        }
    }

    private static string LabelForSelection(string path, bool isDir)
    {
        var baseLabel = isDir ? LabelFromDir(path) : LabelFromGguf(path);
        // Show where it lives so the dropdown clearly reflects a Browse pick.
        try
        {
            var root = Path.GetPathRoot(path) ?? "";
            if (!path.StartsWith(ModelsRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)
                && !string.Equals(path, ModelsRoot, StringComparison.OrdinalIgnoreCase))
            {
                var tip = path.Length <= 56 ? path : ("…" + path.Substring(path.Length - 54));
                return $"{baseLabel}  ({tip})";
            }
            if (!string.IsNullOrEmpty(root)
                && !root.StartsWith("C:", StringComparison.OrdinalIgnoreCase))
            {
                return $"{baseLabel}  ({root.TrimEnd('\\')})";
            }
        }
        catch { /* keep base label */ }
        return baseLabel;
    }

    /// <summary>Insert/select browsed model so the dropdown text and list match the pick.</summary>
    private void AcceptBrowsedPath(string path, bool isDir)
    {
        path = NormalizeModelPath(path, isDir);
        var label = LabelForSelection(path, isDir);

        _suppressModelSave = true;
        try
        {
            var existingIdx = -1;
            for (var i = 0; i < _models.Count; i++)
            {
                if (string.Equals(_models[i].Path, path, StringComparison.OrdinalIgnoreCase))
                {
                    existingIdx = i;
                    break;
                }
            }

            ModelEntry entry;
            if (existingIdx < 0)
            {
                entry = new ModelEntry(label, path, isDir);
                _models.Insert(0, entry);
            }
            else
            {
                // Refresh label (may now include path hint) and move to top.
                entry = new ModelEntry(label, _models[existingIdx].Path, _models[existingIdx].IsDirectory);
                _models.RemoveAt(existingIdx);
                _models.Insert(0, entry);
            }

            // Force the closed combo to show this item (SelectedItem alone can lag).
            ModelCombo.ItemsSource = null;
            ModelCombo.ItemsSource = _models;
            ModelCombo.SelectedIndex = 0;
            ModelCombo.SelectedItem = entry;
        }
        finally
        {
            _suppressModelSave = false;
        }

        _settings.RememberModelPath(path);
        if (isDir)
            _settings.LastBrowseDirectory = path;
        else
        {
            var parent = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(parent))
                _settings.LastBrowseDirectory = parent;
        }
        _settings.Save();
        SetStatus($"idle — selected {label}");
    }

    // ── Chat UI helpers ──────────────────────────────────────────────────

    private void ClearChat_Click(object sender, RoutedEventArgs e)
    {
        ChatPanel.Children.Clear();
        EmptyHint.Visibility = Visibility.Visible;
        ChatPanel.Children.Add(EmptyHint);
        _streamingTextBlock = null;
        _streamingBubble = null;
        SetStatus("idle");
        SpeedText.Text = "";
    }

    private void HideEmptyHint()
    {
        if (EmptyHint.Visibility == Visibility.Visible)
        {
            EmptyHint.Visibility = Visibility.Collapsed;
            ChatPanel.Children.Remove(EmptyHint);
        }
    }

    /// <summary>
    /// Compact left-aligned log row (no bubbles). Both You and Model stack on the left.
    /// </summary>
    private Border AddBubble(string role, string text, bool streaming = false)
    {
        HideEmptyHint();

        var isUser = role.Equals("user", StringComparison.OrdinalIgnoreCase);

        var row = new Border
        {
            Background = Brushes.Transparent,
            Padding = new Thickness(0, 2, 0, 2),
            HorizontalAlignment = HorizontalAlignment.Stretch,
            Margin = new Thickness(0, 2, 0, 6),
        };

        var stack = new StackPanel();

        var header = new TextBlock
        {
            Text = isUser ? "You" : "Model",
            FontSize = 11,
            FontWeight = FontWeights.SemiBold,
            Foreground = isUser ? AccentBrush : TextSecondaryBrush,
            Margin = new Thickness(0, 0, 0, 2),
        };
        stack.Children.Add(header);

        var body = new TextBlock
        {
            Text = text,
            TextWrapping = TextWrapping.Wrap,
            FontSize = 13,
            Foreground = TextPrimaryBrush,
            FontFamily = new FontFamily("Consolas, Cascadia Mono, Courier New, monospace"),
            LineHeight = 18,
        };
        stack.Children.Add(body);

        if (streaming)
            _streamingTextBlock = body;

        row.Child = stack;
        ChatPanel.Children.Add(row);
        _streamingBubble = streaming ? row : _streamingBubble;
        ScrollToBottom();
        return row;
    }

    private void FinishStreamingBubble()
    {
        _streamingTextBlock = null;
        _streamingBubble = null;
    }

    private void AppendStreaming(string chunk)
    {
        if (_streamingTextBlock == null) return;
        var visible = _stripThink ? FilterThinkTags(chunk) : chunk;
        if (visible.Length == 0) return;
        _streamingTextBlock.Text += visible;
        ScrollToBottom();
    }

    private void ResetThinkFilter()
    {
        _inThinkBlock = false;
        _thinkCarry.Clear();
        var v = Environment.GetEnvironmentVariable("KATALI_THINK")
             ?? Environment.GetEnvironmentVariable("KATALI_ENABLE_THINK");
        _stripThink = !EnvFlagOn(v);
    }

    private static bool EnvFlagOn(string? v)
    {
        if (string.IsNullOrEmpty(v)) return false;
        if (v is "0" or "false" or "False" or "FALSE" or "no" or "No" or "NO")
            return false;
        return true;
    }

    /// <summary>Drop &lt;think&gt;...&lt;/think&gt; spanning chunk boundaries.</summary>
    private string FilterThinkTags(string chunk)
    {
        var s = _thinkCarry.Length > 0 ? _thinkCarry + chunk : chunk;
        _thinkCarry.Clear();
        var sb = new StringBuilder(s.Length);
        const string open = "<think>";
        const string close = "</think>";
        var idx = 0;
        while (idx < s.Length)
        {
            if (_inThinkBlock)
            {
                var c = s.IndexOf(close, idx, StringComparison.Ordinal);
                if (c < 0)
                {
                    var keep = Math.Min(close.Length - 1, s.Length - idx);
                    if (keep > 0 && close.Substring(0, keep) == s.Substring(s.Length - keep))
                        _thinkCarry.Append(s, s.Length - keep, keep);
                    return sb.ToString();
                }
                idx = c + close.Length;
                _inThinkBlock = false;
                continue;
            }

            var o = s.IndexOf(open, idx, StringComparison.Ordinal);
            if (o < 0)
            {
                var keep = 0;
                var maxKeep = Math.Min(open.Length - 1, s.Length - idx);
                for (var k = maxKeep; k > 0; k--)
                {
                    if (open.Substring(0, k) == s.Substring(s.Length - k))
                    {
                        keep = k;
                        break;
                    }
                }
                sb.Append(s, idx, s.Length - idx - keep);
                if (keep > 0) _thinkCarry.Append(s, s.Length - keep, keep);
                break;
            }

            sb.Append(s, idx, o - idx);
            idx = o + open.Length;
            _inThinkBlock = true;
        }
        return sb.ToString();
    }

    private void GenTimer_Tick(object? sender, EventArgs e)
    {
        if (!_isGenerating) return;
        var sec = (DateTime.UtcNow - _genStartedUtc).TotalSeconds;
        // Status strip under the input: live elapsed for this reply.
        StatusText.Text = _seenOutput
            ? $"Generating… {sec:0.0}s"
            : $"Loading… {sec:0.0}s";
    }

    private void StartGenTimer()
    {
        _genStartedUtc = DateTime.UtcNow;
        _lastReplySec = 0;
        _lastToks = "";
        _seenOutput = false;
        StatusText.Text = "Loading… 0.0s";
        SpeedText.Text = "";
        _genTimer.Start();
    }

    private void StopGenTimer(bool stopped)
    {
        _genTimer.Stop();
        _lastReplySec = (DateTime.UtcNow - _genStartedUtc).TotalSeconds;
        if (_lastReplySec < 0) _lastReplySec = 0;
        if (stopped)
        {
            StatusText.Text = $"Stopped · {_lastReplySec:0.0}s";
        }
        else if (!string.IsNullOrEmpty(_lastToks))
        {
            StatusText.Text = $"Last reply: {_lastReplySec:0.0}s · {_lastToks}";
            SpeedText.Text = _lastToks;
        }
        else if (!string.IsNullOrEmpty(_lastSpeed))
        {
            // Prefer a compact tok/s extract from the raw speed line when possible.
            StatusText.Text = $"Last reply: {_lastReplySec:0.0}s";
            SpeedText.Text = _lastSpeed;
        }
        else
        {
            StatusText.Text = $"Last reply: {_lastReplySec:0.0}s";
        }
    }

    private void ScrollToBottom()
    {
        ChatScroll.Dispatcher.InvokeAsync(() =>
        {
            ChatScroll.UpdateLayout();
            ChatScroll.ScrollToEnd();
        }, DispatcherPriority.Background);
    }

    // ── Input handling ───────────────────────────────────────────────────

    private void PromptBox_TextChanged(object sender, TextChangedEventArgs e)
    {
        PromptPlaceholder.Visibility = string.IsNullOrEmpty(PromptBox.Text)
            ? Visibility.Visible
            : Visibility.Collapsed;
    }

    private void PromptBox_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && (Keyboard.Modifiers & ModifierKeys.Shift) == 0)
        {
            e.Handled = true;
            if (!_isGenerating) Send_Click(sender, e);
        }
    }

    // ── Generate / Stop ──────────────────────────────────────────────────

    private async void Send_Click(object sender, RoutedEventArgs e)
    {
        if (_isGenerating) return;

        var prompt = PromptBox.Text.Trim();
        if (string.IsNullOrEmpty(prompt)) return;

        if (ModelCombo.SelectedItem is not ModelEntry model)
        {
            MessageBox.Show("Select a model first.", "Katali Lab",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (!int.TryParse(MaxTokensBox.Text.Trim(), out var maxTokens) || maxTokens < 1)
            maxTokens = 512;

        var exePath = ResolveKataliLabExe();
        if (exePath == null)
        {
            MessageBox.Show(
                "Could not find katali-lab.exe.\nPlace it next to this GUI (or in the parent of a gui\\bin folder).",
                "Katali Lab", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        PersistSelectedModel();

        PromptBox.Text = "";
        AddBubble("user", prompt);
        ResetThinkFilter();
        _streamingBubble = AddBubble("assistant", "", streaming: true);

        _stopRequested = false;
        var runId = ++_runId;
        SetGenerating(true);
        StartGenTimer(); // status strip: Loading… then Generating… with live seconds
        _lastSpeed = "";
        _lastToks = "";

        _cts = new CancellationTokenSource();
        var token = _cts.Token;

        try
        {
            await RunGenerateAsync(exePath, model.Path, prompt, maxTokens, token);
            if (!_stopRequested && runId == _runId)
                StopGenTimer(stopped: false);
        }
        catch (OperationCanceledException)
        {
            // Stop_Click may already have cleared streaming / set "stopped".
            if (runId == _runId)
            {
                if (_streamingTextBlock != null)
                    AppendStreaming("\n[stopped]");
                StopGenTimer(stopped: true);
            }
        }
        catch (Exception ex)
        {
            if (runId == _runId)
            {
                AppendStreaming($"\n[error] {ex.Message}");
                StopGenTimer(stopped: true);
                SetStatus($"Error · {_lastReplySec:0.0}s");
            }
        }
        finally
        {
            // Only the latest run owns the UI / CTS — a superseded run after Stop→re-Send must not clobber.
            if (runId == _runId)
            {
                FinishStreamingBubble();
                // Ensure timer stopped even on odd exit paths
                if (_genTimer.IsEnabled)
                    StopGenTimer(stopped: _stopRequested);
                SetGenerating(false);
                if (!string.IsNullOrEmpty(_lastToks))
                    SpeedText.Text = _lastToks;
                else if (!string.IsNullOrEmpty(_lastSpeed))
                    SpeedText.Text = _lastSpeed;
                _runningProcess = null;
                _cts?.Dispose();
                _cts = null;
                _stopRequested = false;
                PromptBox.Focus();
            }
        }
    }

    private async Task RunGenerateAsync(
        string exePath, string modelPath, string prompt, int maxTokens, CancellationToken token)
    {
        var psi = new ProcessStartInfo
        {
            FileName = exePath,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
            WorkingDirectory = Path.GetDirectoryName(exePath) ?? Environment.CurrentDirectory,
        };
        psi.ArgumentList.Add("generate");
        psi.ArgumentList.Add(modelPath);
        psi.ArgumentList.Add(prompt);
        psi.ArgumentList.Add("--max");
        psi.ArgumentList.Add(maxTokens.ToString());

        var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
        _runningProcess = proc;

        if (!proc.Start())
            throw new InvalidOperationException("Failed to start katali-lab.exe");

        using var reg = token.Register(() => KillProcessTree(proc));

        var stdoutTask = Task.Run(async () =>
        {
            var buf = new char[512];
            var reader = proc.StandardOutput;
            while (true)
            {
                token.ThrowIfCancellationRequested();
                var n = await reader.ReadAsync(buf.AsMemory(0, buf.Length), token).ConfigureAwait(false);
                if (n == 0) break;
                var chunk = new string(buf, 0, n);
                await Dispatcher.InvokeAsync(() =>
                {
                    AdvanceStatusOnOutput();
                    AppendStreaming(chunk);
                });
            }
        }, token);

        var stderrTask = Task.Run(async () =>
        {
            while (true)
            {
                token.ThrowIfCancellationRequested();
                var line = await proc.StandardError.ReadLineAsync(token).ConfigureAwait(false);
                if (line == null) break;
                var trimmed = line.Trim();
                if (trimmed.Length == 0) continue;

                await Dispatcher.InvokeAsync(() => HandleStderrLine(trimmed));
            }
        }, token);

        try
        {
            await Task.WhenAll(proc.WaitForExitAsync(token), stdoutTask, stderrTask);
        }
        catch (OperationCanceledException)
        {
            throw;
        }

        if (token.IsCancellationRequested)
            throw new OperationCanceledException(token);
    }

    /// <summary>
    /// Approximate phase when we only see stdout: loading… → generating…
    /// </summary>
    private void AdvanceStatusOnOutput()
    {
        if (_stopRequested || !_isGenerating) return;
        _seenOutput = true; // first stdout chunk → switch Loading… → Generating…
        // Timer tick owns the "Generating… N.Ns" string; don't clobber it here.
    }

    private void HandleStderrLine(string line)
    {
        if (_stopRequested || !_isGenerating) return;

        // speed: decode_tokens=… tok/s=…   OR   speed: decode_tokens=N decode_sec=S tok/s=R
        if (line.StartsWith("speed:", StringComparison.OrdinalIgnoreCase))
        {
            _lastSpeed = line;
            var toks = TryExtractToks(line);
            if (!string.IsNullOrEmpty(toks))
            {
                _lastToks = toks;
                SpeedText.Text = toks;
            }
            else
            {
                SpeedText.Text = line;
            }
            return;
        }

        // Explicit phase lines from katali-lab (if emitted)
        foreach (var (needle, status) in PhaseHints)
        {
            if (line.Contains(needle, StringComparison.OrdinalIgnoreCase))
            {
                // Don't regress from generating / timer back to loading
                if (status.StartsWith("loading", StringComparison.Ordinal)
                    && (StatusText.Text.StartsWith("Generating", StringComparison.Ordinal)
                        || StatusText.Text is "done" or "prefill…"))
                    return;
                if (status == "prefill…"
                    && StatusText.Text.StartsWith("Generating", StringComparison.Ordinal))
                    return;
                // Never overwrite live timer text with a bare "generating…"
                if (StatusText.Text.StartsWith("Generating…", StringComparison.Ordinal)
                    && status.StartsWith("generating", StringComparison.OrdinalIgnoreCase))
                    return;
                SetStatus(status);
                return;
            }
        }

        // Any other stderr activity while still in loading → bump toward generating
        if (StatusText.Text is "loading model…" or "loading…")
            SetStatus("Generating…");
    }

    private static string? TryExtractToks(string speedLine)
    {
        // Match tok/s=3.14 or tok/s:3.14
        var m = System.Text.RegularExpressions.Regex.Match(
            speedLine, @"tok/s[=:\s]+([0-9]+(?:\.[0-9]+)?)",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        if (m.Success) return $"{m.Groups[1].Value} tok/s";
        return null;
    }

    private void Stop_Click(object sender, RoutedEventArgs e)
    {
        if (!_isGenerating && _runningProcess == null) return;

        _stopRequested = true;
        _runId++; // invalidate in-flight Send finally/catch

        // Immediate UI feedback — don't wait for process teardown.
        StopBtn.IsEnabled = false;
        SetStatus("stopping…");

        try { _cts?.Cancel(); } catch { /* ignore */ }
        KillRunningProcess();
        try { _cts?.Dispose(); } catch { /* ignore */ }
        _cts = null;

        // Mark transcript, clear streaming chrome, re-enable Send — don't leave UI stuck.
        if (_streamingTextBlock != null &&
            !_streamingTextBlock.Text.EndsWith("[stopped]", StringComparison.Ordinal))
        {
            AppendStreaming("\n[stopped]");
        }
        FinishStreamingBubble();
        StopGenTimer(stopped: true);
        SetGenerating(false);
        PromptBox.Focus();
    }

    private void KillRunningProcess()
    {
        var p = _runningProcess;
        if (p == null) return;
        KillProcessTree(p);
        _runningProcess = null;
    }

    private static void KillProcessTree(Process proc)
    {
        try
        {
            if (proc.HasExited) return;
            try
            {
                var kill = new ProcessStartInfo
                {
                    FileName = "taskkill",
                    Arguments = $"/PID {proc.Id} /T /F",
                    CreateNoWindow = true,
                    UseShellExecute = false,
                };
                using var k = Process.Start(kill);
                k?.WaitForExit(2000);
            }
            catch
            {
                try { proc.Kill(entireProcessTree: true); } catch { /* ignore */ }
            }
        }
        catch
        {
            /* ignore */
        }
    }

    private void SetGenerating(bool on)
    {
        _isGenerating = on;
        SendBtn.IsEnabled = !on;
        ModelCombo.IsEnabled = !on;
        BrowseModelBtn.IsEnabled = !on;
        BrowseFolderBtn.IsEnabled = !on;
        RefreshModelsBtn.IsEnabled = !on;
        MaxTokensBox.IsEnabled = !on;
        ClearChatBtn.IsEnabled = !on;
        StopBtn.IsEnabled = on;
    }

    private void SetStatus(string text) => StatusText.Text = text;

    // ── Resolve katali-lab.exe ───────────────────────────────────────────

    /// <summary>
    /// Same directory as the GUI exe, then walk parents looking for katali-lab.exe
    /// (covers gui\bin\Release\net9.0-windows\win-x64\publish layouts).
    /// </summary>
    private static string? ResolveKataliLabExe()
    {
        var start = AppContext.BaseDirectory;
        var dir = new DirectoryInfo(start);
        for (var i = 0; i < 8 && dir != null; i++, dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, "katali-lab.exe");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }
}
