using System.Collections.ObjectModel;
using Microsoft.Win32;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
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
        (@"C:\models\qwen38-27b\Qwen3.8-27B-Q4_K_M.gguf", "Qwen3.8-27B Q4_K_M (dense, CPU)"),
    ];

    // stderr phase keywords katali-lab may emit (case-insensitive substring match)
    private static readonly (string Needle, string Status)[] PhaseHints =
    [
        ("loading model", "loading model..."),
        ("loading", "loading model..."),
        ("prefill", "prefill..."),
        ("prefilling", "prefill..."),
        ("generating", "generating..."),
        ("decode", "generating..."),
        ("done", "done"),
        ("finished", "done"),
    ];

    private readonly ObservableCollection<ModelEntry> _models = new();
    private readonly ObservableCollection<SessionListItem> _sessionItems = new();
    private readonly GuiSettings _settings;
    private Process? _runningProcess;
    private CancellationTokenSource? _cts;
    private TextBlock? _streamingTextBlock;
    private Border? _streamingBubble;
    private bool _isGenerating;
    private bool _stopRequested;
    private int _runId;
    private bool _suppressModelSave;
    private bool _suppressSessionSelect;
    private string _lastSpeed = "";
    private string? _screenshotPath;

    // Multi-turn chat history (user/assistant turns sent to the engine)
    private readonly List<ChatMessage> _history = new();
    private ChatSession _currentSession = null!;
    private ChatPromptBuilder.ModelFamily _lastFamily = ChatPromptBuilder.ModelFamily.Qwen35Moe;
    private string? _promptTempFile;

    // Generation wall-clock timer (compare with/without think)
    private readonly DispatcherTimer _genTimer = new() { Interval = TimeSpan.FromMilliseconds(100) };
    private DateTime _genStartedUtc;
    private double _lastReplySec;
    private string _lastToks = "";
    private bool _seenOutput;

    // Residual <think>...</think> and ChatML marker strip across stdout chunks
    private bool _stripThink = true;
    private bool _inThinkBlock;
    private readonly StringBuilder _thinkCarry = new();
    private readonly StringBuilder _specialCarry = new();
    private bool _streamHitImEnd;

    private static readonly Brush UserBubbleBrush = new SolidColorBrush(Color.FromRgb(0x1E, 0x3A, 0x4C));
    private static readonly Brush AssistantBubbleBrush = new SolidColorBrush(Color.FromRgb(0x1C, 0x23, 0x30));
    private static readonly Brush TextPrimaryBrush = new SolidColorBrush(Color.FromRgb(0xE6, 0xED, 0xF3));
    private static readonly Brush TextSecondaryBrush = new SolidColorBrush(Color.FromRgb(0x8B, 0x94, 0x9E));
    private static readonly Brush AccentBrush = new SolidColorBrush(Color.FromRgb(0x2D, 0xD4, 0xBF));
    private static readonly Brush GeneratingBrush = new SolidColorBrush(Color.FromRgb(0x6E, 0x76, 0x81));

    /// <summary>Tag attached to each transcript row for context-menu actions.</summary>
    private sealed class MessageRowInfo
    {
        public required string Role { get; init; }
        public required TextBlock Body { get; init; }
        public required Border Row { get; init; }
        public string PlainText { get; set; } = "";
    }

    /// <summary>Sidebar list row bound to SessionList.</summary>
    public sealed class SessionListItem
    {
        public string Id { get; set; } = "";
        public string Title { get; set; } = "New chat";
        public DateTime UpdatedUtc { get; set; }
        public string UpdatedLocal =>
            UpdatedUtc == default
                ? ""
                : UpdatedUtc.ToLocalTime().ToString("MMM d | h:mm tt");
    }

    public MainWindow()
    {
        InitializeComponent();
        _settings = GuiSettings.Load();
        ModelCombo.ItemsSource = _models;
        SessionList.ItemsSource = _sessionItems;
        ParseScreenshotArg();
        Loaded += OnLoaded;
        Closed += OnClosed;
        _genTimer.Tick += GenTimer_Tick;
    }

    private void OnLoaded(object sender, RoutedEventArgs e)
    {
        MaxTokensBox.Text = _settings.MaxTokens > 0 ? _settings.MaxTokens.ToString() : "512";
        RefreshModels();
        if (ModelCombo.SelectedItem is ModelEntry me)
            SyncThinkingDefaultForModel(me);
        InitSessions();
        UpdateChatStatus();
        UpdateContextMeter();
        PromptBox.Focus();
        if (!string.IsNullOrEmpty(_screenshotPath))
        {
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
        PersistMaxTokensFromBox();
        SaveCurrentSession(persistSettings: true);
        CleanupPromptTempFile();
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

    // - Sessions -

    private void InitSessions()
    {
        RefreshSessionList();
        ChatSession? session = null;
        if (!string.IsNullOrEmpty(_settings.LastSessionId))
            session = ChatSessionStore.Load(_settings.LastSessionId!);

        if (session == null)
        {
            if (_sessionItems.Count > 0)
                session = ChatSessionStore.Load(_sessionItems[0].Id);
        }

        if (session == null)
            session = ChatSessionStore.CreateNew();

        LoadSessionIntoUi(session, selectInList: true);
    }

    private void RefreshSessionList()
    {
        _suppressSessionSelect = true;
        try
        {
            var selectedId = _currentSession?.Id;
            _sessionItems.Clear();
            foreach (var s in ChatSessionStore.LoadAll())
            {
                _sessionItems.Add(new SessionListItem
                {
                    Id = s.Id,
                    Title = string.IsNullOrWhiteSpace(s.Title) ? "New chat" : s.Title,
                    UpdatedUtc = s.UpdatedUtc,
                });
            }

            if (!string.IsNullOrEmpty(selectedId))
            {
                var match = _sessionItems.FirstOrDefault(i => i.Id == selectedId);
                if (match != null)
                    SessionList.SelectedItem = match;
            }
        }
        finally
        {
            _suppressSessionSelect = false;
        }
    }

    private void LoadSessionIntoUi(ChatSession session, bool selectInList)
    {
        _currentSession = session;
        _history.Clear();
        foreach (var m in session.Messages ?? Enumerable.Empty<ChatSessionMessage>())
        {
            var role = (m.Role ?? "").Trim().ToLowerInvariant();
            if (role is not ("user" or "assistant")) continue;
            _history.Add(new ChatMessage(role, m.Content ?? ""));
        }

        RebuildChatFromHistory();
        _settings.LastSessionId = session.Id;
        _settings.Save();

        if (selectInList)
        {
            _suppressSessionSelect = true;
            try
            {
                var item = _sessionItems.FirstOrDefault(i => i.Id == session.Id);
                if (item == null)
                {
                    item = new SessionListItem
                    {
                        Id = session.Id,
                        Title = string.IsNullOrWhiteSpace(session.Title) ? "New chat" : session.Title,
                        UpdatedUtc = session.UpdatedUtc,
                    };
                    _sessionItems.Insert(0, item);
                }
                SessionList.SelectedItem = item;
            }
            finally
            {
                _suppressSessionSelect = false;
            }
        }

        UpdateChatStatus();
    }

    private void SaveCurrentSession(bool persistSettings = false)
    {
        if (_currentSession == null) return;

        _currentSession.Messages = _history
            .Where(m => m.IsUser || m.IsAssistant)
            .Select(m => new ChatSessionMessage { Role = m.Role, Content = m.Content ?? "" })
            .ToList();

        MaybeAutoTitle();
        ChatSessionStore.Save(_currentSession);

        var item = _sessionItems.FirstOrDefault(i => i.Id == _currentSession.Id);
        if (item != null)
        {
            item.Title = _currentSession.Title;
            item.UpdatedUtc = _currentSession.UpdatedUtc;
            // Force ListBox refresh for bound props without INotifyPropertyChanged
            var idx = _sessionItems.IndexOf(item);
            if (idx >= 0)
            {
                _suppressSessionSelect = true;
                try
                {
                    _sessionItems.RemoveAt(idx);
                    _sessionItems.Insert(0, item);
                    SessionList.SelectedItem = item;
                }
                finally
                {
                    _suppressSessionSelect = false;
                }
            }
        }
        else
        {
            _sessionItems.Insert(0, new SessionListItem
            {
                Id = _currentSession.Id,
                Title = _currentSession.Title,
                UpdatedUtc = _currentSession.UpdatedUtc,
            });
        }

        _settings.LastSessionId = _currentSession.Id;
        if (persistSettings)
            _settings.Save();
        else
            _settings.Save();
    }

    private void MaybeAutoTitle()
    {
        if (_currentSession == null) return;
        var title = _currentSession.Title ?? "";
        if (!string.Equals(title, "New chat", StringComparison.OrdinalIgnoreCase)
            && !string.IsNullOrWhiteSpace(title))
            return;

        var firstUser = _history.FirstOrDefault(m => m.IsUser);
        if (firstUser == null) return;
        _currentSession.Title = ChatSessionStore.TitleFromFirstUser(firstUser.Content);
    }

    private void NewChat_Click(object sender, RoutedEventArgs e)
    {
        if (_isGenerating) return;
        SaveCurrentSession();
        var session = ChatSessionStore.CreateNew();
        RefreshSessionList();
        LoadSessionIntoUi(session, selectInList: true);
        PromptBox.Focus();
    }

    private void SessionList_PreviewMouseRightButtonDown(object sender, MouseButtonEventArgs e)
    {
        var item = ItemsControl.ContainerFromElement(SessionList, e.OriginalSource as DependencyObject) as ListBoxItem;
        if (item != null)
            item.IsSelected = true;
    }
    private void SessionList_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_suppressSessionSelect) return;
        if (SessionList.SelectedItem is not SessionListItem item) return;
        if (_currentSession != null && item.Id == _currentSession.Id) return;
        if (_isGenerating)
        {
            // Revert selection while generating
            _suppressSessionSelect = true;
            try
            {
                var cur = _sessionItems.FirstOrDefault(i => i.Id == _currentSession?.Id);
                SessionList.SelectedItem = cur;
            }
            finally { _suppressSessionSelect = false; }
            return;
        }

        SaveCurrentSession();
        var loaded = ChatSessionStore.Load(item.Id);
        if (loaded == null)
        {
            MessageBox.Show("Could not load that chat session.", "Katali Lab",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            RefreshSessionList();
            return;
        }
        LoadSessionIntoUi(loaded, selectInList: false);
        PromptBox.Focus();
    }

    private void RenameSession_Click(object sender, RoutedEventArgs e)
    {
        if (SessionList.SelectedItem is not SessionListItem item) return;
        var current = item.Id == _currentSession?.Id
            ? _currentSession!
            : ChatSessionStore.Load(item.Id);
        if (current == null) return;

        var next = PromptForText("Rename chat", "Title:", current.Title);
        if (next == null) return;
        next = next.Trim();
        if (string.IsNullOrEmpty(next)) return;

        current.Title = next;
        ChatSessionStore.Save(current);
        if (_currentSession != null && current.Id == _currentSession.Id)
            _currentSession.Title = next;
        item.Title = next;
        RefreshSessionList();
        _suppressSessionSelect = true;
        try
        {
            SessionList.SelectedItem = _sessionItems.FirstOrDefault(i => i.Id == current.Id);
        }
        finally { _suppressSessionSelect = false; }
    }

    private void DeleteSessionItem(SessionListItem item)
    {
        if (item == null) return;
        var r = MessageBox.Show(
            $"Delete chat \"{item.Title}\"?",
            "Katali Lab", MessageBoxButton.YesNo, MessageBoxImage.Warning);
        if (r != MessageBoxResult.Yes) return;

        var deletingCurrent = _currentSession != null && item.Id == _currentSession.Id;
        // Do not SaveCurrentSession for the deleted id — that would recreate the JSON.
        ChatSessionStore.Delete(item.Id);
        _sessionItems.Remove(item);

        if (deletingCurrent)
        {
            var next = _sessionItems.FirstOrDefault();
            if (next != null)
            {
                var loaded = ChatSessionStore.Load(next.Id) ?? ChatSessionStore.CreateNew();
                if (loaded.Id != next.Id)
                    RefreshSessionList();
                // LoadSessionIntoUi updates LastSessionId and saves settings.
                LoadSessionIntoUi(loaded, selectInList: true);
            }
            else
            {
                var fresh = ChatSessionStore.CreateNew();
                RefreshSessionList();
                LoadSessionIntoUi(fresh, selectInList: true);
            }
        }
        else if (_settings.LastSessionId == item.Id)
        {
            _settings.LastSessionId = _currentSession?.Id;
            _settings.Save();
        }
    }

    private void DeleteSession_Click(object sender, RoutedEventArgs e)
    {
        if (SessionList.SelectedItem is not SessionListItem item) return;
        DeleteSessionItem(item);
    }

    private void DeleteSessionButton_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        // Keep ListBox from changing selection when clicking the row's X.
        // Handled=true also prevents Button.Click from firing, so delete must run here.
        e.Handled = true;
        if (sender is not Button btn) return;
        SessionListItem? item = btn.Tag as SessionListItem;
        if (item == null && btn.DataContext is SessionListItem dc)
            item = dc;
        if (item == null) return;
        DeleteSessionItem(item);
    }

    private void DeleteSessionButton_Click(object sender, RoutedEventArgs e)
    {
        // Fallback only: with Preview Handled=true this path does not run.
        e.Handled = true;
        if (sender is not Button btn) return;
        SessionListItem? item = btn.Tag as SessionListItem;
        if (item == null && btn.DataContext is SessionListItem dc)
            item = dc;
        if (item == null) return;
        DeleteSessionItem(item);
    }
    private string? PromptForText(string title, string label, string initial)
    {
        var dlg = new Window
        {
            Title = title,
            Width = 380,
            Height = 150,
            WindowStartupLocation = WindowStartupLocation.CenterOwner,
            Owner = this,
            ResizeMode = ResizeMode.NoResize,
            Background = (Brush)FindResource("BgPanel"),
            ShowInTaskbar = false,
        };

        var root = new Grid { Margin = new Thickness(16) };
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

        var lbl = new TextBlock
        {
            Text = label,
            Foreground = TextSecondaryBrush,
            Margin = new Thickness(0, 0, 0, 6),
            FontSize = 12,
        };
        Grid.SetRow(lbl, 0);
        root.Children.Add(lbl);

        var box = new TextBox
        {
            Text = initial,
            Background = (Brush)FindResource("BgInput"),
            Foreground = TextPrimaryBrush,
            BorderBrush = (Brush)FindResource("BorderSubtle"),
            Padding = new Thickness(8, 6, 8, 6),
            CaretBrush = AccentBrush,
            FontSize = 13,
        };
        Grid.SetRow(box, 1);
        root.Children.Add(box);

        var buttons = new StackPanel
        {
            Orientation = Orientation.Horizontal,
            HorizontalAlignment = HorizontalAlignment.Right,
            Margin = new Thickness(0, 14, 0, 0),
        };
        var cancel = new Button
        {
            Content = "Cancel",
            Style = (Style)FindResource("GhostButton"),
            Margin = new Thickness(0, 0, 8, 0),
            IsCancel = true,
            MinWidth = 72,
        };
        cancel.Click += (_, _) => { dlg.DialogResult = false; };
        var ok = new Button
        {
            Content = "OK",
            Style = (Style)FindResource("PrimaryButton"),
            IsDefault = true,
            MinWidth = 72,
        };
        ok.Click += (_, _) => { dlg.DialogResult = true; };
        buttons.Children.Add(cancel);
        buttons.Children.Add(ok);
        Grid.SetRow(buttons, 2);
        root.Children.Add(buttons);

        dlg.Content = root;
        dlg.Loaded += (_, _) => { box.SelectAll(); box.Focus(); };
        return dlg.ShowDialog() == true ? box.Text : null;
    }

    // - Model discovery / persistence -

    private void RefreshModels_Click(object sender, RoutedEventArgs e) => RefreshModels();

    private void ModelCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_suppressModelSave) return;
        if (ModelCombo.SelectedItem is ModelEntry m)
        {
            PersistSelectedModel();
            SyncThinkingDefaultForModel(m);
            MaybeSoftSuggest35B(m);
            UpdateChatStatus();
            UpdateContextMeter();
        }
    }

    private void SyncThinkingDefaultForModel(ModelEntry model)
    {
        var family = ChatPromptBuilder.DetectFamily(model.Path + " " + model.Label);
        var newDefault = family == ChatPromptBuilder.ModelFamily.Qwen3Moe;
        var oldDefault = _lastFamily == ChatPromptBuilder.ModelFamily.Qwen3Moe;
        if (_settings.EnableThinking == oldDefault)
        {
            _settings.EnableThinking = newDefault;
            _settings.Save();
        }
        _lastFamily = family;
    }

    private void PersistMaxTokensFromBox()
    {
        if (int.TryParse(MaxTokensBox.Text.Trim(), out var max) && max > 0)
            _settings.MaxTokens = max;
        _settings.Save();
    }

    private void Settings_Click(object sender, RoutedEventArgs e)
    {
        PersistMaxTokensFromBox();
        var dlg = new SettingsWindow(_settings) { Owner = this };
        if (dlg.ShowDialog() == true)
        {
            MaxTokensBox.Text = _settings.MaxTokens.ToString();
            UpdateChatStatus();
            UpdateContextMeter();
        }
    }

    private void UpdateChatStatus()
    {
        if (_isGenerating) return;
        var n = _history.Count;
        if (n == 0)
            SetStatus("Ready");
        else
            SetStatus($"Chat | {n} messages");
        UpdateContextMeter();
    }

    private void MaybeSoftSuggest35B(ModelEntry model)
    {
        if (!GuiSettings.Is35BModel(model.Path + " " + model.Label)) return;
        if (_settings.SoftFill35BUnsetFields())
            _settings.Save();
    }

    /// <summary>Approximate ChatML prompt size (chars/4) vs soft 8k budget.</summary>
    private void UpdateContextMeter()
    {
        if (ContextMeterText == null) return;
        var tokens = EstimatePromptTokens();
        var budget = GuiSettings.ContextSoftBudgetTokens;
        ContextMeterText.Text = $"~{FormatTokenCount(tokens)} / {FormatTokenCount(budget)} tokens in prompt";
        if (tokens >= GuiSettings.ContextWarnTokens)
        {
            ContextMeterText.Foreground = new SolidColorBrush(Color.FromRgb(0xF8, 0x51, 0x49));
            ContextMeterText.ToolTip =
                $"Prompt estimate is high (~{tokens} tokens). Long history slows 35B laptop runs - clear chat or start a new session.";
        }
        else
        {
            ContextMeterText.Foreground = TextSecondaryBrush;
            ContextMeterText.ToolTip =
                "Approximate tokens in the ChatML prompt (characters / 4). Soft budget 8192.";
        }
    }

    private int EstimatePromptTokens()
    {
        var draft = (PromptBox?.Text ?? "").Trim();
        var msgs = new List<ChatMessage>(_history);
        if (!string.IsNullOrEmpty(draft))
            msgs.Add(new ChatMessage("user", draft));
        var wantThinking = _settings.EnableThinking;
        var fam = ModelCombo.SelectedItem is ModelEntry me
            ? ChatPromptBuilder.DetectFamily(me.Path + " " + me.Label)
            : ChatPromptBuilder.DetectFamily(_settings.LastModelPath);
        var prompt = ChatPromptBuilder.Build(
            msgs,
            _settings.SystemPrompt,
            wantThinking,
            includeAssistantHeader: true,
            family: fam);
        return Math.Max(0, prompt.Length / 4);
    }

    private static string FormatTokenCount(int n)
    {
        if (n >= 1000) return $"{n / 1000.0:0.#}k";
        return n.ToString();
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
        var previousOk = !string.IsNullOrEmpty(previous) && ModelPathExists(previous!);
        _suppressModelSave = true;
        try
        {
            _models.Clear();

            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var scanned = new List<ModelEntry>();

            // Known product defaults first (35B primary), then scan C:\models, then recents.
            foreach (var (path, label) in KnownDefaults)
            {
                if (File.Exists(path))
                    TryCollect(scanned, path, label, isDir: false, seen);
                else if (Directory.Exists(path) && DirectoryContainsGguf(path))
                    TryCollect(scanned, path, label, isDir: true, seen);
            }

            if (Directory.Exists(ModelsRoot))
            {
                foreach (var file in Directory.EnumerateFiles(ModelsRoot, "*.gguf", SearchOption.TopDirectoryOnly))
                    TryCollect(scanned, file, LabelFromGguf(file), isDir: false, seen);

                foreach (var dir in Directory.EnumerateDirectories(ModelsRoot))
                {
                    // Skip hidden/system-ish dirs like .cache
                    var dirName = Path.GetFileName(dir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
                    if (dirName.StartsWith(".", StringComparison.Ordinal)) continue;

                    CollectGgufsFromModelDir(scanned, dir, seen);
                }
            }

            foreach (var recent in _settings.RecentModelPaths.ToList())
                TryCollectExisting(scanned, recent, seen);

            if (!string.IsNullOrEmpty(_settings.LastModelPath))
                TryCollectExisting(scanned, _settings.LastModelPath!, seen);

            // Stable order: 35B models first (primary path wins), then everything else by label.
            foreach (var m in scanned
                         .OrderBy(m => GuiSettings.Is35BModel(m.Path + " " + m.Label) ? 0 : 1)
                         .ThenBy(m => string.Equals(m.Path, GuiSettings.Primary35BPath, StringComparison.OrdinalIgnoreCase) ? 0 : 1)
                         .ThenBy(m => m.Label, StringComparer.OrdinalIgnoreCase))
                _models.Add(m);

            if (_models.Count == 0)
            {
                SetStatus("No models yet - download a 35B GGUF, then Browse");
                if (_history.Count == 0) { ApplyEmptyHintContent(); EmptyHint.Visibility = Visibility.Visible; if (!ChatPanel.Children.Contains(EmptyHint)) ChatPanel.Children.Add(EmptyHint); }
                UpdateContextMeter();
                return;
            }

            ModelEntry? match = null;
            if (previousOk)
                match = _models.FirstOrDefault(m =>
                    string.Equals(m.Path, previous, StringComparison.OrdinalIgnoreCase));
            match ??= _models.FirstOrDefault(m =>
                string.Equals(m.Path, GuiSettings.Primary35BPath, StringComparison.OrdinalIgnoreCase));
            match ??= _models.FirstOrDefault(m => GuiSettings.Is35BModel(m.Path + " " + m.Label));
            ModelCombo.SelectedItem = match ?? _models[0];
            SetStatus($"Ready | {_models.Count} model{(_models.Count == 1 ? "" : "s")}");
            UpdateContextMeter();
        }
        finally
        {
            _suppressModelSave = false;
            PersistSelectedModel();
            if (ModelCombo.SelectedItem is ModelEntry selected)
                MaybeSoftSuggest35B(selected);
        }
    }

    private static bool ModelPathExists(string path) =>
        File.Exists(path) || (Directory.Exists(path) && DirectoryContainsGguf(path));

    private static void TryCollect(
        List<ModelEntry> list, string path, string label, bool isDir, HashSet<string> seen)
    {
        if (!seen.Add(path)) return;
        list.Add(new ModelEntry(label, path, isDir));
    }

    private void TryCollectExisting(List<ModelEntry> list, string path, HashSet<string> seen)
    {
        if (string.IsNullOrWhiteSpace(path) || seen.Contains(path)) return;
        if (File.Exists(path))
            TryCollect(list, path, LabelFromGguf(path), isDir: false, seen);
        else if (Directory.Exists(path) && DirectoryContainsGguf(path))
            TryCollect(list, path, LabelFromDir(path), isDir: true, seen);
    }

    /// <summary>
    /// Collect GGUFs under a C:\models child folder.
    /// Single file => file entry (clearer label); multi-shard => directory entry.
    /// Also peeks one level of nested subfolders (e.g. vendor/model/).
    /// </summary>
    private static void CollectGgufsFromModelDir(List<ModelEntry> list, string dir, HashSet<string> seen)
    {
        string[] ggufs;
        try { ggufs = Directory.GetFiles(dir, "*.gguf", SearchOption.TopDirectoryOnly); }
        catch { return; }

        if (ggufs.Length == 1)
        {
            TryCollect(list, ggufs[0], LabelFromGguf(ggufs[0]), isDir: false, seen);
            return;
        }
        if (ggufs.Length > 1)
        {
            TryCollect(list, dir, LabelFromDir(dir), isDir: true, seen);
            return;
        }

        try
        {
            foreach (var sub in Directory.EnumerateDirectories(dir))
            {
                var subName = Path.GetFileName(sub.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
                if (subName.StartsWith(".", StringComparison.Ordinal)) continue;
                string[] nested;
                try { nested = Directory.GetFiles(sub, "*.gguf", SearchOption.TopDirectoryOnly); }
                catch { continue; }
                if (nested.Length == 1)
                    TryCollect(list, nested[0], LabelFromGguf(nested[0]), isDir: false, seen);
                else if (nested.Length > 1)
                    TryCollect(list, sub, LabelFromDir(sub), isDir: true, seen);
            }
        }
        catch { /* ignore */ }
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

    // - Browse any drive -

    private void BrowseModel_Click(object sender, RoutedEventArgs e)
    {
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

        var parent = Path.GetDirectoryName(chosen);
        if (!string.IsNullOrEmpty(parent) && CountGgufSiblings(parent) > 1)
        {
            AcceptBrowsedPath(parent, isDir: true);
            SetStatus($"Ready | multi-shard folder: {Path.GetFileName(parent.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))}");
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
        SetStatus($"Ready | folder: {Path.GetFileName(folder.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))}");
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
        try
        {
            var root = Path.GetPathRoot(path) ?? "";
            if (!path.StartsWith(ModelsRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)
                && !string.Equals(path, ModelsRoot, StringComparison.OrdinalIgnoreCase))
            {
                var tip = path.Length <= 56 ? path : ("..." + path.Substring(path.Length - 54));
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
                entry = new ModelEntry(label, _models[existingIdx].Path, _models[existingIdx].IsDirectory);
                _models.RemoveAt(existingIdx);
                _models.Insert(0, entry);
            }

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
        if (ModelCombo.SelectedItem is ModelEntry selected)
            SyncThinkingDefaultForModel(selected);
        UpdateChatStatus();
        SetStatus($"Ready | {label}");
    }

    // - Chat UI helpers -

    private void ClearChat_Click(object sender, RoutedEventArgs e)
    {
        if (_isGenerating) return;
        if (_history.Count > 2)
        {
            var r = MessageBox.Show(
                "Clear the conversation history in this session?",
                "Katali Lab", MessageBoxButton.YesNo, MessageBoxImage.Question);
            if (r != MessageBoxResult.Yes) return;
        }

        _history.Clear();
        RebuildChatFromHistory();
        SpeedText.Text = "";
        SaveCurrentSession();
        UpdateChatStatus();
    }

    private void RebuildChatFromHistory()
    {
        ChatPanel.Children.Clear();
        _streamingTextBlock = null;
        _streamingBubble = null;

        if (_history.Count == 0)
        {
            ApplyEmptyHintContent();
            EmptyHint.Visibility = Visibility.Visible;
            if (!ChatPanel.Children.Contains(EmptyHint))
                ChatPanel.Children.Add(EmptyHint);
            return;
        }

        EmptyHint.Visibility = Visibility.Collapsed;
        foreach (var msg in _history)
            AddBubble(msg.Role, msg.Content ?? "");
    }


    private void ApplyEmptyHintContent()
    {
        EmptyHint.Inlines.Clear();
        EmptyHint.Text = null;
        if (_models.Count == 0)
        {
            EmptyHint.Inlines.Add(new Run("Add a model to get started."));
            EmptyHint.Inlines.Add(new LineBreak());
            EmptyHint.Inlines.Add(new LineBreak());
            EmptyHint.Inlines.Add(new Run("Recommended: download a Qwen3.6-35B-A3B Q4_K_M GGUF, then use Browse... or Folder..."));
            EmptyHint.Inlines.Add(new LineBreak());
            EmptyHint.Inlines.Add(new Run("Optional: put .gguf files in C:\\models and click Refresh models."));
            EmptyHint.Inlines.Add(new LineBreak());
            EmptyHint.Inlines.Add(new LineBreak());
            EmptyHint.Inlines.Add(new Run("Model weights are not bundled - you download them separately."));
        }
        else
        {
            EmptyHint.Inlines.Add(new Run("Select a model and send a message to begin."));
            EmptyHint.Inlines.Add(new LineBreak());
            EmptyHint.Inlines.Add(new LineBreak());
            EmptyHint.Inlines.Add(new Run("Qwen3.6-35B-A3B is recommended when you have it. Right-click messages for Copy, Retry, or Edit."));
        }
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
    /// Compact left-aligned transcript row. User: monospace. Assistant: Markdown when finished.
    /// Streaming assistant stays plain text until FinishStreamingBubble.
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
            Text = isUser ? "You" : "Assistant",
            FontSize = 11,
            FontWeight = FontWeights.SemiBold,
            Foreground = isUser ? AccentBrush : TextSecondaryBrush,
            Margin = new Thickness(0, 0, 0, 2),
        };
        stack.Children.Add(header);

        var body = new TextBlock
        {
            TextWrapping = TextWrapping.Wrap,
            FontSize = 13,
            Foreground = TextPrimaryBrush,
            LineHeight = 20,
        };

        if (isUser || streaming)
        {
            body.Text = text;
            body.FontFamily = new FontFamily("Consolas, Cascadia Mono, Courier New, monospace");
            body.LineHeight = 18;
        }
        else
        {
            SimpleMarkdown.RenderTo(body, text, TextPrimaryBrush);
        }

        stack.Children.Add(body);

        if (streaming)
            _streamingTextBlock = body;

        row.Child = stack;

        var info = new MessageRowInfo
        {
            Role = isUser ? "user" : "assistant",
            Body = body,
            Row = row,
            PlainText = text ?? "",
        };
        row.Tag = info;
        row.ContextMenu = BuildMessageContextMenu(info);

        ChatPanel.Children.Add(row);
        _streamingBubble = streaming ? row : _streamingBubble;
        ScrollToBottom();
        return row;
    }

    private ContextMenu BuildMessageContextMenu(MessageRowInfo info)
    {
        var menu = new ContextMenu();
        var copy = new MenuItem { Header = "Copy" };
        copy.Click += (_, _) =>
        {
            try
            {
                var plain = !string.IsNullOrEmpty(info.PlainText)
                    ? info.PlainText
                    : (info.Body.Text ?? "");
                Clipboard.SetText(plain);
            }
            catch { /* clipboard busy */ }
        };
        menu.Items.Add(copy);

        var retry = new MenuItem { Header = "Retry" };
        retry.Click += async (_, _) => await RetryLastAssistantAsync(info);
        menu.Items.Add(retry);

        var edit = new MenuItem { Header = "Edit last user" };
        edit.Click += (_, _) => EditUserMessage(info);
        menu.Items.Add(edit);

        menu.Opened += (_, _) =>
        {
            var generating = _isGenerating;
            copy.IsEnabled = !string.IsNullOrEmpty(info.PlainText) || !string.IsNullOrEmpty(info.Body.Text);
            retry.Visibility = CanRetry(info) && !generating ? Visibility.Visible : Visibility.Collapsed;
            edit.Visibility = CanEditUser(info) && !generating ? Visibility.Visible : Visibility.Collapsed;
        };

        return menu;
    }

    private bool CanRetry(MessageRowInfo info)
    {
        if (!info.Role.Equals("assistant", StringComparison.OrdinalIgnoreCase)) return false;
        if (_history.Count == 0 || !_history[^1].IsAssistant) return false;
        // Must be the last assistant UI row (or the streaming/finished last row)
        return IsLastAssistantRow(info);
    }

    private bool IsLastAssistantRow(MessageRowInfo info)
    {
        MessageRowInfo? lastAsst = null;
        foreach (var child in ChatPanel.Children)
        {
            if (child is Border b && b.Tag is MessageRowInfo mi
                && mi.Role.Equals("assistant", StringComparison.OrdinalIgnoreCase))
                lastAsst = mi;
        }
        return lastAsst == info;
    }

    private bool CanEditUser(MessageRowInfo info)
    {
        // Edit available on the most recent user row (clicked user if it is that one,
        // or from any row when targeting "Edit last user" via that user row).
        var lastUserRow = FindLastUserRow();
        if (lastUserRow == null) return false;
        if (info.Role.Equals("user", StringComparison.OrdinalIgnoreCase))
            return info == lastUserRow;
        // Allow Edit from the following assistant row that targets the last user
        if (info.Role.Equals("assistant", StringComparison.OrdinalIgnoreCase)
            && IsLastAssistantRow(info))
            return true;
        return false;
    }

    private MessageRowInfo? FindLastUserRow()
    {
        MessageRowInfo? last = null;
        foreach (var child in ChatPanel.Children)
        {
            if (child is Border b && b.Tag is MessageRowInfo mi
                && mi.Role.Equals("user", StringComparison.OrdinalIgnoreCase))
                last = mi;
        }
        return last;
    }

    private async Task RetryLastAssistantAsync(MessageRowInfo info)
    {
        if (_isGenerating || !CanRetry(info)) return;
        if (!TryPrepareGenerate(out var exePath, out var model, out var maxTokens, out var wantThinking, out var family))
            return;

        // Remove last assistant from history + UI (keep last user)
        if (_history.Count == 0 || !_history[^1].IsAssistant) return;
        _history.RemoveAt(_history.Count - 1);
        RemoveChatRow(info.Row);
        if (_history.Count == 0 || !_history[^1].IsUser)
        {
            RebuildChatFromHistory();
            SaveCurrentSession();
            return;
        }

        ResetThinkFilter(wantThinking);
        _streamingBubble = AddBubble("assistant", "", streaming: true);
        await ExecuteGenerateAsync(exePath, model, maxTokens, wantThinking, family);
    }

    private void EditUserMessage(MessageRowInfo info)
    {
        if (_isGenerating) return;

        // Find last user index in history
        var userIdx = -1;
        for (var i = _history.Count - 1; i >= 0; i--)
        {
            if (_history[i].IsUser) { userIdx = i; break; }
        }
        if (userIdx < 0) return;

        // If invoked from a specific user row that is not the last user, ignore
        if (info.Role.Equals("user", StringComparison.OrdinalIgnoreCase))
        {
            var lastUser = FindLastUserRow();
            if (lastUser != null && info != lastUser) return;
        }

        var text = _history[userIdx].Content ?? "";
        _history.RemoveRange(userIdx, _history.Count - userIdx);
        RebuildChatFromHistory();
        SaveCurrentSession();

        PromptBox.Text = text;
        PromptBox.CaretIndex = text.Length;
        PromptPlaceholder.Visibility = string.IsNullOrEmpty(text)
            ? Visibility.Visible
            : Visibility.Collapsed;
        PromptBox.Focus();
        UpdateChatStatus();
    }

    private void RemoveChatRow(Border row)
    {
        ChatPanel.Children.Remove(row);
        if (ChatPanel.Children.Count == 0
 || (ChatPanel.Children.Count == 1 && ReferenceEquals(ChatPanel.Children[0], EmptyHint)))
        {
            // keep empty hint handling via Rebuild if needed
        }
        if (_history.Count == 0)
            RebuildChatFromHistory();
    }

    private void FinishStreamingBubble()
    {
        if (_streamingBubble?.Tag is MessageRowInfo info && _streamingTextBlock != null)
        {
            var plain = _streamingTextBlock.Text ?? "";
            info.PlainText = plain;
            if (info.Role.Equals("assistant", StringComparison.OrdinalIgnoreCase))
                SimpleMarkdown.RenderTo(info.Body, plain, TextPrimaryBrush);
        }
        _streamingTextBlock = null;
        _streamingBubble = null;
        UpdateContextMeter();
    }

    private void AppendStreaming(string chunk)
    {
        if (_streamingTextBlock == null) return;
        if (_streamHitImEnd) return;

        var visible = FilterAssistantStream(chunk, out var hitEnd);
        if (hitEnd)
        {
            _streamHitImEnd = true;
            RequestStopFromStream();
        }
        if (visible.Length == 0) return;
        _streamingTextBlock.Text += visible;
        if (_streamingBubble?.Tag is MessageRowInfo info)
            info.PlainText = _streamingTextBlock.Text ?? "";
        ScrollToBottom();
    }

    /// <summary>
    /// Soft-stop when ChatML end marker appears in the byte stream (BPE leak path).
    /// Engine should also stop on im_end_id after tokenizer special-match fix.
    /// </summary>
    private void RequestStopFromStream()
    {
        if (_stopRequested) return;
        _stopRequested = true;
        try { _cts?.Cancel(); } catch { /* ignore */ }
        var proc = _runningProcess;
        if (proc != null)
            KillProcessTree(proc);
    }

    private void ResetThinkFilter(bool wantThinking)
    {
        _inThinkBlock = false;
        _thinkCarry.Clear();
        _specialCarry.Clear();
        _streamHitImEnd = false;
        // Always strip think tags from the visible bubble when thinking is off.
        _stripThink = !wantThinking;
    }

    /// <summary>
    /// Strip &lt;think&gt; blocks (when enabled) and ChatML / special markers from
    /// streamed assistant text. Signals hitEnd when &lt;|im_end|&gt; / EOS appears
    /// so generation can stop even if the engine missed the CONTROL token.
    /// </summary>
    private string FilterAssistantStream(string chunk, out bool hitEnd)
    {
        hitEnd = false;
        var s = _specialCarry.Length > 0 ? _specialCarry + chunk : chunk;
        _specialCarry.Clear();

        // Hold back a short suffix that might be a partial special marker.
        const int maxMark = 16; // longest: <|endoftext|> = 13
        var hold = 0;
        if (s.Length > 0)
        {
            var maxKeep = Math.Min(maxMark - 1, s.Length);
            for (var k = maxKeep; k > 0; k--)
            {
                var suffix = s.Substring(s.Length - k);
                if (CouldBeSpecialPrefix(suffix))
                {
                    hold = k;
                    break;
                }
            }
        }

        var body = hold > 0 ? s.Substring(0, s.Length - hold) : s;
        if (hold > 0) _specialCarry.Append(s, s.Length - hold, hold);

        if (body.Length == 0) return "";

        // Detect stop markers before stripping so we still stop cleanly.
        if (body.Contains("<|im_end|>", StringComparison.Ordinal)
            || body.Contains("<|endoftext|>", StringComparison.Ordinal))
            hitEnd = true;

        body = StripChatMlMarkers(body);
        if (_stripThink)
            body = FilterThinkTags(body);
        return body;
    }

    private static bool CouldBeSpecialPrefix(string suffix)
    {
        // Partial prefixes of <|im_end|>, <|im_start|>, <|endoftext|>, <think>, </think>
        ReadOnlySpan<string> marks =
        [
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "<think>",
            "</think>",
        ];
        foreach (var m in marks)
        {
            if (m.StartsWith(suffix, StringComparison.Ordinal)) return true;
            // also allow suffix that is a trailing fragment mid-marker
            for (var i = 1; i < m.Length; i++)
            {
                var frag = m.Substring(i);
                if (frag.Length == suffix.Length
                    && frag.Equals(suffix, StringComparison.Ordinal))
                    return true;
            }
        }
        return suffix.Length > 0 && suffix[0] == '<';
    }

    private static string StripChatMlMarkers(string text)
    {
        if (string.IsNullOrEmpty(text)) return text;
        if (text.IndexOf('<') < 0) return text;

        var sb = new StringBuilder(text.Length);
        var i = 0;
        while (i < text.Length)
        {
            if (text[i] != '<')
            {
                sb.Append(text[i]);
                i++;
                continue;
            }

            // <|im_end|> or <|endoftext|>
            if (MatchAt(text, i, "<|im_end|>"))
            {
                i += "<|im_end|>".Length;
                continue;
            }
            if (MatchAt(text, i, "<|endoftext|>"))
            {
                i += "<|endoftext|>".Length;
                continue;
            }

            // <|im_start|>role\n  -- drop header through the first newline
            if (MatchAt(text, i, "<|im_start|>"))
            {
                i += "<|im_start|>".Length;
                while (i < text.Length && text[i] != '\n') i++;
                if (i < text.Length && text[i] == '\n') i++;
                continue;
            }

            sb.Append(text[i]);
            i++;
        }
        return sb.ToString();
    }

    private static bool MatchAt(string text, int i, string lit)
    {
        if (i + lit.Length > text.Length) return false;
        return string.CompareOrdinal(text, i, lit, 0, lit.Length) == 0;
    }

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
        StatusText.Text = _seenOutput
            ? $"Generating... {sec:0.0}s"
            : $"Loading... {sec:0.0}s";
    }

    private void StartGenTimer()
    {
        _genStartedUtc = DateTime.UtcNow;
        _lastReplySec = 0;
        _lastToks = "";
        _seenOutput = false;
        StatusText.Text = "Loading... 0.0s";
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
            StatusText.Text = $"Stopped | {_lastReplySec:0.0}s";
        }
        else if (!string.IsNullOrEmpty(_lastToks))
        {
            StatusText.Text = $"Last reply: {_lastReplySec:0.0}s | {_lastToks}";
            SpeedText.Text = _lastToks;
        }
        else if (!string.IsNullOrEmpty(_lastSpeed))
        {
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

    // - Input handling -

    private void PromptBox_TextChanged(object sender, TextChangedEventArgs e)
    {
        PromptPlaceholder.Visibility = string.IsNullOrEmpty(PromptBox.Text)
            ? Visibility.Visible
            : Visibility.Collapsed;
        if (!_isGenerating)
            UpdateContextMeter();
    }

    private void PromptBox_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && (Keyboard.Modifiers & ModifierKeys.Shift) == 0)
        {
            e.Handled = true;
            if (!_isGenerating) Send_Click(sender, e);
        }
    }

    // - Generate / Stop -

    private bool TryPrepareGenerate(
        out string exePath,
        out ModelEntry model,
        out int maxTokens,
        out bool wantThinking,
        out ChatPromptBuilder.ModelFamily family)
    {
        exePath = null!;
        model = null!;
        maxTokens = 512;
        wantThinking = false;
        family = ChatPromptBuilder.ModelFamily.Qwen35Moe;

        if (ModelCombo.SelectedItem is not ModelEntry m)
        {
            MessageBox.Show("Select a model first.", "Katali Lab",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return false;
        }
        model = m;

        if (!int.TryParse(MaxTokensBox.Text.Trim(), out maxTokens) || maxTokens < 1)
            maxTokens = 512;

        // Stock katali-lab.exe is the ship path for dense Qwen3 as well as MoE.
        // Optional katali-lab-qwen3.exe remains a fallback only if stock is absent.
        var denseQwen3 = IsDenseQwen3Model(model.Path + " " + model.Label);
        var resolved = ResolveKataliLabExe() ?? (denseQwen3 ? ResolveDenseQwen3Exe() : null);
        if (resolved == null)
        {
            var name = denseQwen3 ? "katali-lab.exe (or katali-lab-qwen3.exe)" : "katali-lab.exe";
            MessageBox.Show(
                $"Could not find {name}.\nPlace it next to this GUI (or in the parent of a gui\bin folder).",
                "Katali Lab", MessageBoxButton.OK, MessageBoxImage.Error);
            return false;
        }
        exePath = resolved;

        PersistSelectedModel();
        PersistMaxTokensFromBox();
        SyncThinkingDefaultForModel(model);

        family = ChatPromptBuilder.DetectFamily(model.Path + " " + model.Label);
        wantThinking = _settings.EnableThinking;
        return true;
    }

    private async void Send_Click(object sender, RoutedEventArgs e)
    {
        if (_isGenerating) return;

        var prompt = PromptBox.Text.Trim();
        if (string.IsNullOrEmpty(prompt)) return;

        if (!TryPrepareGenerate(out var exePath, out var model, out var maxTokens, out var wantThinking, out var family))
            return;

        PromptBox.Text = "";
        _history.Add(new ChatMessage("user", prompt));
        AddBubble("user", prompt);
        MaybeAutoTitle();
        UpdateContextMeter();
        ResetThinkFilter(wantThinking);
        _streamingBubble = AddBubble("assistant", "", streaming: true);

        await ExecuteGenerateAsync(exePath, model, maxTokens, wantThinking, family);
    }

    private async Task ExecuteGenerateAsync(
        string exePath,
        ModelEntry model,
        int maxTokens,
        bool wantThinking,
        ChatPromptBuilder.ModelFamily family)
    {
        var chatPrompt = ChatPromptBuilder.Build(_history, _settings.SystemPrompt, wantThinking, includeAssistantHeader: true, family: family);

        _stopRequested = false;
        var runId = ++_runId;
        SetGenerating(true);
        StartGenTimer();
        _lastSpeed = "";
        _lastToks = "";

        _cts = new CancellationTokenSource();
        var token = _cts.Token;

        try
        {
            await RunGenerateAsync(exePath, model, chatPrompt, maxTokens, wantThinking, family, token);
            if (!_stopRequested && runId == _runId)
            {
                CommitAssistantFromStream();
                StopGenTimer(stopped: false);
                SaveCurrentSession();
            }
        }
        catch (OperationCanceledException)
        {
            if (runId == _runId)
            {
                if (_streamingTextBlock != null &&
                    !_streamingTextBlock.Text.EndsWith("[stopped]", StringComparison.Ordinal))
                    AppendStreaming("\n[stopped]");
                CommitAssistantFromStream(stripStoppedMarker: true);
                StopGenTimer(stopped: true);
                SaveCurrentSession();
            }
        }
        catch (Exception ex)
        {
            if (runId == _runId)
            {
                AppendStreaming($"\n[error] {ex.Message}");
                CommitAssistantFromStream();
                StopGenTimer(stopped: true);
                SetStatus($"Error | {_lastReplySec:0.0}s");
                SaveCurrentSession();
            }
        }
        finally
        {
            if (runId == _runId)
            {
                FinishStreamingBubble();
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
                UpdateChatStatus();
                PromptBox.Focus();
            }
        }
    }

    private void FlushStreamCarry()
    {
        if (_specialCarry.Length == 0 && _thinkCarry.Length == 0) return;
        var leftover = _specialCarry.ToString() + _thinkCarry.ToString();
        _specialCarry.Clear();
        _thinkCarry.Clear();
        if (leftover.Length == 0 || _streamingTextBlock == null) return;
        leftover = StripChatMlMarkers(leftover);
        if (_stripThink) leftover = FilterThinkTags(leftover);
        if (leftover.Length == 0) return;
        _streamingTextBlock.Text += leftover;
        if (_streamingBubble?.Tag is MessageRowInfo info)
            info.PlainText = _streamingTextBlock.Text ?? "";
    }
    private void CommitAssistantFromStream(bool stripStoppedMarker = false)
    {
        if (_streamingTextBlock == null) return;
        // Flush any held partial special/think suffix before committing.
        FlushStreamCarry();
        var text = _streamingTextBlock.Text ?? "";
        text = StripChatMlMarkers(text);
        if (_stripThink)
            text = System.Text.RegularExpressions.Regex.Replace(
                text, @"<think>[\s\S]*?(</think>|$)", "",
                System.Text.RegularExpressions.RegexOptions.CultureInvariant);
        if (stripStoppedMarker)
        {
            const string mark = "\n[stopped]";
            if (text.EndsWith(mark, StringComparison.Ordinal))
                text = text.Substring(0, text.Length - mark.Length);
            else if (text.EndsWith("[stopped]", StringComparison.Ordinal))
                text = text.Substring(0, text.Length - "[stopped]".Length);
        }
        text = text.TrimEnd();
        _history.Add(new ChatMessage("assistant", text));
    }

    private async Task RunGenerateAsync(
        string exePath,
        ModelEntry model,
        string chatPrompt,
        int maxTokens,
        bool wantThinking,
        ChatPromptBuilder.ModelFamily family,
        CancellationToken token)
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

        var denseQwen3 = Path.GetFileName(exePath).Equals("katali-lab-qwen3.exe", StringComparison.OrdinalIgnoreCase);
        if (denseQwen3)
        {
            psi.Environment["KATALI_CUDA"] = _settings.CudaMoe ? "1" : "0";
            psi.ArgumentList.Add("run");
            psi.ArgumentList.Add(model.Path);
            var densePrompt = ExtractDenseUserPrompt(chatPrompt);
            psi.ArgumentList.Add("--prompt");
            psi.ArgumentList.Add(densePrompt);
            psi.ArgumentList.Add("--max-tokens");
            psi.ArgumentList.Add(maxTokens.ToString());
            psi.ArgumentList.Add(wantThinking ? "--think" : "--no-think");
        }
        else
        {        psi.Environment["KATALI_RAW_PROMPT"] = "1";
        psi.Environment["KATALI_CHATML"] = "1";
        if (wantThinking)
        {
            psi.Environment["KATALI_THINK"] = "1";
            psi.Environment.Remove("KATALI_PREFILL_THINK");
        }
        else
        {
            psi.Environment["KATALI_THINK"] = "0";
            if (family == ChatPromptBuilder.ModelFamily.Qwen3Dense)
            {
                // Dense Qwen3: no empty-think prefill and no /no_think (see ChatPromptBuilder).
                psi.Environment.Remove("KATALI_PREFILL_THINK");
            }
            else
            {
                // MoE families: ask engine for empty-think prefill when thinking is off.
                // GUI also embeds EmptyThinkPrefill in the RAW prompt + /no_think.
                psi.Environment["KATALI_PREFILL_THINK"] = "1";
            }
        }

        psi.Environment["KATALI_CUDA_MOE"] = _settings.CudaMoe ? "1" : "0";
        psi.Environment["KATALI_ECACHE_MMAP"] = _settings.EcacheMmap ? "1" : "0";
        if (_settings.Threads is int tw && tw > 0)
            psi.Environment["KATALI_EC_WORKERS"] = tw.ToString();
        else
            psi.Environment.Remove("KATALI_EC_WORKERS");

        psi.ArgumentList.Add("generate");
        psi.ArgumentList.Add(model.Path);

        CleanupPromptTempFile();
        var tmpDir = Path.Combine(Path.GetTempPath(), "KataliLab");
        Directory.CreateDirectory(tmpDir);
        _promptTempFile = Path.Combine(tmpDir, $"prompt-{Guid.NewGuid():N}.txt");
        await File.WriteAllTextAsync(_promptTempFile, chatPrompt, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false), token);
        psi.ArgumentList.Add(".");
        psi.ArgumentList.Add("--file");
        psi.ArgumentList.Add(_promptTempFile);

        psi.ArgumentList.Add("--max");
        psi.ArgumentList.Add(maxTokens.ToString());
        if (_settings.PinPercent is int pin && pin > 0)
        {
            psi.ArgumentList.Add("--pin");
            psi.ArgumentList.Add(pin.ToString());
        }
        if (_settings.CacheGb is int cacheGb && cacheGb > 0)
        {
            psi.ArgumentList.Add("--cache-gb");
            psi.ArgumentList.Add(cacheGb.ToString());
        }

        }
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

    private void AdvanceStatusOnOutput()
    {
        if (_stopRequested || !_isGenerating) return;
        _seenOutput = true;
    }

    private void HandleStderrLine(string line)
    {
        if (_stopRequested || !_isGenerating) return;

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

        foreach (var (needle, status) in PhaseHints)
        {
            if (line.Contains(needle, StringComparison.OrdinalIgnoreCase))
            {
                if (status.StartsWith("loading", StringComparison.Ordinal)
                    && (StatusText.Text.StartsWith("Generating", StringComparison.Ordinal)
 || StatusText.Text is "done" or "prefill..."))
                    return;
                if (status == "prefill..."
                    && StatusText.Text.StartsWith("Generating", StringComparison.Ordinal))
                    return;
                if (StatusText.Text.StartsWith("Generating...", StringComparison.Ordinal)
                    && status.StartsWith("generating", StringComparison.OrdinalIgnoreCase))
                    return;
                SetStatus(status);
                return;
            }
        }

        if (StatusText.Text is "loading model..." or "loading...")
            SetStatus("Generating...");
    }

    private static string? TryExtractToks(string speedLine)
    {
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
        _runId++;

        StopBtn.IsEnabled = false;
        SetStatus("stopping...");

        try { _cts?.Cancel(); } catch { /* ignore */ }
        KillRunningProcess();
        try { _cts?.Dispose(); } catch { /* ignore */ }
        _cts = null;

        if (_streamingTextBlock != null &&
            !_streamingTextBlock.Text.EndsWith("[stopped]", StringComparison.Ordinal))
        {
            AppendStreaming("\n[stopped]");
        }
        CommitAssistantFromStream(stripStoppedMarker: true);
        FinishStreamingBubble();
        StopGenTimer(stopped: true);
        SetGenerating(false);
        SaveCurrentSession();
        UpdateChatStatus();
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
        SettingsBtn.IsEnabled = !on;
        NewChatBtn.IsEnabled = !on;
        // Keep the dark history panel rendered while generation is active; disabling a WPF ListBox applies the default white theme.\n        SessionList.IsHitTestVisible = !on;
        StopBtn.IsEnabled = on;
    }

    private void SetStatus(string text) => StatusText.Text = text;

    private void CleanupPromptTempFile()
    {
        var f = _promptTempFile;
        _promptTempFile = null;
        if (string.IsNullOrEmpty(f)) return;
        try { if (File.Exists(f)) File.Delete(f); } catch { /* ignore */ }
    }

    private static string ExtractDenseUserPrompt(string chatPrompt)
    {
        const string end = "<|im_end|>";
        const string marker = "<|im_start|>user";
        var at = chatPrompt.LastIndexOf(marker, StringComparison.Ordinal);
        if (at < 0) return chatPrompt;
        // History text may use CRLF even though the live builder uses LF.
        var begin = chatPrompt.IndexOf('\n', at + marker.Length);
        if (begin < 0)
            begin = chatPrompt.IndexOf('\r', at + marker.Length);
        if (begin < 0)
            return chatPrompt.Substring(at + marker.Length).Trim();
        begin++;
        while (begin < chatPrompt.Length && (chatPrompt[begin] == '\r' || chatPrompt[begin] == '\n'))
            begin++;
        var stop = chatPrompt.IndexOf(end, begin, StringComparison.Ordinal);
        var text = (stop >= 0 ? chatPrompt.Substring(begin, stop - begin) : chatPrompt.Substring(begin)).Trim();
        if (text.EndsWith(" /no_think", StringComparison.OrdinalIgnoreCase))
            text = text.Substring(0, text.Length - " /no_think".Length).TrimEnd();
        return text;
    }
    private static string? ResolveDenseQwen3Exe()
    {
        var start = AppContext.BaseDirectory;
        var dir = new DirectoryInfo(start);
        for (var i = 0; i < 8 && dir != null; i++, dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, "katali-lab-qwen3.exe");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    private static bool IsDenseQwen3Model(string value)
    {
        var s = value.ToLowerInvariant();
        return s.Contains("minicpm5") ||
            (s.Contains("qwen3-1.7b") || s.Contains("qwen3-4b") || s.Contains("qwen3-8b"))
            && !s.Contains("coder") && !s.Contains("qwen3.5") && !s.Contains("qwen3.6");
    }
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
