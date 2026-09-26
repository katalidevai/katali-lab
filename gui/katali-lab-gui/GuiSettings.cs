using System.IO;
using System.Text.Json;

namespace KataliLabGui;

/// <summary>
/// Persists GUI preferences under %AppData%\KataliLab\gui-settings.json
/// (Environment.SpecialFolder.ApplicationData).
/// </summary>
public sealed class GuiSettings
{
    public const int MaxRecentModels = 8;

    /// <summary>Primary laptop model path preferred on first launch.</summary>
    public const string Primary35BPath = @"C:\models\Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf";

    /// <summary>Soft prompt-token budget shown in the context meter.</summary>
    public const int ContextSoftBudgetTokens = 8192;

    /// <summary>Warn in the status meter when estimated prompt tokens exceed this.</summary>
    public const int ContextWarnTokens = 6000;

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };

    public string? LastModelPath { get; set; }

    /// <summary>Initial directory for the next Browse… / Folder… dialog.</summary>
    public string? LastBrowseDirectory { get; set; }

    /// <summary>Recently browsed model paths (file or multi-shard folder), newest first.</summary>
    public List<string> RecentModelPaths { get; set; } = new();

    public int MaxTokens { get; set; } = 512;

    /// <summary>Expert-cache size in GiB (--cache-gb). Null = omit flag (engine default).</summary>
    public int? CacheGb { get; set; } = 8;

    /// <summary>Pin percent (--pin). Null = omit flag (engine default ~25).</summary>
    public int? PinPercent { get; set; } = 25;

    /// <summary>Optional worker-thread override. Null = engine default.</summary>
    public int? Threads { get; set; }

    /// <summary>
    /// Thinking override. true =&gt; think on. false =&gt; empty-think prefill (no-think chat).
    /// Default false. Enable for Qwen3-Coder-30B if you want the engine's think-on default.
    /// </summary>
    public bool EnableThinking { get; set; }

    public string SystemPrompt { get; set; } = "";

    /// <summary>Use GPU for MoE experts when true (often slower for short 35B CPU runs).</summary>
    public bool CudaMoe { get; set; }

    /// <summary>Memory-map the expert cache when true (default).</summary>
    public bool EcacheMmap { get; set; } = true;

    /// <summary>Last open chat session id under chats\.</summary>
    public string? LastSessionId { get; set; }

    public static string SettingsDirectory =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "KataliLab");

    public static string SettingsPath =>
        Path.Combine(SettingsDirectory, "gui-settings.json");

    public static GuiSettings Load()
    {
        try
        {
            var path = SettingsPath;
            if (!File.Exists(path)) return new GuiSettings();
            var json = File.ReadAllText(path);
            var s = JsonSerializer.Deserialize<GuiSettings>(json, JsonOpts) ?? new GuiSettings();
            s.RecentModelPaths ??= new List<string>();
            s.SystemPrompt ??= "";
            if (s.MaxTokens < 1) s.MaxTokens = 512;
            return s;
        }
        catch
        {
            return new GuiSettings();
        }
    }

    public void Save()
    {
        try
        {
            Directory.CreateDirectory(SettingsDirectory);
            if (RecentModelPaths != null && RecentModelPaths.Count > MaxRecentModels)
                RecentModelPaths = RecentModelPaths.Take(MaxRecentModels).ToList();
            var json = JsonSerializer.Serialize(this, JsonOpts);
            File.WriteAllText(SettingsPath, json);
        }
        catch
        {
            /* ignore IO failures — settings are best-effort */
        }
    }

    /// <summary>Push path to front of recents (case-insensitive), capped at MaxRecentModels.</summary>
    public void RememberModelPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path)) return;
        RecentModelPaths ??= new List<string>();
        RecentModelPaths.RemoveAll(p =>
            string.Equals(p, path, StringComparison.OrdinalIgnoreCase));
        RecentModelPaths.Insert(0, path);
        while (RecentModelPaths.Count > MaxRecentModels)
            RecentModelPaths.RemoveAt(RecentModelPaths.Count - 1);
        LastModelPath = path;
    }

    /// <summary>True when path/label looks like the primary 35B MoE laptop model.</summary>
    public static bool Is35BModel(string? pathOrLabel)
    {
        var s = (pathOrLabel ?? "").ToLowerInvariant().Replace('_', '-');
        if (s.Contains("35b-a3b")) return true;
        if (s.Contains("qwen3.6-35b") || s.Contains("qwen3.5-35b")) return true;
        if (s.Contains("qwen3-6-35b") || s.Contains("qwen3-5-35b")) return true;
        return false;
    }

    /// <summary>
    /// Fill only null cache/pin fields with the measured 35B laptop profile.
    /// Never overwrites user-saved non-null values.
    /// </summary>
    public bool SoftFill35BUnsetFields()
    {
        var changed = false;
        if (CacheGb is null) { CacheGb = 8; changed = true; }
        if (PinPercent is null) { PinPercent = 25; changed = true; }
        if (MaxTokens < 1) { MaxTokens = 512; changed = true; }
        return changed;
    }
}
