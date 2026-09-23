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
            // Cap recents before write
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
}
