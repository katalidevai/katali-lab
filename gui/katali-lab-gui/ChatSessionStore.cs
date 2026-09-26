using System.IO;
using System.Text.Json;
using KataliLabGui.Models;

namespace KataliLabGui;

/// <summary>
/// Load/save chat sessions as JSON under %AppData%\KataliLab\chats\.
/// </summary>
public static class ChatSessionStore
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };

    public static string ChatsDirectory =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "KataliLab",
            "chats");

    public static string PathForId(string id) =>
        Path.Combine(ChatsDirectory, id + ".json");

    public static ChatSession CreateNew()
    {
        var now = DateTime.UtcNow;
        var session = new ChatSession
        {
            Id = Guid.NewGuid().ToString("N"),
            Title = "New chat",
            CreatedUtc = now,
            UpdatedUtc = now,
            Messages = new List<ChatSessionMessage>(),
        };
        Save(session);
        return session;
    }

    public static void Save(ChatSession session)
    {
        if (session == null || string.IsNullOrWhiteSpace(session.Id)) return;
        try
        {
            Directory.CreateDirectory(ChatsDirectory);
            session.UpdatedUtc = DateTime.UtcNow;
            session.Messages ??= new List<ChatSessionMessage>();
            var json = JsonSerializer.Serialize(session, JsonOpts);
            File.WriteAllText(PathForId(session.Id), json);
        }
        catch
        {
            /* best-effort */
        }
    }

    public static ChatSession? Load(string id)
    {
        if (string.IsNullOrWhiteSpace(id)) return null;
        try
        {
            var path = PathForId(id);
            if (!File.Exists(path)) return null;
            var json = File.ReadAllText(path);
            var s = JsonSerializer.Deserialize<ChatSession>(json, JsonOpts);
            if (s == null || string.IsNullOrWhiteSpace(s.Id)) return null;
            s.Messages ??= new List<ChatSessionMessage>();
            s.Title ??= "New chat";
            return s;
        }
        catch
        {
            return null;
        }
    }

    public static List<ChatSession> LoadAll()
    {
        var list = new List<ChatSession>();
        try
        {
            Directory.CreateDirectory(ChatsDirectory);
            foreach (var file in Directory.EnumerateFiles(ChatsDirectory, "*.json"))
            {
                try
                {
                    var json = File.ReadAllText(file);
                    var s = JsonSerializer.Deserialize<ChatSession>(json, JsonOpts);
                    if (s == null || string.IsNullOrWhiteSpace(s.Id)) continue;
                    s.Messages ??= new List<ChatSessionMessage>();
                    s.Title ??= "New chat";
                    list.Add(s);
                }
                catch
                {
                    /* skip corrupt file */
                }
            }
        }
        catch
        {
            /* ignore */
        }

        list.Sort((a, b) => b.UpdatedUtc.CompareTo(a.UpdatedUtc));
        return list;
    }

    public static bool Delete(string id)
    {
        if (string.IsNullOrWhiteSpace(id)) return false;
        try
        {
            var path = PathForId(id);
            if (!File.Exists(path)) return false;
            File.Delete(path);
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Truncate first user message to ~40 chars for the session title.</summary>
    public static string TitleFromFirstUser(string? content, int maxChars = 40)
    {
        if (string.IsNullOrWhiteSpace(content)) return "New chat";
        var t = content.Trim().Replace('\r', ' ').Replace('\n', ' ');
        while (t.Contains("  ", StringComparison.Ordinal))
            t = t.Replace("  ", " ", StringComparison.Ordinal);
        if (t.Length <= maxChars) return t;
        return t.Substring(0, maxChars - 1).TrimEnd() + "...";
    }
}
