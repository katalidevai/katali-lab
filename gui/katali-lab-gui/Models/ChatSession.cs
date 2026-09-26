using System;
using System.Collections.Generic;

namespace KataliLabGui.Models;

/// <summary>Persisted chat session under %AppData%\KataliLab\chats\.</summary>
public sealed class ChatSession
{
    public string Id { get; set; } = "";
    public string Title { get; set; } = "New chat";
    public DateTime CreatedUtc { get; set; }
    public DateTime UpdatedUtc { get; set; }
    public List<ChatSessionMessage> Messages { get; set; } = new();
}

public sealed class ChatSessionMessage
{
    public string Role { get; set; } = "";
    public string Content { get; set; } = "";
}
