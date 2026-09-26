using System;

namespace KataliLabGui.Models;

/// <summary>One turn in the multi-turn chat history sent to the engine.</summary>
public sealed class ChatMessage
{
    public string Role { get; }
    public string Content { get; set; }

    public ChatMessage(string role, string content)
    {
        Role = role ?? throw new ArgumentNullException(nameof(role));
        Content = content ?? "";
    }

    public bool IsUser => Role.Equals("user", StringComparison.OrdinalIgnoreCase);
    public bool IsAssistant => Role.Equals("assistant", StringComparison.OrdinalIgnoreCase);
    public bool IsSystem => Role.Equals("system", StringComparison.OrdinalIgnoreCase);
}
