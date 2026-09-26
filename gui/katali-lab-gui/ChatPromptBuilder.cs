using System;
using System.Collections.Generic;
using System.Text;
using KataliLabGui.Models;

namespace KataliLabGui;

/// <summary>
/// Builds Qwen ChatML for katali-lab. GUI sets KATALI_RAW_PROMPT=1 so the engine
/// encodes this string as-is: we include the final assistant header and, when
/// thinking is off (MoE families), the empty think prefill (same as host.c
/// format_prompt). Dense Qwen3 skips empty-think and /no_think.
/// </summary>
public static class ChatPromptBuilder
{
    public const string ImStart = "<|im_start|>";
    public const string ImEnd = "<|im_end|>";
    /// <summary>Matches host.c empty-think prefill after assistant header.</summary>
    public const string EmptyThinkPrefill = "<think>\n\n</think>\n\n";

    public enum ModelFamily
    {
        /// <summary>Qwen3.5 / 3.6 MoE and Coder-Next - engine default: no thinking.</summary>
        Qwen35Moe,
        /// <summary>Qwen3-Coder-30B (arch qwen3moe) - engine default: thinking ON.</summary>
        Qwen3Moe,
        /// <summary>Dense Qwen3 (1.7B/4B/8B/...) - no empty-think /no_think prefill.</summary>
        Qwen3Dense,
    }

    /// <summary>
    /// Detect family from model path or label (case-insensitive).
    /// Coder-Next => Qwen35Moe-style; Coder-30 / qwen3-coder (not Next) => Qwen3Moe;
    /// dense Qwen3 size tags => Qwen3Dense.
    /// </summary>
    public static ModelFamily DetectFamily(string? modelPathOrLabel)
    {
        var s = modelPathOrLabel ?? "";
        var lower = s.ToLowerInvariant();

        if (lower.Contains("coder-next") || lower.Contains("coder_next"))
            return ModelFamily.Qwen35Moe;

        if (lower.Contains("coder-30") || lower.Contains("coder_30") ||
            lower.Contains("coder-30b") || lower.Contains("coder_30b"))
            return ModelFamily.Qwen3Moe;

        if ((lower.Contains("qwen3-coder") || lower.Contains("qwen3_coder")) &&
            !lower.Contains("next"))
            return ModelFamily.Qwen3Moe;

        // Dense Qwen3 (not MoE / not 3.5 / 3.6 hybrid).
        if (IsDenseQwen3(lower))
            return ModelFamily.Qwen3Dense;

        return ModelFamily.Qwen35Moe;
    }

    /// <summary>Path/label heuristic matching MainWindow.IsDenseQwen3Model.</summary>
    public static bool IsDenseQwen3(string lowerPathOrLabel)
    {
        var s = lowerPathOrLabel ?? "";
        if (s.Contains("minicpm5")) return true;
        if (s.Contains("qwen3.5") || s.Contains("qwen3.6") || s.Contains("qwen35") || s.Contains("qwen36"))
            return false;
        if (s.Contains("moe") || s.Contains("a3b") || s.Contains("a10b") || s.Contains("coder"))
            return false;
        return s.Contains("qwen3-1.7") || s.Contains("qwen3_1.7") ||
               s.Contains("qwen3-1_7") || s.Contains("qwen3-4b") ||
               s.Contains("qwen3_4b") || s.Contains("qwen3-8b") ||
               s.Contains("qwen3_8b") || s.Contains("qwen3-14b") ||
               s.Contains("qwen3-32b") ||
               (s.Contains("qwen3") && (s.Contains("1.7b") || s.Contains("4b") || s.Contains("8b")));
    }

    /// <summary>
    /// Build full ChatML ending with assistant header (RAW_PROMPT path).
    /// MoE families with thinking off: append " /no_think" + EmptyThinkPrefill.
    /// Dense Qwen3 with thinking off: plain assistant header only.
    /// </summary>
    public static string Build(
        IReadOnlyList<ChatMessage> history,
        string? systemPrompt,
        bool wantThinking,
        bool includeAssistantHeader = true,
        ModelFamily family = ModelFamily.Qwen35Moe)
    {
        var sb = new StringBuilder(Math.Max(256, history.Count * 64));
        var denseNoThinkPrefill = family == ModelFamily.Qwen3Dense && !wantThinking;

        if (!string.IsNullOrWhiteSpace(systemPrompt))
        {
            sb.Append(ImStart).Append("system\n");
            sb.Append(systemPrompt.TrimEnd());
            sb.Append(ImEnd).Append('\n');
        }

        var lastUserIdx = -1;
        for (var i = 0; i < history.Count; i++)
        {
            var r = (history[i]?.Role ?? "").Trim().ToLowerInvariant();
            if (r == "user") lastUserIdx = i;
        }

        for (var i = 0; i < history.Count; i++)
        {
            var msg = history[i];
            if (msg == null) continue;
            var role = (msg.Role ?? "").Trim().ToLowerInvariant();
            if (role is not ("user" or "assistant" or "system")) continue;
            sb.Append(ImStart).Append(role).Append('\n');
            sb.Append(msg.Content ?? "");
            if (!wantThinking && !denseNoThinkPrefill && role == "user" && i == lastUserIdx)
            {
                var c = msg.Content ?? "";
                if (!c.Contains("/no_think", StringComparison.OrdinalIgnoreCase)
                    && !c.Contains("/think", StringComparison.OrdinalIgnoreCase))
                    sb.Append(" /no_think");
            }
            sb.Append(ImEnd).Append('\n');
        }

        if (includeAssistantHeader)
        {
            sb.Append(ImStart).Append("assistant\n");
            if (!wantThinking && !denseNoThinkPrefill)
                sb.Append(EmptyThinkPrefill);
        }

        return sb.ToString();
    }
}