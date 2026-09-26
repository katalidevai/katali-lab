using System.Diagnostics;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;

namespace KataliLabGui;

/// <summary>
/// Lightweight Markdown → WPF Inlines (bold, italic, code, fences, lists, links).
/// No external packages. Safe for public chat UI; falls back to plain text on odd input.
/// </summary>
public static class SimpleMarkdown
{
    private static readonly Brush CodeBg = new SolidColorBrush(Color.FromRgb(0x21, 0x26, 0x2D));
    private static readonly Brush CodeFg = new SolidColorBrush(Color.FromRgb(0xE6, 0xED, 0xF3));
    private static readonly Brush LinkFg = new SolidColorBrush(Color.FromRgb(0x2D, 0xD4, 0xBF));
    private static readonly FontFamily Mono = new("Consolas, Cascadia Mono, Courier New, monospace");
    private static readonly FontFamily Ui = new("Segoe UI, sans-serif");

    private static readonly Regex FenceRegex = new(
        @"^```[^\n]*\n([\s\S]*?)(?:\n```|$)",
        RegexOptions.Multiline | RegexOptions.Compiled);

    private static readonly Regex InlineCode = new(@"`([^`\n]+)`", RegexOptions.Compiled);
    private static readonly Regex Bold = new(@"\*\*([^*]+)\*\*", RegexOptions.Compiled);
    private static readonly Regex Italic = new(@"(?<!\*)\*([^*\n]+)\*(?!\*)", RegexOptions.Compiled);
    private static readonly Regex Link = new(@"\[([^\]]+)\]\((https?://[^)\s]+)\)", RegexOptions.Compiled);
    private static readonly Regex Unordered = new(@"^\s*[-*]\s+(.+)$", RegexOptions.Compiled);
    private static readonly Regex Ordered = new(@"^\s*\d+\.\s+(.+)$", RegexOptions.Compiled);

    public static void RenderTo(TextBlock target, string? markdown, Brush foreground)
    {
        target.Inlines.Clear();
        target.Text = null;
        target.FontFamily = Ui;
        var text = markdown ?? "";
        if (text.Length == 0) return;

        try
        {
            var pos = 0;
            foreach (Match fence in FenceRegex.Matches(text))
            {
                if (fence.Index > pos)
                    AppendParagraphBlock(target, text.Substring(pos, fence.Index - pos), foreground);
                AppendCodeFence(target, fence.Groups[1].Value.TrimEnd('\r', '\n'), foreground);
                pos = fence.Index + fence.Length;
            }
            if (pos < text.Length)
                AppendParagraphBlock(target, text.Substring(pos), foreground);
        }
        catch
        {
            target.Inlines.Clear();
            target.Text = text;
            target.FontFamily = Mono;
            target.Foreground = foreground;
        }
    }

    private static void AppendCodeFence(TextBlock target, string code, Brush foreground)
    {
        if (target.Inlines.Count > 0)
            target.Inlines.Add(new LineBreak());

        var run = new Run(code)
        {
            FontFamily = Mono,
            FontSize = 12.5,
            Foreground = CodeFg,
            Background = CodeBg,
        };
        target.Inlines.Add(run);
        target.Inlines.Add(new LineBreak());
    }

    private static void AppendParagraphBlock(TextBlock target, string block, Brush foreground)
    {
        var lines = block.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');
        var first = true;
        foreach (var raw in lines)
        {
            if (!first) target.Inlines.Add(new LineBreak());
            first = false;

            var line = raw;
            var mList = Unordered.Match(line);
            if (mList.Success)
            {
                target.Inlines.Add(new Run("• ") { Foreground = foreground, FontFamily = Ui });
                AppendInlines(target, mList.Groups[1].Value, foreground);
                continue;
            }
            var mOrd = Ordered.Match(line);
            if (mOrd.Success)
            {
                var prefix = Regex.Match(line, @"^\s*(\d+)\.\s+").Groups[1].Value;
                target.Inlines.Add(new Run(prefix + ". ") { Foreground = foreground, FontFamily = Ui });
                AppendInlines(target, mOrd.Groups[1].Value, foreground);
                continue;
            }

            AppendInlines(target, line, foreground);
        }
    }

    private static void AppendInlines(TextBlock target, string line, Brush foreground)
    {
        if (string.IsNullOrEmpty(line)) return;

        // Tokenize by finding earliest of link / code / bold / italic
        var i = 0;
        while (i < line.Length)
        {
            var nextLink = Link.Match(line, i);
            var nextCode = InlineCode.Match(line, i);
            var nextBold = Bold.Match(line, i);
            var nextItalic = Italic.Match(line, i);

            Match? best = null;
            void Consider(Match m)
            {
                if (!m.Success) return;
                if (best == null || m.Index < best.Index) best = m;
            }
            Consider(nextLink);
            Consider(nextCode);
            Consider(nextBold);
            Consider(nextItalic);

            if (best == null)
            {
                target.Inlines.Add(new Run(line.Substring(i)) { Foreground = foreground, FontFamily = Ui });
                break;
            }

            if (best.Index > i)
            {
                target.Inlines.Add(new Run(line.Substring(i, best.Index - i))
                {
                    Foreground = foreground,
                    FontFamily = Ui,
                });
            }

            if (best == nextLink && nextLink.Success && best.Index == nextLink.Index)
            {
                var label = nextLink.Groups[1].Value;
                var url = nextLink.Groups[2].Value;
                var link = new Hyperlink(new Run(label))
                {
                    NavigateUri = Uri.TryCreate(url, UriKind.Absolute, out var uri) ? uri : null,
                    Foreground = LinkFg,
                };
                link.RequestNavigate += (_, e) =>
                {
                    try
                    {
                        Process.Start(new ProcessStartInfo(e.Uri.AbsoluteUri) { UseShellExecute = true });
                        e.Handled = true;
                    }
                    catch { /* ignore */ }
                };
                target.Inlines.Add(link);
                i = nextLink.Index + nextLink.Length;
            }
            else if (best == nextCode && nextCode.Success && best.Index == nextCode.Index)
            {
                target.Inlines.Add(new Run(nextCode.Groups[1].Value)
                {
                    FontFamily = Mono,
                    FontSize = 12.5,
                    Foreground = CodeFg,
                    Background = CodeBg,
                });
                i = nextCode.Index + nextCode.Length;
            }
            else if (best == nextBold && nextBold.Success && best.Index == nextBold.Index)
            {
                target.Inlines.Add(new Run(nextBold.Groups[1].Value)
                {
                    FontWeight = FontWeights.SemiBold,
                    Foreground = foreground,
                    FontFamily = Ui,
                });
                i = nextBold.Index + nextBold.Length;
            }
            else if (best == nextItalic && nextItalic.Success && best.Index == nextItalic.Index)
            {
                target.Inlines.Add(new Run(nextItalic.Groups[1].Value)
                {
                    FontStyle = FontStyles.Italic,
                    Foreground = foreground,
                    FontFamily = Ui,
                });
                i = nextItalic.Index + nextItalic.Length;
            }
            else
            {
                // Shouldn't happen; advance one char to avoid infinite loop
                target.Inlines.Add(new Run(line.Substring(i, 1)) { Foreground = foreground, FontFamily = Ui });
                i++;
            }
        }
    }
}
