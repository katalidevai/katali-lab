namespace KataliLabGui.Models;

/// <summary>
/// A discovered GGUF model: either a single .gguf file or a directory of shards.
/// </summary>
public sealed class ModelEntry
{
    public string Label { get; }
    public string Path { get; }
    public bool IsDirectory { get; }

    public ModelEntry(string label, string path, bool isDirectory = false)
    {
        Label = label;
        Path = path;
        IsDirectory = isDirectory;
    }

    public override string ToString() => Label;
}
