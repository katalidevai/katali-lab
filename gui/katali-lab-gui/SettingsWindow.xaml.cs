using System.Windows;

namespace KataliLabGui;

public partial class SettingsWindow : Window
{
    private readonly GuiSettings _settings;

    public SettingsWindow(GuiSettings settings)
    {
        InitializeComponent();
        _settings = settings;
        LoadFromSettings();
    }

    private void LoadFromSettings()
    {
        MaxTokensBox.Text = _settings.MaxTokens > 0 ? _settings.MaxTokens.ToString() : "512";
        CacheGbBox.Text = _settings.CacheGb?.ToString() ?? "";
        PinPercentBox.Text = _settings.PinPercent?.ToString() ?? "";
        ThreadsBox.Text = _settings.Threads?.ToString() ?? "";
        EnableThinkingCheck.IsChecked = _settings.EnableThinking;
        SystemPromptBox.Text = _settings.SystemPrompt ?? "";
        CudaMoeCheck.IsChecked = _settings.CudaMoe;
        EcacheMmapCheck.IsChecked = _settings.EcacheMmap;
    }

    private void Save_Click(object sender, RoutedEventArgs e)
    {
        if (!int.TryParse(MaxTokensBox.Text.Trim(), out var max) || max < 1)
            max = 512;
        _settings.MaxTokens = max;

        _settings.CacheGb = ParseNullablePositive(CacheGbBox.Text);
        _settings.PinPercent = ParseNullablePositive(PinPercentBox.Text);
        _settings.Threads = ParseNullablePositive(ThreadsBox.Text);

        _settings.EnableThinking = EnableThinkingCheck.IsChecked == true;
        _settings.SystemPrompt = SystemPromptBox.Text ?? "";
        _settings.CudaMoe = CudaMoeCheck.IsChecked == true;
        _settings.EcacheMmap = EcacheMmapCheck.IsChecked != false;

        _settings.Save();
        DialogResult = true;
        Close();
    }

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
        Close();
    }

    private static int? ParseNullablePositive(string? text)
    {
        if (string.IsNullOrWhiteSpace(text)) return null;
        if (int.TryParse(text.Trim(), out var v) && v > 0) return v;
        return null;
    }
}
