using System.Text.Json;
using System.Text.Json.Serialization;

namespace Loralei.Core;

/// <summary>
/// GUI-only preferences. Deliberately never written to <c>~/.config/loralei/config.yaml</c>:
/// that layer belongs to the user and feeds the plain CLI too.
/// </summary>
public sealed class GuiSettings
{
    [JsonPropertyName("executable")] public string? Executable { get; set; }
    [JsonPropertyName("workspace")] public string? Workspace { get; set; }
    [JsonPropertyName("last_dataset")] public string? LastDataset { get; set; }

    /// <summary>
    /// <c>system</c>, <c>light</c> or <c>dark</c>. Kept as a string so this assembly stays
    /// free of any Avalonia reference; the GUI's ThemeService does the parsing.
    /// </summary>
    [JsonPropertyName("theme")] public string Theme { get; set; } = "system";

    [JsonPropertyName("window_width")] public double WindowWidth { get; set; } = 1280;
    [JsonPropertyName("window_height")] public double WindowHeight { get; set; } = 860;
    [JsonPropertyName("log_pane_height")] public double LogPaneHeight { get; set; } = 200;

    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };

    /// <summary>
    /// Where preferences live. <c>LORALEI_GUI_SETTINGS</c> overrides it, which keeps separate
    /// profiles apart and keeps the test suite out of the real one.
    /// </summary>
    public static string DefaultPath =>
        Environment.GetEnvironmentVariable("LORALEI_GUI_SETTINGS") is { Length: > 0 } custom
            ? custom
            : Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".config", "loralei-gui", "settings.json");

    public static GuiSettings Load(string? path = null)
    {
        path ??= DefaultPath;
        try
        {
            return File.Exists(path)
                ? JsonSerializer.Deserialize<GuiSettings>(File.ReadAllText(path)) ?? new GuiSettings()
                : new GuiSettings();
        }
        catch (Exception e) when (e is IOException or JsonException or UnauthorizedAccessException)
        {
            return new GuiSettings();
        }
    }

    public void Save(string? path = null)
    {
        path ??= DefaultPath;
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, JsonSerializer.Serialize(this, Json));
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // Preferences are a convenience; losing them must not break the app.
        }
    }
}
