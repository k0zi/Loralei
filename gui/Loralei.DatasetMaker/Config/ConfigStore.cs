using System.Text.Json;

namespace Loralei.DatasetMaker.Config;

/// <summary>
/// Reads and writes the config file. Nothing here throws: a corrupt or unwritable file costs
/// the remembered values, not the page.
/// </summary>
public sealed class ConfigStore(string path)
{
    /// <summary>Shipped next to the executable; what a first run starts from.</summary>
    public const string DefaultsFileName = "dataset-maker.defaults.json";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
        WriteIndented = true,
    };

    public string Path => path;

    /// <summary>
    /// Loads the config from the store's own file, or from the defaults file next to the
    /// executable. Missing values fall back to the built-in defaults.
    /// </summary>
    public AppConfig Load()
    {
        var config = Read(path)
                     ?? Read(System.IO.Path.Combine(AppContext.BaseDirectory, DefaultsFileName))
                     ?? new AppConfig();

        if (string.IsNullOrWhiteSpace(config.Caption.SystemPrompt))
            config = config with { Caption = config.Caption with { SystemPrompt = CaptionOptions.DefaultSystemPrompt } };

        return config;
    }

    public void Save(AppConfig config)
    {
        try
        {
            if (System.IO.Path.GetDirectoryName(path) is { Length: > 0 } folder)
                Directory.CreateDirectory(folder);

            File.WriteAllText(path, JsonSerializer.Serialize(config, JsonOptions));
            // The file can hold an API key, so it is the owner's alone.
            if (!OperatingSystem.IsWindows())
                File.SetUnixFileMode(path, UnixFileMode.UserRead | UnixFileMode.UserWrite);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
        }
    }

    /// <summary>
    /// The key to send: the configured one, or else <c>DATASETMAKER_API_KEY</c> / <c>OPENAI_API_KEY</c>.
    /// Resolved per run and never stored, so a key from the environment is not copied to disk.
    /// </summary>
    public static string ResolveApiKey(string configured)
    {
        if (!string.IsNullOrWhiteSpace(configured))
            return configured;

        return Environment.GetEnvironmentVariable("DATASETMAKER_API_KEY") is { Length: > 0 } own
            ? own
            : Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? "";
    }

    private static AppConfig? Read(string file)
    {
        try
        {
            return File.Exists(file)
                ? JsonSerializer.Deserialize<AppConfig>(File.ReadAllText(file), JsonOptions)
                : null;
        }
        catch (Exception e) when (e is IOException or JsonException or UnauthorizedAccessException)
        {
            return null;
        }
    }
}
