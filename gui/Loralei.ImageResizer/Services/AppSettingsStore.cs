using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using Loralei.ImageResizer.Models;

namespace Loralei.ImageResizer.Services;

/// <summary>
/// Reads and writes the settings file. Nothing here is allowed to throw: losing a preference is a
/// nuisance, failing to start over one is not acceptable.
/// </summary>
public sealed class AppSettingsStore
{
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter() },
    };

    private readonly string _path;

    public AppSettingsStore()
        : this(DefaultPath())
    {
    }

    /// <summary>Overload for tests, which must never touch the real settings file.</summary>
    public AppSettingsStore(string path) => _path = path;

    public string Path => _path;

    public static string DefaultPath() => System.IO.Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "ImageResizer",
        "settings.json");

    public AppSettings Load()
    {
        try
        {
            if (!File.Exists(_path))
                return new AppSettings();

            var settings = JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(_path), Options);
            return (settings ?? new AppSettings()).Sanitized();
        }
        catch (Exception)
        {
            // Corrupt, unreadable or written by a future version - start fresh rather than fail.
            return new AppSettings();
        }
    }

    public void Save(AppSettings settings)
    {
        try
        {
            var folder = System.IO.Path.GetDirectoryName(_path);
            if (!string.IsNullOrEmpty(folder))
                Directory.CreateDirectory(folder);

            File.WriteAllText(_path, JsonSerializer.Serialize(settings, Options));
        }
        catch (Exception)
        {
            // Read-only home directory, full disk, ... - the app keeps working, just forgetfully.
        }
    }
}
