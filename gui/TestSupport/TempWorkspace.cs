using Loralei.Core;

namespace Loralei.Tests;

/// <summary>
/// A throwaway workspace: its own directory, its own <c>loralei.yaml</c>, its own GUI settings.
/// <para>
/// The GUI writes the flagless settings into the workspace <c>loralei.yaml</c>, so a test that
/// used the real project directory would rewrite the developer's file - and, if it were
/// interrupted before its cleanup ran, leave it rewritten. Tests get their own copy instead,
/// carrying over only <c>models_dir</c> so the base weights are still found.
/// </para>
/// </summary>
internal sealed class TempWorkspace : IDisposable
{
    public string Directory { get; }
    public string SettingsPath { get; }
    public string ProjectConfig => Path.Combine(Directory, "loralei.yaml");

    private readonly string? _previousSettings;

    private TempWorkspace(string directory, string settingsPath, string? previousSettings)
    {
        Directory = directory;
        SettingsPath = settingsPath;
        _previousSettings = previousSettings;
    }

    public static TempWorkspace Create(LoraleiPaths real)
    {
        var dir = System.IO.Directory.CreateTempSubdirectory("loralei-ws").FullName;

        // Carry over only where the base weights live; everything else starts from defaults.
        var roots = File.Exists(real.ProjectConfig)
            ? FlatYaml.Parse(File.ReadAllText(real.ProjectConfig)).Sequence("models_dir")
            : [];

        var doc = FlatYaml.Parse("");
        if (roots.Count > 0) doc.Set("models_dir", roots);
        File.WriteAllText(Path.Combine(dir, "loralei.yaml"), doc.ToString());

        var settings = Path.Combine(dir, "settings.json");
        new GuiSettings { Executable = real.Executable, Workspace = dir }.Save(settings);

        var previous = Environment.GetEnvironmentVariable("LORALEI_GUI_SETTINGS");
        Environment.SetEnvironmentVariable("LORALEI_GUI_SETTINGS", settings);
        return new TempWorkspace(dir, settings, previous);
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("LORALEI_GUI_SETTINGS", _previousSettings);
        try { System.IO.Directory.Delete(Directory, recursive: true); }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
    }
}
