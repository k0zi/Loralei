namespace Loralei.Core;

/// <summary>
/// Where the CLI lives and which directory it runs in.
/// <para>
/// The working directory is load-bearing, not cosmetic: the CLI reads <c>loralei.yaml</c> from
/// its working directory and offers no <c>--config</c> flag, so the only way to set the ~25
/// options that have no flag is to write <c>loralei.yaml</c> into the workspace and launch
/// from there.
/// </para>
/// </summary>
public sealed class LoraleiPaths(string executable, string workspace)
{
    public string Executable { get; } = executable;
    public string Workspace { get; } = workspace;

    public string ProjectConfig => Path.Combine(Workspace, "loralei.yaml");

    /// <summary>Resolves the run root, which is relative to the workspace unless absolute.</summary>
    public string OutputRoot(string outputRoot = "runs") =>
        Path.IsPathRooted(outputRoot) ? outputRoot : Path.Combine(Workspace, outputRoot);

    public string RunDir(string name, string outputRoot = "runs") =>
        Path.Combine(OutputRoot(outputRoot), name);

    /// <summary>
    /// Picks the CLI: an explicit setting, else the one built beside this application, else PATH.
    /// </summary>
    public static LoraleiPaths Discover(string? executable, string? workspace)
    {
        var root = workspace is { Length: > 0 } && Directory.Exists(workspace)
            ? Path.GetFullPath(workspace)
            : FindWorkspace();

        // A setting saved by the all-Python loralei points at the virtualenv's `loralei`
        // script. That is the worker's entry point now, not the CLI, so it is not honoured.
        var exe = executable is { Length: > 0 } && File.Exists(executable) && !IsScript(executable)
            ? Path.GetFullPath(executable)
            : FindExecutable();

        return new LoraleiPaths(exe, root);
    }

    /// <summary>Walks up from the app looking for a directory that holds a loralei.yaml.</summary>
    private static string FindWorkspace()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "loralei.yaml"))) return dir.FullName;
            dir = dir.Parent;
        }
        return Directory.GetCurrentDirectory();
    }

    private static string FindExecutable()
    {
        // The CLI is a project reference of the GUI, so a build puts it next to the app.
        var sibling = Path.Combine(AppContext.BaseDirectory, "loralei");
        if (File.Exists(sibling)) return sibling;

        foreach (var entry in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
        {
            if (entry.Length == 0) continue;
            var candidate = Path.Combine(entry, "loralei");
            // A Python virtualenv may still hold the worker's old `loralei` script; that is
            // not the CLI, and running it would only print a usage error.
            if (File.Exists(candidate) && !IsScript(candidate)) return candidate;
        }
        return "loralei";
    }

    private static bool IsScript(string path)
    {
        try
        {
            using var stream = File.OpenRead(path);
            return stream.ReadByte() == '#' && stream.ReadByte() == '!';
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return false; }
    }

    /// <summary>Why the current selection will not work, or null when it looks usable.</summary>
    public string? Problem()
    {
        if (!Directory.Exists(Workspace)) return $"workspace does not exist: {Workspace}";
        if (Executable != "loralei" && !File.Exists(Executable)) return $"loralei not found at {Executable}";
        return null;
    }
}
