using System.Text.Json;
using Loralei.Core.Dto;

namespace Loralei.Core;

/// <summary>
/// Reads run artefacts straight off disk. These are all files loralei already writes, so the
/// GUI can show a run's state without spawning anything - including while a run is live,
/// and for runs started from a terminal.
/// </summary>
public sealed class RunStore(LoraleiPaths paths)
{
    private static readonly JsonSerializerOptions Json = new() { PropertyNameCaseInsensitive = true };

    public string Root(string outputRoot = "runs") => paths.OutputRoot(outputRoot);

    /// <summary>Same listing as <c>loralei runs ls</c>, read directly rather than scraped.</summary>
    public IReadOnlyList<RunEntry> List(string outputRoot = "runs")
    {
        var root = Root(outputRoot);
        if (!Directory.Exists(root)) return [];

        var entries = new List<RunEntry>();
        foreach (var dir in Directory.EnumerateDirectories(root))
        {
            var manifest = ReadManifest(Path.GetFileName(dir), outputRoot);
            entries.Add(new RunEntry(
                Name: Path.GetFileName(dir),
                Model: manifest?.Model ?? "-",
                Step: manifest?.Step ?? 0,
                State: manifest?.State ?? "unknown",
                Updated: manifest?.Updated,
                Path: dir));
        }
        return entries.OrderByDescending(e => e.Updated ?? DateTimeOffset.MinValue).ToList();
    }

    public RunManifest? ReadManifest(string name, string outputRoot = "runs")
    {
        var path = Path.Combine(paths.RunDir(name, outputRoot), "manifest.json");
        try
        {
            // write_manifest rewrites this file wholesale, so a read can land mid-write.
            return File.Exists(path)
                ? JsonSerializer.Deserialize<RunManifest>(File.ReadAllText(path), Json)
                : null;
        }
        catch (Exception e) when (e is IOException or JsonException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    public string? ReadResolvedConfig(string name, string outputRoot = "runs")
    {
        var path = Path.Combine(paths.RunDir(name, outputRoot), "config.resolved.yaml");
        try { return File.Exists(path) ? File.ReadAllText(path) : null; }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return null; }
    }

    public string EventLogPath(string name, string outputRoot = "runs") =>
        Path.Combine(paths.RunDir(name, outputRoot), "logs", "train.jsonl");

    /// <summary>Adapters in the run directory: the final one first, then any exports.</summary>
    public IReadOnlyList<FileInfo> Adapters(string name, string outputRoot = "runs")
    {
        var dir = new DirectoryInfo(paths.RunDir(name, outputRoot));
        if (!dir.Exists) return [];
        try
        {
            return dir.GetFiles("*.safetensors")
                .OrderBy(f => f.Name.Contains("_comfy") ? 1 : 0)
                .ThenBy(f => f.Name, StringComparer.Ordinal)
                .ToList();
        }
        catch (IOException) { return []; }
    }

    /// <summary>
    /// Checkpoints that <c>--resume</c> can actually use. <c>checkpoint.find_latest</c>
    /// requires the <c>.state.pt</c> sidecar, so a bare .safetensors is not resumable.
    /// </summary>
    public IReadOnlyList<FileInfo> ResumableCheckpoints(string name, string outputRoot = "runs")
    {
        var dir = new DirectoryInfo(Path.Combine(paths.RunDir(name, outputRoot), "checkpoints"));
        if (!dir.Exists) return [];
        try
        {
            return dir.GetFiles("*.safetensors")
                .Where(f => File.Exists(Path.ChangeExtension(f.FullName, null) + ".state.pt"))
                .OrderBy(f => f.Name, StringComparer.Ordinal)
                .ToList();
        }
        catch (IOException) { return []; }
    }

    public bool CanResume(string name, string outputRoot = "runs") =>
        ResumableCheckpoints(name, outputRoot).Count > 0;

    /// <summary>Validation renders, newest first; each step gets its own subfolder.</summary>
    public IReadOnlyList<FileInfo> SampleImages(string name, string outputRoot = "runs", int limit = 60)
    {
        var dir = new DirectoryInfo(Path.Combine(paths.RunDir(name, outputRoot), "samples"));
        if (!dir.Exists) return [];
        try
        {
            return dir.GetFiles("*.png", SearchOption.AllDirectories)
                .OrderByDescending(f => f.LastWriteTimeUtc)
                .Take(limit)
                .ToList();
        }
        catch (IOException) { return []; }
    }
}
