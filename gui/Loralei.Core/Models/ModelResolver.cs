using System.Text.Json;

namespace Loralei.Core.Models;

/// <summary>Where to look for model components, before the default Hugging Face cache.</summary>
public sealed record ModelSources(string? ModelPath = null, IReadOnlyList<string>? ModelsDir = null)
{
    public IReadOnlyList<string> Roots => ModelsDir ?? [];
}

/// <param name="Repo">Where it came from: a Hub repo id, or the local root it was found under.</param>
public sealed record ResolvedComponent(string Component, string Repo, string? Path, bool Complete, string Reason = "")
{
    public bool Missing => !Complete;
}

public sealed record ResolvedModel(ModelSpec Spec, IReadOnlyDictionary<string, ResolvedComponent> Components)
{
    public bool Complete => Components.Values.All(c => c.Complete);

    public IReadOnlyList<ResolvedComponent> Missing =>
        Spec.Components.Select(c => Components[c.Name]).Where(c => c.Missing).ToList();

    public string PathOf(string component)
    {
        var resolved = Components[component];
        if (!resolved.Complete || resolved.Path is null)
            throw new FileNotFoundException(
                $"{Spec.Key}/{component} is not available locally: {resolved.Reason}");
        return resolved.Path;
    }

    public long MissingBytes => Missing.Sum(m => Spec.Component(m.Component).ApproxBytes);
}

/// <summary>
/// Finds model components on disk. The point is that 40 GB of weights already present should
/// never be downloaded a second time - whether they sit in the Hugging Face cache under a
/// mirror repo name, in a shared models directory, or in a folder the user points at directly.
/// <para>
/// Each component is resolved independently, so a half-populated directory works. Lookup
/// order per component: <c>model_path</c>, then each <c>models_dir</c> root in order, then
/// the default Hugging Face cache. A root may be in either layout:
/// </para>
/// <code>
/// &lt;root&gt;/models--&lt;org&gt;--&lt;repo&gt;/snapshots/&lt;rev&gt;/&lt;component&gt;/    Hugging Face cache
/// &lt;root&gt;/&lt;model key&gt;/&lt;component&gt;/                              plain folders
/// </code>
/// </summary>
public static class ModelResolver
{
    private static readonly string[] WeightSuffixes = [".safetensors", ".bin", ".pt", ".pth", ".gguf"];

    /// <summary>A root may be an HF cache itself, or an HF_HOME with the cache inside it.</summary>
    public static string HubRoot(string root) =>
        Directory.Exists(Path.Combine(root, "hub")) ? Path.Combine(root, "hub") : root;

    private static List<string> SnapshotDirs(string repoId, string? root = null)
    {
        var hub = root is not null ? HubRoot(root) : Disk.HfHubDir();
        var snapshots = Path.Combine(hub, "models--" + repoId.Replace("/", "--"), "snapshots");
        if (!Directory.Exists(snapshots)) return [];
        return Directory.EnumerateDirectories(snapshots)
            .OrderByDescending(Directory.GetLastWriteTimeUtc)
            .ToList();
    }

    /// <summary>
    /// A subfolder counts as present only if it actually holds the weights. This is the exact
    /// trap a cached Qwen-Image-2512 repo falls into: its <c>transformer/</c> can contain
    /// config.json and nothing else.
    /// </summary>
    private static (bool Ok, string Reason) FolderIsComplete(string folder, ComponentSpec spec)
    {
        if (!Directory.Exists(folder)) return (false, "folder not in cache");

        // Dangling links are kept in the listing on purpose, so they can be reported as such.
        var files = new DirectoryInfo(folder).EnumerateFileSystemInfos().Where(e => e is FileInfo).ToList();
        if (files.Count == 0) return (false, "folder is empty");
        if (!spec.Weights) return (true, "");

        var weights = files.Where(f => WeightSuffixes.Contains(f.Extension)).ToList();
        if (weights.Count == 0) return (false, "only config files present, no weights");
        foreach (var weight in weights)
            if (!Config.PathText.IsFile(weight)) // dangling symlink into a pruned blob
                return (false, $"dangling blob for {weight.Name}");

        var index = Path.Combine(folder, "diffusion_pytorch_model.safetensors.index.json");
        if (!File.Exists(index)) index = Path.Combine(folder, "model.safetensors.index.json");
        if (!File.Exists(index)) return (true, "");

        HashSet<string> shards;
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(index));
            shards = doc.RootElement.GetProperty("weight_map").EnumerateObject()
                .Select(p => p.Value.GetString() ?? "").ToHashSet();
        }
        catch (Exception e) when (e is IOException or JsonException or KeyNotFoundException or InvalidOperationException)
        {
            return (true, "");
        }

        var missing = shards.Except(weights.Select(w => w.Name)).Order(StringComparer.Ordinal).ToList();
        return missing.Count > 0
            ? (false, $"{missing.Count} shard(s) missing, e.g. {missing[0]}")
            : (true, "");
    }

    /// <summary>Every folder that might hold this component, best first.</summary>
    /// <param name="modelFolder">The model's folder name in the plain layout: its key.</param>
    public static List<(string Label, string Folder)> CandidateFolders(ComponentSpec spec, string modelFolder,
        ModelSources? sources)
    {
        var candidates = new List<(string, string)>();

        if (sources?.ModelPath is { } explicitPath)
            candidates.Add(($"model_path:{explicitPath}", Path.Combine(explicitPath, spec.Name)));

        foreach (var root in sources?.Roots ?? [])
        {
            // Plain layout first: it is unambiguous and cheap to check.
            candidates.Add((root, Path.Combine(root, modelFolder, spec.Name)));
            foreach (var repo in spec.Repos)
            foreach (var snapshot in SnapshotDirs(repo, root))
                candidates.Add(($"{repo} @ {root}", Path.Combine(snapshot, spec.Name)));
        }

        foreach (var repo in spec.Repos)
        foreach (var snapshot in SnapshotDirs(repo))
            candidates.Add((repo, Path.Combine(snapshot, spec.Name)));
        return candidates;
    }

    public static ResolvedComponent ResolveComponent(ComponentSpec spec, string modelFolder = "",
        ModelSources? sources = null)
    {
        var lastReason = "not found in any configured location";
        foreach (var (label, folder) in CandidateFolders(spec, modelFolder, sources))
        {
            var (ok, reason) = FolderIsComplete(folder, spec);
            if (ok) return new ResolvedComponent(spec.Name, label, folder, true);
            if (reason.Length > 0 && reason != "folder not in cache") lastReason = $"{label}: {reason}";
        }
        return new ResolvedComponent(spec.Name, spec.Repos[0], null, false, lastReason);
    }

    public static ResolvedModel Resolve(ModelSpec model, ModelSources? sources = null) =>
        new(model, model.Components.ToDictionary(c => c.Name, c => ResolveComponent(c, model.Key, sources)));

    /// <summary>
    /// Where <c>models pull</c> should write, or null for the default Hugging Face cache.
    /// Downloads go to the first configured root, in the plain layout - a browsable
    /// <c>&lt;root&gt;/&lt;model key&gt;/&lt;component&gt;/</c> tree rather than the cache's blob-and-symlink form.
    /// </summary>
    public static string? DownloadTarget(ModelSpec model, ModelSources? sources) =>
        sources is { Roots.Count: > 0 } ? Path.Combine(sources.Roots[0], model.Key) : null;
}
