using System.Text.Json;
using System.Text.Json.Serialization;
using Loralei.Core.Config;
using Loralei.Core.Data;
using Loralei.Core.Models;

namespace Loralei.Core.Dto;

public static class LoraleiInfo
{
    public const string Version = "0.2.0";

    /// <summary>
    /// How every <c>--json</c> payload is written and read. The CLI and the GUI share these
    /// types, so the two cannot drift apart the way a scraped table and its parser can.
    /// </summary>
    public static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never,
        WriteIndented = true,
    };
}

public sealed record MissingComponent(string Component, string Reason);

/// <param name="Status"><c>ready</c>, <c>incomplete</c>, or <c>needs disk</c> when the download would not fit.</param>
public sealed record ModelStatus(string Key, string Label, string Status, long MissingBytes,
    IReadOnlyList<string> FoundIn, IReadOnlyList<MissingComponent> Missing);

/// <summary><c>loralei models ls --json</c>.</summary>
public sealed record ModelsReport(IReadOnlyList<ModelStatus> Models, long FreeBytes, string Root,
    IReadOnlyList<string> ModelsDir, string? ModelPath, IReadOnlyList<string> Notes)
{
    public static ModelsReport Build(ModelSources sources)
    {
        var root = sources.Roots.Count > 0 ? sources.Roots[0] : Disk.HfHubDir();
        var free = Disk.FreeBytes(root);

        var models = new List<ModelStatus>();
        foreach (var spec in ModelRegistry.All)
        {
            var resolved = ModelResolver.Resolve(spec, sources);
            var need = resolved.MissingBytes;
            var status = resolved.Missing.Count == 0 ? "ready"
                : need * Disk.SafetyFactor > free ? "needs disk"
                : "incomplete";
            models.Add(new ModelStatus(
                spec.Key, spec.Label, status, need,
                resolved.Components.Values.Where(c => c.Complete)
                    .Select(c => PathText.Shorten(c.Repo)).Distinct().Order(StringComparer.Ordinal).ToList(),
                resolved.Missing.Select(m => new MissingComponent(m.Component, PathText.Shorten(m.Reason))).ToList()));
        }

        return new ModelsReport(models, free, root, sources.Roots, sources.ModelPath,
            ModelRegistry.All.SelectMany(spec => spec.Notes.Select(note => $"{spec.Key}: {note}")).ToList());
    }
}

public sealed record BucketCount(int Width, int Height, int Images, int Tokens);

/// <summary><c>loralei data check --json</c>.</summary>
public sealed record DataReport(string Dataset, int Pairs, IReadOnlyList<string> Problems,
    IReadOnlyList<BucketCount> Buckets, string Captions)
{
    /// <param name="divisor">The model's pixels per latent token side; decides the bucket grid and the token counts.</param>
    public static DataReport Build(string dataset, int resolution = 1024, string captionExtension = ".txt",
        Action<int, int>? progress = null, int divisor = BucketPlan.LatentPatchDivisor)
    {
        var scan = DatasetScanner.Scan(dataset, captionExtension, progress: progress);
        var plan = BucketPlan.Build(scan.Items, resolution, divisor: divisor);
        return new DataReport(dataset, scan.Items.Count, scan.Problems,
            plan.ByBucket.Select(b => new BucketCount(b.Bucket.Width, b.Bucket.Height, b.Items.Count, plan.Tokens(b.Bucket))).ToList(),
            scan.CaptionStats());
    }
}

/// <summary><c>loralei models gc --json</c>.</summary>
public sealed record GcReport(IReadOnlyList<CacheEntry> Entries, long TotalBytes, long FreeBytes)
{
    public static GcReport Build(double minGib = 1.0)
    {
        var entries = Disk.Reclaimable((long)(minGib * Disk.Gib));
        return new GcReport(entries, entries.Sum(e => e.Bytes), Disk.FreeBytes(Disk.HfHubDir()));
    }
}
