namespace Loralei.Core;

/// <summary>Common shape for a command the GUI reads: the parsed view plus the text behind it.</summary>
public abstract record ScrapedResult(LoraleiResult Process, IReadOnlyList<string> Lines)
{
    public bool Ok => Process.Ok;
    public string RawText => string.Join('\n', Lines);
}

public sealed record ModelRow(string Key, string Description, string Status, string ToDownload,
    string FoundIn, string Missing)
{
    public bool Ready => Status.StartsWith("ready", StringComparison.OrdinalIgnoreCase);
    public bool NeedsDisk => Status.Contains("disk", StringComparison.OrdinalIgnoreCase);
}

public sealed record ModelsListing(LoraleiResult Process, IReadOnlyList<string> Lines,
    IReadOnlyList<ModelRow> Models, string FreeSpace, IReadOnlyList<string> Roots)
    : ScrapedResult(Process, Lines);

public sealed record BucketRow(string Bucket, string Images, string LatentTokens);

public sealed record DataCheckResult(LoraleiResult Process, IReadOnlyList<string> Lines,
    int Pairs, IReadOnlyList<BucketRow> Buckets, string Captions, IReadOnlyList<string> Problems)
    : ScrapedResult(Process, Lines);

/// <summary>What <c>train --dry-run</c> reports before it stops.</summary>
public sealed record DryRunResult(LoraleiResult Process, IReadOnlyList<string> Lines,
    string Summary, string Estimate, string CacheStatus, string ResolvedConfigPath, bool OverBudget)
    : ScrapedResult(Process, Lines);
