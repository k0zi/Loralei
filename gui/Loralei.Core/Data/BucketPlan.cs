namespace Loralei.Core.Data;

public readonly record struct Bucket(int Width, int Height) : IComparable<Bucket>
{
    public double Aspect => (double)Width / Height;

    public int CompareTo(Bucket other) =>
        Width != other.Width ? Width.CompareTo(other.Width) : Height.CompareTo(other.Height);

    public override string ToString() => $"{Width}x{Height}";
}

/// <summary>
/// Aspect-ratio bucketing. Resizing every image to a square wastes tokens on padding and
/// distorts subjects. Instead a set of (w, h) buckets with roughly constant latent token count
/// is built and each image is assigned to the closest aspect ratio, so every batch has a
/// uniform shape without cropping more than necessary.
/// </summary>
public sealed class BucketPlan
{
    /// <summary>
    /// VAE downscale x transformer patch size for the models with an 8x VAE and 2x2 patches.
    /// The default; a model with a deeper VAE says so in its <c>ModelSpec.BucketDivisor</c>.
    /// </summary>
    public const int LatentPatchDivisor = 16;

    public int Resolution { get; }

    /// <summary>Pixels per latent token side for the model this plan was made for.</summary>
    public int Divisor { get; }

    /// <summary>Latent tokens the transformer sees for an image in a bucket.</summary>
    public int Tokens(Bucket bucket) => bucket.Width / Divisor * (bucket.Height / Divisor);

    /// <summary>Populated buckets, fullest first.</summary>
    public IReadOnlyList<(Bucket Bucket, IReadOnlyList<DatasetItem> Items)> ByBucket { get; }

    private readonly Dictionary<string, Bucket> _assignment;

    private BucketPlan(int resolution, int divisor, IReadOnlyList<(Bucket, IReadOnlyList<DatasetItem>)> byBucket,
        Dictionary<string, Bucket> assignment)
    {
        Resolution = resolution;
        Divisor = divisor;
        ByBucket = byBucket;
        _assignment = assignment;
    }

    public IReadOnlyList<Bucket> Buckets => ByBucket.Select(b => b.Bucket).ToList();

    public Bucket BucketFor(DatasetItem item) => _assignment[item.Key];

    public string Summary() => string.Join(", ", ByBucket.Select(b => $"{b.Bucket}:{b.Items.Count}"));

    /// <summary>Every (w, h) on the <paramref name="step"/> grid whose area stays within <paramref name="tolerance"/> of a square.</summary>
    public static List<Bucket> Generate(int resolution, int step = 64, int minSide = 256, int maxSide = 2048,
        double tolerance = 1.10, int divisor = LatentPatchDivisor)
    {
        if (step <= 0 || step % divisor != 0)
            throw new ArgumentException($"bucket step must be a multiple of {divisor}, got {step}");

        long targetArea = (long)resolution * resolution;
        var buckets = new HashSet<Bucket>();
        for (var width = minSide; width <= maxSide; width += step)
        {
            var height = (int)((double)targetArea / width);
            height -= height % step;
            if (height < minSide || height > maxSide) continue;
            if ((long)width * height > targetArea * tolerance) continue;
            buckets.Add(new Bucket(width, height));
            buckets.Add(new Bucket(height, width));
        }
        buckets.Add(new Bucket(resolution, resolution));
        return buckets.OrderBy(b => b.Aspect).ThenBy(b => b.Width).ToList();
    }

    public static BucketPlan Build(IReadOnlyList<DatasetItem> items, int resolution, int step = 64,
        int minSide = 256, int maxSide = 2048, bool enabled = true, int divisor = LatentPatchDivisor)
    {
        var buckets = enabled
            ? Generate(resolution, step, minSide, maxSide, divisor: divisor)
            : [new Bucket(resolution, resolution)];

        var grouped = new Dictionary<Bucket, List<DatasetItem>>();
        var assignment = new Dictionary<string, Bucket>(StringComparer.Ordinal);
        foreach (var item in items)
        {
            var bucket = buckets.MinBy(b => Math.Abs(b.Aspect - item.Aspect));
            if (!grouped.TryGetValue(bucket, out var group)) grouped[bucket] = group = [];
            group.Add(item);
            assignment[item.Key] = bucket;
        }

        // Keep only populated buckets, fullest first for stable display.
        var ordered = grouped
            .OrderByDescending(g => g.Value.Count)
            .ThenBy(g => g.Key)
            .Select(g => (g.Key, (IReadOnlyList<DatasetItem>)g.Value))
            .ToList();
        return new BucketPlan(resolution, divisor, ordered, assignment);
    }
}
