using System.Text.Json.Nodes;
using Loralei.Core.Caching;
using Loralei.Core.Config;
using Loralei.Core.Data;
using Loralei.Core.Models;

namespace Loralei.Core.Worker;

/// <summary>
/// The shared front half of <c>cache</c> and <c>train</c>: the model resolved, the dataset
/// scanned, the buckets planned, the cache located. Everything here runs without touching the
/// GPU or starting Python, which is why <c>--dry-run</c> and <c>data check</c> are instant.
/// </summary>
public sealed record Prepared(ModelSpec Spec, ResolvedModel Resolved, ScanResult Scan, BucketPlan Plan,
    CacheLayout Layout)
{
    public static Prepared From(RunConfig cfg, Action<int, int>? scanProgress = null)
    {
        var spec = ModelRegistry.Get(cfg.Model);
        var resolved = ModelResolver.Resolve(spec, new ModelSources(cfg.ModelPath, cfg.ModelsDir));
        var scan = DatasetScanner.Scan(cfg.Dataset, cfg.CaptionExtension, triggerWord: cfg.TriggerWord,
            progress: scanProgress);
        var plan = BucketPlan.Build(scan.Items, cfg.Resolution, cfg.BucketStep, cfg.MinBucketResolution,
            cfg.MaxBucketResolution, cfg.Bucketing, spec.BucketDivisor);
        var layout = CacheLayout.For(
            cfg.Dataset, cfg.Model, cfg.Resolution, cfg.BucketStep, cfg.Bucketing,
            resolved.Components["vae"].Complete ? resolved.PathOf("vae") : "missing-vae",
            resolved.Components["text_encoder"].Complete ? resolved.PathOf("text_encoder") : "missing-te",
            spec.MaxSequenceLength);
        return new Prepared(spec, resolved, scan, plan, layout);
    }

    /// <summary>Can training start without touching the VAE or the text encoder?</summary>
    public (bool Complete, string Why) CacheStatus()
    {
        var missingLatents = Scan.Items.Count(i => !File.Exists(Layout.LatentPath(i.Key, Plan.BucketFor(i))));
        var missingText = Scan.Items.Count(i => !File.Exists(Layout.TextPath(i.Caption)));
        return missingLatents > 0 || missingText > 0
            ? (false, $"{missingLatents} latent(s) and {missingText} text embedding(s) missing")
            : (true, "cache complete");
    }
}

/// <summary>
/// Builds the JSON documents the Python worker runs. Each one is complete: the worker never
/// reads <c>loralei.yaml</c>, never searches for a model and never scans a dataset - it is
/// handed absolute paths and decided settings, and does the part that needs torch.
/// </summary>
public static class WorkerJobs
{
    private static JsonObject Base(string job) => new()
    {
        ["job"] = job,
        ["apu_targets"] = Strings(RocmEnvironment.ApuGfxTargets.Order(StringComparer.Ordinal)),
    };

    private static JsonArray Strings(IEnumerable<string> values) =>
        new(values.Select(v => (JsonNode?)JsonValue.Create(v)).ToArray());

    private static JsonObject Spec(ModelSpec spec) => new()
    {
        ["key"] = spec.Key,
        ["pipeline_class"] = spec.PipelineClass,
        ["transformer_class"] = spec.TransformerClass,
        ["blocks_attr"] = spec.BlocksAttr,
        ["max_sequence_length"] = spec.MaxSequenceLength,
        ["family"] = spec.Family,
    };

    /// <summary>Only the components that resolved; the worker fails clearly if it needs one that is absent.</summary>
    private static JsonObject Components(ResolvedModel resolved)
    {
        var json = new JsonObject();
        foreach (var (name, component) in resolved.Components)
            if (component is { Complete: true, Path: { } path }) json[name] = Path.GetFullPath(path);
        return json;
    }

    private static void AddDataset(JsonObject job, RunConfig cfg, Prepared prepared)
    {
        job["config"] = cfg.ToJson();
        job["spec"] = Spec(prepared.Spec);
        job["components"] = Components(prepared.Resolved);
        job["cache"] = new JsonObject
        {
            ["latents"] = Path.GetFullPath(prepared.Layout.Latents),
            ["text"] = Path.GetFullPath(prepared.Layout.Text),
        };
        job["fingerprint"] = prepared.Scan.Fingerprint();
        job["items"] = new JsonArray(prepared.Scan.Items.Select(item =>
        {
            var bucket = prepared.Plan.BucketFor(item);
            return (JsonNode?)new JsonObject
            {
                ["image"] = Path.GetFullPath(item.Image),
                ["caption"] = item.Caption,
                ["key"] = item.Key,
                ["bucket"] = new JsonArray(bucket.Width, bucket.Height),
            };
        }).ToArray());
    }

    public static JsonObject Cache(RunConfig cfg, Prepared prepared, bool force)
    {
        var job = Base("cache");
        AddDataset(job, cfg, prepared);
        job["force"] = force;
        return job;
    }

    /// <param name="cacheFirst">Fill the latent and text caches before training, in the same process.</param>
    public static JsonObject Train(RunConfig cfg, Prepared prepared, bool resume, bool cacheFirst)
    {
        var job = Base("train");
        AddDataset(job, cfg, prepared);
        job["resume"] = resume;
        job["cache_first"] = cacheFirst;
        return job;
    }

    public static JsonObject Sample(ResolvedModel resolved, IReadOnlyList<string> prompts, string outDir,
        string? lora, int steps, double cfgScale, int width, int height, int seed, int? gpu)
    {
        var job = Base("sample");
        job["spec"] = Spec(resolved.Spec);
        job["components"] = Components(resolved);
        job["prompts"] = Strings(prompts);
        job["out"] = Path.GetFullPath(outDir);
        job["lora"] = lora is null ? null : Path.GetFullPath(lora);
        job["steps"] = steps;
        job["cfg"] = cfgScale;
        job["width"] = width;
        job["height"] = height;
        job["seed"] = seed;
        job["gpu"] = gpu;
        return job;
    }

    /// <summary>Loads an adapter back into the real transformer to prove every key lands on a module.</summary>
    public static JsonObject Verify(ResolvedModel resolved, string lora)
    {
        var job = Base("verify");
        job["spec"] = Spec(resolved.Spec);
        job["components"] = Components(resolved);
        job["lora"] = Path.GetFullPath(lora);
        return job;
    }

    /// <summary>Which devices torch sees, and which one a job would land on.</summary>
    public static JsonObject Probe(int? gpu)
    {
        var job = Base("probe");
        job["gpu"] = gpu;
        return job;
    }

    public static JsonObject Doctor(bool quick, IReadOnlyList<string> agents)
    {
        var job = Base("doctor");
        job["quick"] = quick;
        job["agents"] = Strings(agents);
        return job;
    }
}

/// <summary>
/// Line-delimited JSON mirror of the training progress, for out-of-process readers. The
/// progress bar is the terminal's view of a run; this is the one a GUI can tail. Nothing
/// inside loralei reads it back, so failing to write it must never take a run down - every
/// error here degrades to silence.
/// </summary>
public sealed class EventLog : IDisposable
{
    private StreamWriter? _writer;

    public EventLog(string path, bool append = false)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
            var stream = new FileStream(path, append ? FileMode.Append : FileMode.Create, FileAccess.Write,
                FileShare.ReadWrite | FileShare.Delete);
            _writer = new StreamWriter(stream) { AutoFlush = true };
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
    }

    public void Emit(JsonObject fields)
    {
        if (_writer is null) return;
        fields["t"] = Math.Round(DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0, 3);
        try { _writer.WriteLine(fields.ToJsonString()); }
        catch (Exception e) when (e is IOException or ObjectDisposedException) { _writer = null; }
    }

    public void Dispose()
    {
        try { _writer?.Dispose(); }
        catch (IOException) { }
        _writer = null;
    }
}
