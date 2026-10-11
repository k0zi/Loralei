using System.Globalization;

namespace Loralei.Core;

/// <summary>Builds argv for every loralei command the GUI drives.</summary>
public static class CommandBuilder
{
    public static List<string> Doctor(bool json = false, bool thorough = false, bool fix = false)
    {
        var a = new List<string> { "doctor" };
        // --fix prints a shell snippet *after* the JSON, so the two cannot be combined.
        if (json) { a.Add("--json"); return a; }
        if (thorough) a.Add("--thorough");
        if (fix) a.Add("--fix");
        return a;
    }

    /// <param name="json">The typed form the GUI reads; without it the same data is a table.</param>
    public static List<string> ModelsLs(IReadOnlyList<string>? modelsDir = null, string? modelPath = null,
        bool json = false)
    {
        var a = new List<string> { "models", "ls" };
        AddSources(a, modelsDir, modelPath);
        if (json) a.Add("--json");
        return a;
    }

    public static List<string> ModelsPull(string model, IReadOnlyList<string>? modelsDir = null,
        string? modelPath = null, bool dryRun = false) =>
        ModelsPull([model], modelsDir, modelPath, dryRun);

    /// <summary>Several models in one command; each is checked and fetched in turn.</summary>
    public static List<string> ModelsPull(IReadOnlyList<string> models, IReadOnlyList<string>? modelsDir = null,
        string? modelPath = null, bool dryRun = false)
    {
        var a = new List<string> { "models", "pull" };
        a.AddRange(models);
        AddSources(a, modelsDir, modelPath);
        if (dryRun) a.Add("--dry-run");
        return a;
    }

    /// <summary>
    /// Always the listing form. <c>--no-dry-run</c> asks for a confirmation per entry, which
    /// a closed stdin answers with "no"; deleting caches stays a deliberate terminal action.
    /// </summary>
    public static List<string> ModelsGc(double minGib = 1.0, bool json = false)
    {
        var a = new List<string> { "models", "gc", "--dry-run", "--min-gib", Fmt(minGib) };
        if (json) a.Add("--json");
        return a;
    }

    public static List<string> DataCheck(string dataset, int? resolution = null, string? captionExtension = null,
        bool json = false)
    {
        var a = new List<string> { "data", "check", dataset };
        if (resolution is { } r) { a.Add("--resolution"); a.Add(Fmt(r)); }
        if (!string.IsNullOrWhiteSpace(captionExtension)) { a.Add("--caption-extension"); a.Add(captionExtension); }
        if (json) a.Add("--json");
        return a;
    }

    public static List<string> Cache(string dataset, string? model = null, int? resolution = null,
        int? gpu = null, string? vaeDevice = null, IReadOnlyList<string>? modelsDir = null,
        string? modelPath = null, bool force = false)
    {
        var a = new List<string> { "cache", dataset };
        if (!string.IsNullOrWhiteSpace(model)) { a.Add("--model"); a.Add(model); }
        if (resolution is { } r) { a.Add("--resolution"); a.Add(Fmt(r)); }
        if (gpu is { } g) { a.Add("--gpu"); a.Add(Fmt(g)); }
        if (!string.IsNullOrWhiteSpace(vaeDevice)) { a.Add("--vae-device"); a.Add(vaeDevice); }
        AddSources(a, modelsDir, modelPath);
        if (force) a.Add("--force");
        return a;
    }

    public static List<string> Train(TrainRequest r, bool dryRun = false)
    {
        var a = new List<string> { "train", r.Dataset, "--name", r.Name };

        if (!string.IsNullOrWhiteSpace(r.Model)) { a.Add("--model"); a.Add(r.Model); }
        if (!string.IsNullOrWhiteSpace(r.Profile)) { a.Add("--profile"); a.Add(r.Profile); }
        if (r.Rank is { } rank) { a.Add("--rank"); a.Add(Fmt(rank)); }
        if (r.Steps is { } steps) { a.Add("--steps"); a.Add(Fmt(steps)); }
        if (r.Resolution is { } res) { a.Add("--resolution"); a.Add(Fmt(res)); }
        if (r.LearningRate is { } lr) { a.Add("--lr"); a.Add(Fmt(lr)); }
        if (r.BatchSize is { } batch) { a.Add("--batch-size"); a.Add(Fmt(batch)); }
        if (r.BlocksToSwap is { } swap) { a.Add("--blocks-to-swap"); a.Add(Fmt(swap)); }
        if (r.Fp8Base is { } fp8) a.Add(fp8 ? "--fp8-base" : "--no-fp8-base");
        if (!string.IsNullOrWhiteSpace(r.Optimizer)) { a.Add("--optimizer"); a.Add(r.Optimizer); }
        if (!string.IsNullOrWhiteSpace(r.TriggerWord)) { a.Add("--trigger"); a.Add(r.TriggerWord); }
        if (!string.IsNullOrWhiteSpace(r.VaeDevice)) { a.Add("--vae-device"); a.Add(r.VaeDevice); }
        AddSources(a, r.ModelsDir, r.ModelPath);
        if (r.Gpu is { } gpu) { a.Add("--gpu"); a.Add(Fmt(gpu)); }
        if (r.SaveEvery is { } save) { a.Add("--save-every"); a.Add(Fmt(save)); }
        if (r.Seed is { } seed) { a.Add("--seed"); a.Add(Fmt(seed)); }
        if (r.Resume) a.Add("--resume");
        if (dryRun) a.Add("--dry-run");
        return a;
    }

    public static List<string> Sample(IReadOnlyList<string> prompts, string? lora = null, string? model = null,
        string? outDir = null, int? steps = null, double? cfg = null, int? width = null, int? height = null,
        int? seed = null, int? gpu = null, IReadOnlyList<string>? modelsDir = null, string? modelPath = null)
    {
        var a = new List<string> { "sample" };
        foreach (var prompt in prompts) { a.Add("--prompt"); a.Add(prompt); }
        if (!string.IsNullOrWhiteSpace(lora)) { a.Add("--lora"); a.Add(lora); }
        if (!string.IsNullOrWhiteSpace(model)) { a.Add("--model"); a.Add(model); }
        if (!string.IsNullOrWhiteSpace(outDir)) { a.Add("--out"); a.Add(outDir); }
        if (steps is { } s) { a.Add("--steps"); a.Add(Fmt(s)); }
        if (cfg is { } c) { a.Add("--cfg"); a.Add(Fmt(c)); }
        if (width is { } w) { a.Add("--width"); a.Add(Fmt(w)); }
        if (height is { } h) { a.Add("--height"); a.Add(Fmt(h)); }
        if (seed is { } sd) { a.Add("--seed"); a.Add(Fmt(sd)); }
        if (gpu is { } g) { a.Add("--gpu"); a.Add(Fmt(g)); }
        AddSources(a, modelsDir, modelPath);
        return a;
    }

    public static List<string> Export(string run, string format = "both", string? outputRoot = null,
        bool verify = true)
    {
        var a = new List<string> { "export", run, "--format", format };
        if (!string.IsNullOrWhiteSpace(outputRoot)) { a.Add("--output-root"); a.Add(outputRoot); }
        a.Add(verify ? "--verify" : "--no-verify");
        return a;
    }

    public static List<string> RunsLs(string? outputRoot = null)
    {
        var a = new List<string> { "runs", "ls" };
        if (!string.IsNullOrWhiteSpace(outputRoot)) { a.Add("--output-root"); a.Add(outputRoot); }
        return a;
    }

    public static List<string> RunsShow(string name, string? outputRoot = null)
    {
        var a = new List<string> { "runs", "show", name };
        if (!string.IsNullOrWhiteSpace(outputRoot)) { a.Add("--output-root"); a.Add(outputRoot); }
        return a;
    }

    /// <summary>Always <c>--yes</c>: the GUI has already confirmed, and stdin is closed.</summary>
    public static List<string> RunsRm(string name, string? outputRoot = null)
    {
        var a = new List<string> { "runs", "rm", name };
        if (!string.IsNullOrWhiteSpace(outputRoot)) { a.Add("--output-root"); a.Add(outputRoot); }
        a.Add("--yes");
        return a;
    }

    /// <summary>
    /// <c>--models-dir</c> takes a single value even though the config key is a list, so
    /// several roots go through as one path-separator-joined string, which
    /// <c>PathText.AsRoots</c> splits again.
    /// </summary>
    private static void AddSources(List<string> args, IReadOnlyList<string>? modelsDir, string? modelPath)
    {
        if (modelsDir is { Count: > 0 })
        {
            var roots = modelsDir.Where(d => !string.IsNullOrWhiteSpace(d)).ToList();
            if (roots.Count > 0)
            {
                args.Add("--models-dir");
                args.Add(string.Join(Path.PathSeparator, roots));
            }
        }
        if (!string.IsNullOrWhiteSpace(modelPath)) { args.Add("--model-path"); args.Add(modelPath); }
    }

    private static string Fmt(int value) => value.ToString(CultureInfo.InvariantCulture);
    private static string Fmt(double value) => value.ToString("R", CultureInfo.InvariantCulture);

    /// <summary>A copy-pasteable rendering of a command, for the preview strip.</summary>
    public static string Preview(string executable, IReadOnlyList<string> args) =>
        string.Join(' ', args.Prepend(executable).Select(Shell));

    private static string Shell(string token) =>
        token.Length > 0 && token.All(c => char.IsLetterOrDigit(c) || "-_./=:+,@".Contains(c))
            ? token
            : $"'{token.Replace("'", "'\\''")}'";
}
