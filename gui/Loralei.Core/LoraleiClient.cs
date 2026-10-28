using System.Text.Json;
using Loralei.Core.Dto;

namespace Loralei.Core;

/// <summary>
/// One call per loralei command, returning a parsed result and forwarding raw output to the
/// log pane. Every page goes through here, so the rules about which commands may see
/// FORCE_COLOR and which must never see it live in exactly one place.
/// </summary>
public sealed class LoraleiClient
{
    private static readonly JsonSerializerOptions Json = new() { PropertyNameCaseInsensitive = true };

    private readonly LoraleiProcess _process;

    public LoraleiClient(LoraleiPaths paths)
    {
        Paths = paths;
        _process = new LoraleiProcess(paths);
    }

    public LoraleiPaths Paths { get; }

    public string Preview(IReadOnlyList<string> args) => CommandBuilder.Preview(Paths.Executable, args);

    private async Task<(LoraleiResult Result, List<string> Lines)> CaptureAsync(
        IReadOnlyList<string> args, Action<ConsoleLine>? onOutput, bool forceColor, CancellationToken ct)
    {
        var lines = new List<string>();
        var result = await _process.RunAsync(args, line =>
        {
            // Transient frames are progress redraws; keep only committed lines for parsing.
            if (!line.Transient) lines.Add(line.Text);
            onOutput?.Invoke(line);
        }, forceColor, ct).ConfigureAwait(false);
        return (result, lines);
    }

    /// <summary>Runs a <c>--json</c> command and reads its payload back into the type the CLI wrote it from.</summary>
    /// <remarks>
    /// Without FORCE_COLOR on purpose - the payload must be plain. The output is captured
    /// rather than streamed: dumping a JSON document into the log pane would bury the
    /// human-readable output of every other command, so callers log a summary instead.
    /// </remarks>
    private async Task<(T? Payload, LoraleiResult Result, List<string> Lines)> JsonAsync<T>(
        IReadOnlyList<string> args, JsonSerializerOptions options, CancellationToken ct) where T : class
    {
        var (result, lines) = await CaptureAsync(args, null, forceColor: false, ct).ConfigureAwait(false);
        var text = string.Join('\n', lines);
        try
        {
            var start = text.IndexOf('{');
            var end = text.LastIndexOf('}');
            if (start < 0 || end <= start) return (null, result, lines);
            return (JsonSerializer.Deserialize<T>(text[start..(end + 1)], options), result, lines);
        }
        catch (JsonException) { return (null, result, lines); }
    }

    public async Task<(DoctorReport? Report, LoraleiResult Result, string RawText)> DoctorAsync(
        bool thorough = false, CancellationToken ct = default)
    {
        var args = CommandBuilder.Doctor(json: true);
        if (thorough) args.Add("--thorough");
        var (report, result, lines) = await JsonAsync<DoctorReport>(args, Json, ct).ConfigureAwait(false);
        return (report, result, string.Join('\n', lines));
    }

    public async Task<string> DoctorFixSnippetAsync(CancellationToken ct = default)
    {
        var (_, lines) = await CaptureAsync(CommandBuilder.Doctor(fix: true), null, false, ct)
            .ConfigureAwait(false);
        return string.Join('\n', lines);
    }

    public Task<LoraleiResult> DoctorInstallAsync(Action<ConsoleLine>? onOutput = null,
        CancellationToken ct = default) =>
        _process.RunAsync(CommandBuilder.DoctorInstall(), onOutput, forceColor: false, ct);

    private static string Gib(long bytes) =>
        (bytes / (1024.0 * 1024 * 1024)).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + " GiB";

    public async Task<ModelsListing> ModelsLsAsync(IReadOnlyList<string>? modelsDir = null,
        string? modelPath = null, Action<ConsoleLine>? onOutput = null, CancellationToken ct = default)
    {
        var (report, result, lines) = await JsonAsync<ModelsReport>(
            CommandBuilder.ModelsLs(modelsDir, modelPath, json: true), LoraleiInfo.Json, ct).ConfigureAwait(false);
        // Nothing parsed: the raw text is all there is to show, and to diagnose with.
        if (report is null)
        {
            foreach (var line in lines) onOutput?.Invoke(new ConsoleLine(line, false));
            return new ModelsListing(result, lines, [], "", []);
        }

        var rows = report.Models.Select(m => new ModelRow(
            m.Key, m.Label, m.Status,
            m.MissingBytes > 0 ? Gib(m.MissingBytes) : "-",
            m.FoundIn.Count > 0 ? string.Join('\n', m.FoundIn) : "-",
            m.Missing.Count > 0 ? string.Join(", ", m.Missing.Select(x => $"{x.Component} ({x.Reason})")) : "-"))
            .ToList();
        var free = $"{Gib(report.FreeBytes)} free on {report.Root}";
        var roots = report.ModelsDir
            .Select((dir, i) => $"models_dir[{i}]: {dir}{(i == 0 ? "  (downloads land here)" : "")}")
            .ToList();

        foreach (var row in rows)
            onOutput?.Invoke(new ConsoleLine($"{row.Key}: {row.Status}" +
                                             (row.Ready ? "" : $", {row.ToDownload} to download"), false));
        onOutput?.Invoke(new ConsoleLine(free, false));
        return new ModelsListing(result, [], rows, free, roots);
    }

    public Task<LoraleiResult> ModelsPullAsync(string model, IReadOnlyList<string>? modelsDir = null,
        string? modelPath = null, bool dryRun = false, Action<ConsoleLine>? onOutput = null,
        CancellationToken ct = default) =>
        ModelsPullAsync([model], modelsDir, modelPath, dryRun, onOutput, ct);

    public Task<LoraleiResult> ModelsPullAsync(IReadOnlyList<string> models,
        IReadOnlyList<string>? modelsDir = null, string? modelPath = null, bool dryRun = false,
        Action<ConsoleLine>? onOutput = null, CancellationToken ct = default) =>
        // FORCE_COLOR is what asks the CLI for its live download bar through a pipe.
        _process.RunAsync(CommandBuilder.ModelsPull(models, modelsDir, modelPath, dryRun),
            onOutput, forceColor: !dryRun, ct);

    public async Task<ScrapedResult> ModelsGcAsync(double minGib = 1.0,
        Action<ConsoleLine>? onOutput = null, CancellationToken ct = default)
    {
        var (report, result, lines) = await JsonAsync<GcReport>(
            CommandBuilder.ModelsGc(minGib, json: true), LoraleiInfo.Json, ct).ConfigureAwait(false);

        var listing = report is null
            ? lines
            : report.Entries.Count == 0
                ? ["nothing above the threshold"]
                : report.Entries.Select(e => $"{Gib(e.Bytes),10}  {e.Name}  {e.Path}")
                    .Append($"total reclaimable: {Gib(report.TotalBytes)} | currently free: {Gib(report.FreeBytes)}")
                    .ToList();
        foreach (var line in listing) onOutput?.Invoke(new ConsoleLine(line, false));
        return new GcListing(result, listing);
    }

    private sealed record GcListing(LoraleiResult Process, IReadOnlyList<string> Lines)
        : ScrapedResult(Process, Lines);

    public async Task<DataCheckResult> DataCheckAsync(string dataset, int? resolution = null,
        string? captionExtension = null, Action<ConsoleLine>? onOutput = null, CancellationToken ct = default)
    {
        var (report, result, lines) = await JsonAsync<DataReport>(
            CommandBuilder.DataCheck(dataset, resolution, captionExtension, json: true), LoraleiInfo.Json, ct)
            .ConfigureAwait(false);
        if (report is null)
        {
            foreach (var line in lines) onOutput?.Invoke(new ConsoleLine(line, false));
            return new DataCheckResult(result, lines, 0, [], "", []);
        }

        var buckets = report.Buckets
            .Select(b => new BucketRow($"{b.Width} x {b.Height}", b.Images.ToString(), b.Tokens.ToString()))
            .ToList();
        var captions = $"captions: {report.Captions}";

        onOutput?.Invoke(new ConsoleLine(
            $"{report.Pairs} image/caption pairs in {dataset}, {report.Problems.Count} problem(s)", false));
        foreach (var bucket in buckets)
            onOutput?.Invoke(new ConsoleLine($"  {bucket.Bucket}: {bucket.Images} image(s)", false));
        onOutput?.Invoke(new ConsoleLine(captions, false));

        // An empty dataset exits non-zero, but it parsed: the page should show "0 pairs" and
        // the problems, not a command failure.
        var outcome = result.ExitCode == 1 && report.Pairs == 0 ? result with { ExitCode = 0 } : result;
        return new DataCheckResult(outcome, [], report.Pairs, buckets, captions, report.Problems);
    }

    public Task<LoraleiResult> CacheAsync(string dataset, string? model = null, int? resolution = null,
        int? gpu = null, string? vaeDevice = null, IReadOnlyList<string>? modelsDir = null,
        string? modelPath = null, bool force = false, Action<ConsoleLine>? onOutput = null,
        CancellationToken ct = default) =>
        _process.RunAsync(
            CommandBuilder.Cache(dataset, model, resolution, gpu, vaeDevice, modelsDir, modelPath, force),
            onOutput, forceColor: true, ct);

    public async Task<DryRunResult> TrainDryRunAsync(TrainRequest request,
        Action<ConsoleLine>? onOutput = null, CancellationToken ct = default)
    {
        var (result, lines) = await CaptureAsync(
            CommandBuilder.Train(request, dryRun: true), onOutput, false, ct).ConfigureAwait(false);

        string Find(string prefix) =>
            lines.FirstOrDefault(l => l.StartsWith(prefix, StringComparison.Ordinal)) ?? "";

        return new DryRunResult(
            result, lines,
            Summary: lines.Count > 1 ? lines[1] : "",
            Estimate: Find("estimate:"),
            CacheStatus: Find("cache:"),
            ResolvedConfigPath: Find("resolved config ->").Replace("resolved config ->", "").Trim(),
            // Estimator.Format appends this marker when the run will not fit the card.
            OverBudget: lines.Any(l => l.Contains(Estimator.OverBudgetMarker, StringComparison.Ordinal)));
    }

    public Task<LoraleiResult> TrainAsync(TrainRequest request, Action<ConsoleLine>? onOutput = null,
        CancellationToken ct = default) =>
        _process.RunAsync(CommandBuilder.Train(request), onOutput, forceColor: true, ct);

    public Task<LoraleiResult> SampleAsync(IReadOnlyList<string> prompts, string? lora, string? model,
        string? outDir, int? steps, double? cfg, int? width, int? height, int? seed, int? gpu,
        IReadOnlyList<string>? modelsDir = null, string? modelPath = null,
        Action<ConsoleLine>? onOutput = null, CancellationToken ct = default) =>
        _process.RunAsync(
            CommandBuilder.Sample(prompts, lora, model, outDir, steps, cfg, width, height, seed, gpu,
                modelsDir, modelPath),
            onOutput, forceColor: true, ct);

    public Task<LoraleiResult> ExportAsync(string run, string format = "both", string? outputRoot = null,
        bool verify = true, Action<ConsoleLine>? onOutput = null, CancellationToken ct = default) =>
        _process.RunAsync(CommandBuilder.Export(run, format, outputRoot, verify), onOutput, true, ct);

    public Task<LoraleiResult> RunsRmAsync(string name, string? outputRoot = null,
        Action<ConsoleLine>? onOutput = null, CancellationToken ct = default) =>
        _process.RunAsync(CommandBuilder.RunsRm(name, outputRoot), onOutput, false, ct);
}
