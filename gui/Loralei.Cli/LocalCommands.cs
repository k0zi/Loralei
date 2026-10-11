using System.Text.Json;
using Loralei.Core;
using Loralei.Core.Config;
using Loralei.Core.Dto;
using Loralei.Core.Models;
using static Loralei.Cli.Terminal;

namespace Loralei.Cli;

/// <summary>The commands that never start Python: everything about models on disk, datasets and runs.</summary>
public static class LocalCommands
{
    internal static OptionSet WithSources(this OptionSet options) =>
        options.Value("--models-dir").Value("--model-path");

    internal static Dictionary<string, object?> SourceOverrides(Args args) => new()
    {
        ["models_dir"] = args.Get("--models-dir"),
        ["model_path"] = args.Get("--model-path"),
    };

    internal static Args Parse(OptionSet options, IReadOnlyList<string> argv, string usage)
    {
        var args = options.Parse(argv);
        return args.Help ? throw new HelpRequested(usage) : args;
    }

    private static string Gib(long bytes) => $"{F(bytes / (double)Disk.Gib)} GiB";

    internal static void PrintJson<T>(T value) => Line(JsonSerializer.Serialize(value, LoraleiInfo.Json));

    public static async Task<int> Models(IReadOnlyList<string> argv, CancellationToken ct)
    {
        if (argv.Count == 0) throw new UsageException("models needs a subcommand: ls, pull or gc");
        var rest = argv.Skip(1).ToList();
        return argv[0] switch
        {
            "ls" => ModelsLs(rest),
            "pull" => await ModelsPull(rest, ct),
            "gc" => ModelsGc(rest),
            "--help" or "-h" => throw new HelpRequested(Help.Models),
            _ => throw new UsageException($"no such subcommand 'models {argv[0]}'"),
        };
    }

    private static int ModelsLs(IReadOnlyList<string> argv)
    {
        var args = Parse(new OptionSet().WithSources().Flag("--json"), argv, Help.Models);
        args.NoMoreThan(0);
        var report = ModelsReport.Build(ConfigResolver.ResolveSources(SourceOverrides(args)));

        if (args.On("--json"))
        {
            PrintJson(report);
            return 0;
        }

        Table(["model", "status", "to download", "found in", "missing components"],
            report.Models.Select(m => (IReadOnlyList<string>)
            [
                m.Key,
                m.Status switch { "ready" => Green(m.Status), "needs disk" => Red(m.Status), _ => Yellow(m.Status) },
                m.MissingBytes > 0 ? Gib(m.MissingBytes) : "-",
                m.FoundIn.Count > 0 ? string.Join(", ", m.FoundIn) : "-",
                m.Missing.Count > 0 ? string.Join(", ", m.Missing.Select(x => $"{x.Component} ({x.Reason})")) : "-",
            ]).ToList(), 2);

        Line();
        Line(Dim($"{Gib(report.FreeBytes)} free on {report.Root}"));
        for (var i = 0; i < report.ModelsDir.Count; i++)
            Line(Dim($"models_dir[{i}]: {report.ModelsDir[i]}{(i == 0 ? "  (downloads land here)" : "")}"));
        if (report.ModelPath is not null) Line(Dim($"model_path: {report.ModelPath}"));
        foreach (var model in report.Models) Line(Dim($"{model.Key}: {model.Label}"));
        foreach (var note in report.Notes) Line(Dim(note));
        return 0;
    }

    private static async Task<int> ModelsPull(IReadOnlyList<string> argv, CancellationToken ct)
    {
        var args = Parse(new OptionSet().WithSources().Flag("--dry-run"), argv, Help.Models);
        args.Required(0, "the model to pull");
        // Every name is checked before the first byte moves: a typo in the third model should
        // not surface after the first two have spent an hour downloading.
        var specs = args.Positional.Distinct().Select(ModelRegistry.Get).ToList();
        var dryRun = args.On("--dry-run");
        var sources = ConfigResolver.ResolveSources(SourceOverrides(args));

        var failed = 0;
        foreach (var spec in specs)
        {
            if (specs.Count > 1) Line(Bold($"== {spec.Key}"));
            try
            {
                if (await PullOne(spec, sources, dryRun, ct) != 0) failed++;
            }
            catch (Exception e) when (e is NotEnoughSpaceException or HttpRequestException or IOException
                                          or InvalidOperationException)
            {
                // One model that does not fit, or is gated, should not stop the others.
                EndLive();
                Line(Red(e.Message));
                failed++;
            }
        }
        return failed == 0 ? 0 : 1;
    }

    private static async Task<int> PullOne(ModelSpec spec, ModelSources sources, bool dryRun, CancellationToken ct)
    {
        var before = ModelResolver.Resolve(spec, sources);
        if (before.Complete)
        {
            Line(Green($"{spec.Key} is already complete."));
            foreach (var component in spec.Components)
                Line($"  {component.Name,-14} {before.Components[component.Name].Repo}");
            return 0;
        }

        Line($"missing: {string.Join(", ", before.Missing.Select(m => m.Component))} (~{Gib(before.MissingBytes)})");
        if (ModelResolver.DownloadTarget(spec, sources) is { } target) Line($"downloading into {target}");

        using var http = ModelDownloader.CreateClient();
        var lastDraw = 0L;
        var after = await new ModelDownloader(http).DownloadMissingAsync(spec, sources, dryRun, p =>
        {
            // Called from several downloads at once and for every megabyte; redraw a few times a second.
            var now = Environment.TickCount64;
            if (now - Interlocked.Read(ref lastDraw) < 200 && p.FilesDone < p.FilesTotal) return;
            Interlocked.Exchange(ref lastDraw, now);
            Live($"{Bold("pull")} {Bar(p.BytesDone, p.BytesTotal)} {Gib(p.BytesDone)} / {Gib(p.BytesTotal)}  " +
                 $"{p.FilesDone}/{p.FilesTotal} files  {Dim(p.Current)}");
        }, ct);
        EndLive();

        if (dryRun)
        {
            Line(Yellow("--dry-run: nothing downloaded, but there is room."));
            return 0;
        }
        if (after.Complete)
        {
            Line(Green($"{spec.Key} is ready."));
            return 0;
        }
        Line(Red($"still missing: {string.Join(", ", after.Missing.Select(m => m.Component))}"));
        return 1;
    }

    private static int ModelsGc(IReadOnlyList<string> argv)
    {
        var args = Parse(
            new OptionSet().Flag("--dry-run", "--no-dry-run").Value("--min-gib").Flag("--yes", null, "-y").Flag("--json"),
            argv, Help.Models);
        args.NoMoreThan(0);
        var report = GcReport.Build(args.Double("--min-gib") ?? 1.0);

        if (args.On("--json"))
        {
            PrintJson(report);
            return 0;
        }
        if (report.Entries.Count == 0)
        {
            Line("nothing above the threshold");
            return 0;
        }

        Table(["size", "what", "path"],
            report.Entries.Select(e => (IReadOnlyList<string>)[Gib(e.Bytes), e.Name, Dim(e.Path)]).ToList(), 0);
        Line();
        Line($"total reclaimable: {Gib(report.TotalBytes)} | currently free: {Gib(report.FreeBytes)}");

        if (args.On("--dry-run", fallback: true))
        {
            Line(Dim("--no-dry-run deletes, after an explicit confirmation per entry."));
            return 0;
        }

        foreach (var entry in report.Entries)
        {
            if (!args.On("--yes") && !Confirm($"delete {entry.Name} ({Gib(entry.Bytes)}) at {entry.Path}?")) continue;
            try
            {
                Directory.Delete(entry.Path, recursive: true);
                Line($"{Green("removed")} {entry.Name}");
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                Line(Red($"could not remove {entry.Name}: {e.Message}"));
            }
        }
        return 0;
    }

    public static int Data(IReadOnlyList<string> argv)
    {
        if (argv is ["--help" or "-h", ..]) throw new HelpRequested(Help.Data);
        if (argv.Count == 0 || argv[0] != "check") throw new UsageException("data needs the subcommand: check");

        var args = Parse(
            new OptionSet().Value("--model", "-m").Value("--resolution").Value("--caption-extension").Flag("--json"),
            argv.Skip(1).ToList(), Help.Data);
        var dataset = args.Required(0, "the dataset directory");
        args.NoMoreThan(1);
        var json = args.On("--json");

        var report = DataReport.Build(dataset, args.Int("--resolution") ?? 1024,
            args.Get("--caption-extension") ?? ".txt",
            json ? null : (done, total) => Live($"{Bold("scan")} {Bar(done, total)} {done}/{total}"),
            // The bucket grid is the model's: 16 px per latent token for some, 32 for others.
            ModelRegistry.Get(args.Get("--model") ?? ModelRegistry.DefaultKey).BucketDivisor);
        EndLive();

        if (json)
        {
            PrintJson(report);
            return report.Pairs > 0 ? 0 : 1;
        }

        Line($"{Bold(report.Pairs.ToString())} image/caption pairs in {dataset}" +
             (report.Problems.Count > 0 ? ", " + Yellow($"{report.Problems.Count} problem(s)") : ""));
        foreach (var problem in report.Problems.Take(20)) Line("  " + Yellow(problem));
        if (report.Problems.Count > 20) Line(Dim($"  ... and {report.Problems.Count - 20} more"));
        if (report.Pairs == 0) return 1;

        Table(["bucket (w x h)", "images", "latent tokens"],
            report.Buckets.Select(b => (IReadOnlyList<string>)
                [$"{b.Width} x {b.Height}", b.Images.ToString(), b.Tokens.ToString()]).ToList(), 1, 2);
        Line(Dim($"captions: {report.Captions}"));
        return 0;
    }

    public static int Runs(IReadOnlyList<string> argv)
    {
        if (argv.Count == 0) throw new UsageException("runs needs a subcommand: ls, show or rm");
        if (argv[0] is "--help" or "-h") throw new HelpRequested(Help.Runs);

        var args = Parse(new OptionSet().Value("--output-root").Flag("--yes", null, "-y"), argv.Skip(1).ToList(), Help.Runs);
        var outputRoot = args.Get("--output-root") ?? "runs";
        var store = new RunStore(new LoraleiPaths("loralei", Directory.GetCurrentDirectory()));

        switch (argv[0])
        {
            case "ls":
            {
                args.NoMoreThan(0);
                var entries = store.List(outputRoot).OrderBy(e => e.Name, StringComparer.Ordinal).ToList();
                if (entries.Count == 0)
                {
                    Line($"no runs under {outputRoot}");
                    return 0;
                }
                Table(["run", "model", "step", "state", "updated"],
                    entries.Select(e => (IReadOnlyList<string>)
                    [
                        e.Name, e.Model, e.Step.ToString(), e.State,
                        Dim(e.Updated?.ToLocalTime().ToString("yyyy-MM-dd HH:mm") ?? "-"),
                    ]).ToList(), 2);
                return 0;
            }
            case "show":
            {
                var name = args.Required(0, "the run name");
                var dir = Path.Combine(outputRoot, name);
                if (!Directory.Exists(dir)) return NoSuchRun(name, outputRoot);

                var manifest = store.ReadManifest(name, outputRoot);
                if (manifest is null) Line(Yellow($"{dir} has no manifest.json"));
                else
                    Table(["key", "value"],
                        manifest.AllFields().Select(f => (IReadOnlyList<string>)[f.Key, f.Value]).ToList());

                foreach (var file in Directory.EnumerateFiles(dir, "*.safetensors", SearchOption.AllDirectories)
                             .Order(StringComparer.Ordinal))
                    Line($"  {Path.GetRelativePath(dir, file)}  ({F(new FileInfo(file).Length / 1048576.0)} MiB)");
                return 0;
            }
            case "rm":
            {
                var name = args.Required(0, "the run name");
                var dir = Path.Combine(outputRoot, name);
                if (!Directory.Exists(dir)) return NoSuchRun(name, outputRoot);

                var size = Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories).Sum(f => new FileInfo(f).Length);
                if (!args.On("--yes") && !Confirm($"delete {dir} ({F(size / 1048576.0)} MiB)?")) return 0;
                Directory.Delete(dir, recursive: true);
                Line($"{Green("removed")} {dir}");
                return 0;
            }
            default:
                throw new UsageException($"no such subcommand 'runs {argv[0]}'");
        }
    }

    private static int NoSuchRun(string name, string outputRoot)
    {
        Line(Red($"no run named {name} under {outputRoot}"));
        return 1;
    }
}
