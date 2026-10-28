using System.Text.Json;
using System.Text.Json.Nodes;
using Loralei.Core;
using Loralei.Core.Config;
using Loralei.Core.Diagnostics;
using Loralei.Core.Lora;
using Loralei.Core.Models;
using Loralei.Core.Worker;
using static Loralei.Cli.Terminal;

namespace Loralei.Cli;

/// <summary>
/// The commands that need torch. Each one decides everything it can here - config, model
/// folders, dataset, buckets, cache paths - and hands the Python worker a finished job.
/// </summary>
public static class WorkerCommands
{
    private sealed record Device(int Index, string Name, string Arch, double FreeGib, double TotalGib)
    {
        public static Device From(JsonElement e) => new(
            e.GetProperty("index").GetInt32(),
            e.GetProperty("name").GetString() ?? "",
            e.GetProperty("arch").GetString() ?? "",
            e.GetProperty("free_gib").GetDouble(),
            e.GetProperty("total_gib").GetDouble());

        /// <summary>How torch names it. A negative index is the worker's CPU smoke-test device.</summary>
        public string Torch => Index < 0 ? "cpu" : $"cuda:{Index}";
    }

    private static string Kind(JsonElement e) =>
        e.TryGetProperty("event", out var kind) ? kind.GetString() ?? "" : "";

    private static PythonWorker Worker(out EnvReport env)
    {
        env = RocmEnvironment.Bootstrap();
        return new PythonWorker(PythonWorker.FindPython(), env.Environment);
    }

    /// <summary>Turns a finished worker into an exit code, saying why when it failed.</summary>
    private static int Finish(WorkerResult result)
    {
        EndLive();
        if (result.Ok) return 0;
        if (result.Cancelled)
        {
            Line(Yellow("interrupted"));
            return 130;
        }
        if (result.Error is { } error) Line(Red(error));
        return result.ExitCode != 0 ? result.ExitCode : 1;
    }

    private static string Missing(ResolvedModel resolved) =>
        string.Join(", ", resolved.Missing.Select(m => m.Component));

    // ------------------------------------------------------------------- doctor

    public static async Task<int> Doctor(IReadOnlyList<string> argv, CancellationToken ct)
    {
        var args = LocalCommands.Parse(new OptionSet().Flag("--json").Flag("--fix").Flag("--thorough").Flag("--install"), argv, Help.Doctor);
        args.NoMoreThan(0);
        if (args.On("--install")) return await Install(ct);
        var json = args.On("--json");

        var env = RocmEnvironment.Bootstrap();
        // With --json, stdout is the payload; worker chatter would make it unparseable for
        // anything that reads both streams.
        var report = await Loralei.Core.Diagnostics.Doctor.RunAsync(!args.On("--thorough"), env, PythonWorker.FindPython(),
            json ? null : Passthrough, ct);
        ct.ThrowIfCancellationRequested();

        if (json)
        {
            Line(JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
        }
        else
        {
            Table(["", "check", "detail"],
                report.Checks.Select(c => (IReadOnlyList<string>)[Mark(c.Status), Bold(c.Name), c.Detail]).ToList());
            foreach (var check in report.Checks.Where(c => c.Remedy.Length > 0 && c.Status is "fail" or "warn"))
                Line($"{Paint(check.Status, $"-> {check.Name}:")} {check.Remedy}");
            if (report.Notes.Count > 0)
            {
                Line();
                foreach (var note in report.Notes) Line(Dim($"note: {note}"));
            }
        }

        if (args.On("--fix"))
        {
            Line();
            Line(RocmEnvironment.FixSnippet(env));
        }
        return report.Ok ? 0 : 1;
    }

    /// <summary>Sets up the worker's Python environment; the checks are a separate run.</summary>
    private static async Task<int> Install(CancellationToken ct)
    {
        var plan = WorkerInstaller.Plan();
        Line($"installing the worker into {Bold(plan.Venv)}");
        if (Environment.GetEnvironmentVariable("LORALEI_PYTHON") is { Length: > 0 } configured &&
            Path.GetFullPath(configured) != plan.Python)
            Line(Yellow($"note: LORALEI_PYTHON={configured} is set and will be used instead of this environment"));

        var code = await WorkerInstaller.RunAsync(plan, step =>
        {
            Line();
            Line(Bold($"== {step.Title}"));
            Line(Dim("$ " + string.Join(' ', step.Command)));
        }, Passthrough, ct);

        Line();
        if (code != 0)
        {
            Line(Red($"the install stopped with exit code {code}; fix the error above and run it again"));
            return code;
        }
        Line(Green("worker installed.") + " Run `loralei doctor` to check it.");
        return 0;
    }

    private static string Paint(string status, string text) => status switch
    {
        "ok" => Green(text), "warn" => Yellow(text), "fail" => Red(text), _ => Dim(text),
    };

    private static string Mark(string status) => Paint(status, status switch
    {
        "ok" => "PASS", "warn" => "WARN", "fail" => "FAIL", _ => "SKIP",
    });

    // -------------------------------------------------------------------- cache

    public static async Task<int> Cache(IReadOnlyList<string> argv, CancellationToken ct)
    {
        var args = LocalCommands.Parse(
            new OptionSet().Value("--model", "-m").Value("--resolution").Value("--gpu").Value("--vae-device")
                .WithSources().Flag("--force"),
            argv, Help.Cache);
        var dataset = args.Required(0, "the dataset directory");
        args.NoMoreThan(1);
        var model = args.Get("--model") ?? ModelRegistry.DefaultKey;

        var overrides = LocalCommands.SourceOverrides(args);
        overrides["name"] = "_cache";
        overrides["dataset"] = dataset;
        overrides["resolution"] = args.Int("--resolution");
        overrides["gpu"] = args.Int("--gpu");
        overrides["vae_device"] = args.Get("--vae-device");
        var cfg = ConfigResolver.Resolve(model, overrides: overrides);

        var prepared = Scan(cfg);
        if (prepared.Scan.Items.Count == 0)
        {
            Line(Red($"no usable image/caption pairs found in {dataset}"));
            return 1;
        }
        if (prepared.Resolved.Missing.Count > 0)
        {
            Line(Red($"{model} is missing: {Missing(prepared.Resolved)}"));
            Line($"run: loralei models pull {model}");
            return 1;
        }

        Line($"{prepared.Scan.Items.Count} pairs, buckets: {prepared.Plan.Summary()}");

        var progress = new CacheProgress();
        var result = await Worker(out _).RunAsync(WorkerJobs.Cache(cfg, prepared, args.On("--force")), e =>
        {
            switch (Kind(e))
            {
                case "device":
                    var device = Device.From(e);
                    var where = cfg.VaeDevice == "cpu" ? "cpu" : device.Torch;
                    Line($"VAE on {Bold(where)}, text encoder on {Bold(device.Torch)}");
                    break;
                case "progress": progress.Draw(e); break;
                case "cached": Line(CacheProgress.Summary(e)); break;
            }
        }, Passthrough, ct);

        var code = Finish(result);
        if (code == 0) Line(Dim($"cache: {prepared.Layout.Root}"));
        return code;
    }

    private static Prepared Scan(RunConfig cfg)
    {
        var prepared = Prepared.From(cfg, (done, total) => Live($"{Bold("scan")} {Bar(done, total)} {done}/{total}"));
        EndLive();
        return prepared;
    }

    /// <summary>The two caching tasks on one status line.</summary>
    private sealed class CacheProgress
    {
        private (int Done, int Total) _latents, _text;

        public void Draw(JsonElement e)
        {
            var value = (e.GetProperty("done").GetInt32(), e.GetProperty("total").GetInt32());
            if (e.GetProperty("task").GetString() == "latents") _latents = value;
            else _text = value;
            Live($"{Bold("latents")} {Bar(_latents.Done, _latents.Total, 16)} {_latents.Done}/{_latents.Total}   " +
                 $"{Bold("text")} {Bar(_text.Done, _text.Total, 16)} {_text.Done}/{_text.Total}");
        }

        public static string Summary(JsonElement e)
        {
            var latents = e.GetProperty("latents");
            var text = e.GetProperty("text");
            return $"{Green("latents")} {latents.GetProperty("written").GetInt32()} written, " +
                   $"{latents.GetProperty("reused").GetInt32()} reused | " +
                   $"{Green("text")} {text.GetProperty("written").GetInt32()} written, " +
                   $"{text.GetProperty("reused").GetInt32()} reused";
        }
    }

    // -------------------------------------------------------------------- train

    public static async Task<int> Train(IReadOnlyList<string> argv, CancellationToken ct)
    {
        var args = LocalCommands.Parse(
            new OptionSet()
                .Value("--name", "-n").Value("--model", "-m").Value("--profile").Value("--rank").Value("--steps")
                .Value("--resolution").Value("--lr").Value("--batch-size").Value("--blocks-to-swap")
                .Flag("--fp8-base", "--no-fp8-base").Value("--optimizer").Value("--trigger").Value("--vae-device")
                .WithSources().Value("--gpu").Value("--save-every").Value("--seed")
                .Flag("--resume").Flag("--dry-run"),
            argv, Help.Train);
        var dataset = args.Required(0, "the dataset directory");
        args.NoMoreThan(1);
        var name = args.Get("--name") ?? throw new UsageException("--name is required");
        var model = args.Get("--model") ?? ModelRegistry.DefaultKey;
        var resume = args.On("--resume");

        var overrides = LocalCommands.SourceOverrides(args);
        overrides["name"] = name;
        overrides["dataset"] = dataset;
        overrides["rank"] = args.Int("--rank");
        overrides["max_steps"] = args.Int("--steps");
        overrides["resolution"] = args.Int("--resolution");
        overrides["learning_rate"] = args.Double("--lr");
        overrides["batch_size"] = args.Int("--batch-size");
        overrides["blocks_to_swap"] = args.Int("--blocks-to-swap");
        overrides["fp8_base"] = args.Flag("--fp8-base");
        overrides["optimizer"] = args.Get("--optimizer");
        overrides["trigger_word"] = args.Get("--trigger");
        overrides["vae_device"] = args.Get("--vae-device");
        overrides["gpu"] = args.Int("--gpu");
        overrides["save_every"] = args.Int("--save-every");
        overrides["seed"] = args.Int("--seed");
        var cfg = ConfigResolver.Resolve(model, args.Get("--profile") ?? "balanced", overrides);

        var prepared = Scan(cfg);
        if (prepared.Scan.Items.Count == 0)
        {
            Line(Red($"no usable image/caption pairs in {dataset}"));
            foreach (var problem in prepared.Scan.Problems.Take(10)) Line("  " + Yellow(problem));
            return 1;
        }
        if (prepared.Resolved.Missing.Count > 0)
        {
            Line($"{Red($"{model} is missing: {Missing(prepared.Resolved)}")} -> loralei models pull {model}");
            return 1;
        }

        var images = prepared.Scan.Items.Count;
        Line($"{Bold(cfg.Name)}  model={cfg.Model} rank={cfg.Rank} steps={cfg.MaxSteps} " +
             $"res={cfg.Resolution} lr={cfg.LearningRate:G} optimizer={cfg.Optimizer}");
        Line($"{images} images -> buckets {prepared.Plan.Summary()} | " +
             $"blocks_to_swap={cfg.BlocksToSwap} fp8_base={cfg.Fp8Base} grad_ckpt={cfg.GradientCheckpointing}");

        void Announce(Device device)
        {
            Line($"device {device.Torch} ({device.Name}, {device.Arch}, {F(device.FreeGib)} GiB free)");
            if (prepared.Scan.Problems.Count > 0)
                Line(Yellow($"{prepared.Scan.Problems.Count} dataset problem(s); run `loralei data check {dataset}` for details"));
            Line($"estimate: {Estimator.Format(Estimator.For(cfg), cfg, device.TotalGib)}");
        }

        var worker = Worker(out _);
        var (cached, why) = prepared.CacheStatus();

        if (args.On("--dry-run"))
        {
            Device? found = null;
            var probe = await worker.RunAsync(WorkerJobs.Probe(cfg.Gpu), e =>
            {
                if (Kind(e) == "device") found = Device.From(e);
            }, Passthrough, ct);
            if (!probe.Ok || found is null) return Finish(probe) is var code and not 0 ? code : 1;

            Announce(found);
            Line(Dim($"resolved config -> {cfg.WriteResolved()}"));
            Line($"cache: {why}");
            Line(Yellow("--dry-run: stopping before training."));
            return 0;
        }

        if (!cached) Line($"caching ({why})");
        cfg.WriteResolved();

        using var events = new EventLog(Path.Combine(cfg.RunDir, "logs", "train.jsonl"), append: resume);
        var cacheProgress = new CacheProgress();
        var started = false;
        JsonElement? done = null;

        void Start()
        {
            if (started) return;
            started = true;
            events.Emit(new JsonObject
            {
                ["event"] = "start", ["name"] = cfg.Name, ["model"] = cfg.Model, ["max_steps"] = cfg.MaxSteps,
                ["log_every"] = cfg.LogEvery, ["images"] = images, ["resume"] = resume,
            });
        }

        var result = await worker.RunAsync(WorkerJobs.Train(cfg, prepared, resume, cacheFirst: !cached), e =>
        {
            switch (Kind(e))
            {
                case "device":
                    Announce(Device.From(e));
                    if (cached) Start();
                    break;
                case "progress": cacheProgress.Draw(e); break;
                case "cached":
                    Line(CacheProgress.Summary(e));
                    Start();
                    break;
                case "step":
                {
                    Start();
                    var step = e.GetProperty("step").GetInt32();
                    var loss = e.GetProperty("loss").GetDouble();
                    var vram = e.GetProperty("vram_gib").GetDouble();
                    Live($"{Bold("train")} {Bar(step, cfg.MaxSteps)} {step}/{cfg.MaxSteps}  " +
                         $"loss {F(loss, "F4")}  vram {F(vram)} GiB");
                    // Every step would be thousands of lines; log_every means "how often is
                    // this run worth recording".
                    if (step == 1 || step == cfg.MaxSteps || (cfg.LogEvery > 0 && step % cfg.LogEvery == 0))
                        events.Emit(new JsonObject
                        {
                            ["step"] = step, ["epoch"] = e.GetProperty("epoch").GetInt32(),
                            ["loss"] = loss, ["vram_gib"] = vram,
                        });
                    break;
                }
                case "save":
                {
                    Start();
                    var path = e.GetProperty("path").GetString() ?? "";
                    var step = e.GetProperty("step").GetInt32();
                    Line(Dim($"saved {Path.GetFileName(path)} @ step {step}"));
                    events.Emit(new JsonObject { ["event"] = "save", ["step"] = step, ["path"] = path });
                    break;
                }
                case "done":
                    done = e.Clone();
                    events.Emit(JsonNode.Parse(e.GetRawText())!.AsObject());
                    break;
            }
        }, Passthrough, ct);

        var exit = Finish(result);
        if (exit != 0 || done is not { } summary) return exit != 0 ? exit : 1;

        var loss = summary.GetProperty("loss").ValueKind == JsonValueKind.Number ? summary.GetProperty("loss").GetDouble() : 0;
        Line($"{Green("done")} {summary.GetProperty("step").GetInt32()} steps in " +
             $"{F(summary.GetProperty("seconds").GetDouble() / 60)} min, loss {F(loss, "F4")}, " +
             $"peak VRAM {F(summary.GetProperty("vram_gib").GetDouble(), "F2")} GiB");
        Line(Bold(Path.Combine(cfg.RunDir, cfg.Name + ".safetensors")));
        return 0;
    }

    // ------------------------------------------------------------------- sample

    public static async Task<int> Sample(IReadOnlyList<string> argv, CancellationToken ct)
    {
        var args = LocalCommands.Parse(
            new OptionSet().Value("--prompt", "-p").Value("--lora").Value("--model", "-m").Value("--out", "-o")
                .Value("--steps").Value("--cfg").Value("--width").Value("--height").Value("--seed").Value("--gpu")
                .WithSources(),
            argv, Help.Sample);
        args.NoMoreThan(0);
        var prompts = args.All("--prompt");
        if (prompts.Count == 0) throw new UsageException("at least one --prompt is required");

        var lora = args.Get("--lora");
        if (lora is not null && !File.Exists(lora))
        {
            // Checked here rather than left to diffusers, which would read a missing path
            // as a Hugging Face repo id and try to fetch it.
            Line(Red($"LoRA file not found: {lora}"));
            return 1;
        }

        var model = args.Get("--model") ?? ModelRegistry.DefaultKey;
        var resolved = ModelResolver.Resolve(ModelRegistry.Get(model),
            ConfigResolver.ResolveSources(LocalCommands.SourceOverrides(args)));
        if (resolved.Missing.Count > 0)
        {
            Line($"{Red($"{model} is missing: {Missing(resolved)}")} -> loralei models pull {model}");
            return 1;
        }

        var verified = true;
        var job = WorkerJobs.Sample(resolved, prompts, args.Get("--out") ?? "samples", lora,
            args.Int("--steps") ?? 30, args.Double("--cfg") ?? 4.0, args.Int("--width") ?? 1024,
            args.Int("--height") ?? 1024, args.Int("--seed") ?? 0, args.Int("--gpu"));
        var result = await Worker(out _).RunAsync(job, e =>
        {
            switch (Kind(e))
            {
                case "device":
                    var device = Device.From(e);
                    Line($"sampling on {device.Torch} ({device.Name})");
                    break;
                case "image": Line($"  {e.GetProperty("path").GetString()}"); break;
                case "verify":
                    verified = e.GetProperty("ok").GetBoolean();
                    var text = $"LoRA load: {e.GetProperty("summary").GetString()}";
                    Line(verified ? Green(text) : Red(text));
                    break;
            }
        }, Passthrough, ct);

        var code = Finish(result);
        return code == 0 && !verified ? 1 : code;
    }

    // ------------------------------------------------------------------- export

    public static async Task<int> Export(IReadOnlyList<string> argv, CancellationToken ct)
    {
        var args = LocalCommands.Parse(
            new OptionSet().Value("--format").Value("--output-root").Flag("--verify", "--no-verify"),
            argv, Help.Export);
        var run = args.Required(0, "a run name, or a path to a .safetensors file");
        args.NoMoreThan(1);
        var format = args.Get("--format") ?? "both";
        if (!LoraleiVocabulary.ExportFormats.Contains(format))
            throw new UsageException($"--format must be one of {string.Join(", ", LoraleiVocabulary.ExportFormats)}");

        var source = run;
        if (!File.Exists(source))
        {
            source = Path.Combine(args.Get("--output-root") ?? "runs", run, $"{run}.safetensors");
            if (!File.Exists(source))
            {
                Line(Red($"no LoRA found at {source}"));
                return 1;
            }
        }
        var modelKey = ManifestModel(Path.GetDirectoryName(Path.GetFullPath(source))!) ?? ModelRegistry.DefaultKey;

        var last = source;
        if (format is "diffusers" or "both") Line($"{Green("diffusers")} {source}");
        if (format is "comfyui" or "both")
        {
            last = Path.Combine(Path.GetDirectoryName(source) ?? "",
                Path.GetFileNameWithoutExtension(source) + "_comfy.safetensors");
            var count = ComfyConverter.ConvertFile(source, last);
            Line($"{Green("comfyui")}   {last}  ({count} tensors)");
        }

        if (!args.On("--verify", fallback: true)) return 0;

        var resolved = ModelResolver.Resolve(ModelRegistry.Get(modelKey), ConfigResolver.ResolveSources());
        if (resolved.Components["transformer"].Missing)
        {
            Line(Yellow("base model not local; skipping the load-back check"));
            return 0;
        }

        Line(Dim("loading the base transformer to verify the adapter attaches..."));
        var ok = false;
        var result = await Worker(out _).RunAsync(WorkerJobs.Verify(resolved, source), e =>
        {
            if (Kind(e) != "verify") return;
            ok = e.GetProperty("ok").GetBoolean();
            var text = $"verify: {e.GetProperty("summary").GetString()}";
            Line(ok ? Green(text) : Red(text));
        }, Passthrough, ct);

        var code = Finish(result);
        if (code == 0) Line(Dim($"comfy file has {SafeTensorsFile.Keys(last).Count} tensors"));
        return code == 0 && !ok ? 1 : code;
    }

    /// <summary>The model a run was trained on, from the manifest beside its adapter.</summary>
    private static string? ManifestModel(string runDir)
    {
        try
        {
            var path = Path.Combine(runDir, "manifest.json");
            if (!File.Exists(path)) return null;
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            return doc.RootElement.TryGetProperty("model", out var model) ? model.GetString() : null;
        }
        catch (Exception e) when (e is IOException or JsonException) { return null; }
    }
}
