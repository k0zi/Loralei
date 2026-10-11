using System.Text.Json;
using System.Text.Json.Nodes;
using Loralei.Core.Config;
using Loralei.Core.Worker;
using Loralei.Tests;
using Xunit;

namespace Loralei.Core.Tests;

/// <summary>
/// The contract with the Python worker, exercised against a stand-in interpreter: a shell
/// script that reads its job from stdin and answers in the worker's own protocol. No torch,
/// no GPU - these pin what the C# side sends and how it reads what comes back.
/// </summary>
public class WorkerTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("loralei-worker").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private PythonWorker Stub(string body)
    {
        var path = Path.Combine(_dir, $"python-{Guid.NewGuid():N}");
        File.WriteAllText(path, "#!/bin/sh\n" + body + "\n");
        if (!OperatingSystem.IsWindows())
            File.SetUnixFileMode(path, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        return new PythonWorker(path, new Dictionary<string, string>
        {
            ["PATH"] = Environment.GetEnvironmentVariable("PATH") ?? "/usr/bin:/bin",
            ["JOB_COPY"] = Path.Combine(_dir, "job.json"),
            ["MARKER"] = "set-by-the-host",
        });
    }

    [Fact]
    public async Task TheJobArrivesOnStdinAndEventsComeBackInOrder()
    {
        var worker = Stub("""
            cat > "$JOB_COPY"
            echo "args: $*" >&2
            echo '{"event": "device", "index": 1, "name": "RX 9060 XT"}'
            echo 'a library printed this'
            echo '{"event": "step", "step": 1, "loss": 0.5}'
            echo "marker=$MARKER" >&2
            """);

        var events = new List<string>();
        var text = new List<string>();
        var result = await worker.RunAsync(new JsonObject { ["job"] = "probe", ["gpu"] = null },
            e => events.Add(e.GetProperty("event").GetString()!), text.Add, TestContext.Current.CancellationToken);

        Assert.True(result.Ok);
        Assert.Equal(["device", "step"], events);
        var printed = string.Concat(text);
        Assert.Contains("args: -m loralei -", printed);
        Assert.Contains("a library printed this", printed);
        // The worker sees exactly the environment it was given - that is how the ROCm flags reach torch.
        Assert.Contains("marker=set-by-the-host", printed);

        using var job = JsonDocument.Parse(File.ReadAllText(Path.Combine(_dir, "job.json")));
        Assert.Equal("probe", job.RootElement.GetProperty("job").GetString());
    }

    [Fact]
    public async Task AFailureCarriesTheWorkersOwnMessage()
    {
        var worker = Stub("""
            cat > /dev/null
            echo '{"event": "error", "message": "RuntimeError: no ROCm device visible to torch"}'
            exit 1
            """);
        var result = await worker.RunAsync(new JsonObject { ["job"] = "probe" }, ct: TestContext.Current.CancellationToken);

        Assert.False(result.Ok);
        Assert.Equal(1, result.ExitCode);
        Assert.Equal("RuntimeError: no ROCm device visible to torch", result.Error);
    }

    [Fact]
    public async Task AMissingInterpreterIsExplainedNotThrown()
    {
        var worker = new PythonWorker(Path.Combine(_dir, "no-such-python"), new Dictionary<string, string>());
        var result = await worker.RunAsync(new JsonObject { ["job"] = "probe" }, ct: TestContext.Current.CancellationToken);

        Assert.False(result.Ok);
        Assert.Contains("LORALEI_PYTHON", result.Error);
    }

    [Fact]
    public async Task CancellingInterruptsTheWorkerAndLetsItCleanUp()
    {
        var cleaned = Path.Combine(_dir, "cleaned-up");
        var worker = Stub($$"""
            trap 'touch "{{cleaned}}"; exit 130' INT
            cat > /dev/null
            echo '{"event": "device", "index": 0}'
            while true; do sleep 0.1; done
            """);

        using var cts = new CancellationTokenSource();
        var started = new TaskCompletionSource();
        var run = worker.RunAsync(new JsonObject { ["job"] = "train" }, _ => started.TrySetResult(), ct: cts.Token);
        await started.Task.WaitAsync(TimeSpan.FromSeconds(20), TestContext.Current.CancellationToken);
        await cts.CancelAsync();

        var result = await run.WaitAsync(TimeSpan.FromSeconds(20), TestContext.Current.CancellationToken);
        Assert.True(result.Cancelled);
        Assert.Equal(130, result.ExitCode);
        // SIGINT, not a kill: the handler ran, which is what stamps a real run's manifest.
        Assert.True(File.Exists(cleaned));
    }

    [Fact]
    public void ATrainJobIsCompleteAndEveryPathInItIsAbsolute()
    {
        var dataset = Directory.CreateDirectory(Path.Combine(_dir, "ds")).FullName;
        Png.Write(Path.Combine(dataset, "a.png"), 128, 96);
        File.WriteAllText(Path.Combine(dataset, "a.txt"), "a circle");
        var model = Path.Combine(_dir, "models", "krea-2-raw");
        foreach (var component in new[] { "transformer", "text_encoder", "vae" })
        {
            Directory.CreateDirectory(Path.Combine(model, component));
            File.WriteAllText(Path.Combine(model, component, "model.safetensors"), "x");
        }
        Directory.CreateDirectory(Path.Combine(model, "scheduler"));
        File.WriteAllText(Path.Combine(model, "scheduler", "scheduler_config.json"), "{}");

        var cfg = ConfigResolver.Resolve("krea-2-raw", overrides: new Dictionary<string, object?>
        {
            ["name"] = "t", ["dataset"] = dataset, ["resolution"] = 256, ["models_dir"] = Path.Combine(_dir, "models"),
            ["output_root"] = Path.Combine(_dir, "runs"),
        }, projectConfig: Path.Combine(_dir, "absent.yaml"));
        var prepared = Prepared.From(cfg);
        var job = WorkerJobs.Train(cfg, prepared, resume: true, cacheFirst: true);

        Assert.Equal("train", job["job"]!.GetValue<string>());
        Assert.True(job["resume"]!.GetValue<bool>());
        Assert.True(job["cache_first"]!.GetValue<bool>());
        Assert.Equal("transformer_blocks", job["spec"]!["blocks_attr"]!.GetValue<string>());
        Assert.Equal(512, job["spec"]!["max_sequence_length"]!.GetValue<int>());
        // The worker is told which family drives the model.
        Assert.Equal("krea2", job["spec"]!["family"]!.GetValue<string>());
        Assert.Contains("gfx1150", job["apu_targets"]!.AsArray().Select(n => n!.GetValue<string>()));

        // The whole resolved config travels with the job, in the keys config.resolved.yaml uses.
        Assert.Equal(256, job["config"]!["resolution"]!.GetValue<int>());
        Assert.Equal(14, job["config"]!["blocks_to_swap"]!.GetValue<int>());
        Assert.True(job["config"]!.AsObject().ContainsKey("swap_h2d_only"));

        var item = job["items"]!.AsArray().Single()!;
        Assert.Equal("a circle", item["caption"]!.GetValue<string>());
        Assert.Equal(16, item["key"]!.GetValue<string>().Length);
        Assert.Equal(2, item["bucket"]!.AsArray().Count);
        Assert.True(Path.IsPathRooted(item["image"]!.GetValue<string>()));
        Assert.True(Path.IsPathRooted(job["cache"]!["latents"]!.GetValue<string>()));

        // The tokenizer was never created: an unresolved component is simply absent from the job.
        var components = job["components"]!.AsObject();
        Assert.True(components.ContainsKey("vae"));
        Assert.True(components.ContainsKey("transformer"));
        Assert.False(components.ContainsKey("tokenizer"));
        Assert.All(components, c => Assert.True(Path.IsPathRooted(c.Value!.GetValue<string>())));

        Assert.Equal((false, "1 latent(s) and 1 text embedding(s) missing"), prepared.CacheStatus());
    }

    [Fact]
    public void TheEventLogIsWhatTheTailerReads()
    {
        var path = Path.Combine(_dir, "run", "logs", "train.jsonl");
        using var tailer = new TrainEventTailer(path);
        using (var log = new EventLog(path))
        {
            log.Emit(new JsonObject { ["event"] = "start", ["max_steps"] = 20 });
            log.Emit(new JsonObject { ["step"] = 1, ["epoch"] = 0, ["loss"] = 0.25, ["vram_gib"] = 7.5 });
            // Read while the writer still holds the file open, as the GUI does during a run.
            var events = tailer.Poll();
            Assert.Equal(2, events.Count);
            Assert.Equal("start", events[0].Event);
            Assert.True(events[1].IsMetric);
            Assert.Equal(0.25, events[1].Loss);
            Assert.True(events[1].Timestamp > 0);
        }

        using (var resumed = new EventLog(path, append: true))
            resumed.Emit(new JsonObject { ["event"] = "done", ["step"] = 20 });
        Assert.Equal("done", Assert.Single(tailer.Poll()).Event);
    }
}

public class RocmEnvironmentTests
{
    [Fact]
    public void IntegratedGpusAreDroppedWhenADiscreteCardExists()
    {
        // The order this machine family really reports: the 890M first, ahead of both cards.
        Assert.Equal([1, 2], RocmEnvironment.SelectDevices(["gfx1150", "gfx1200", "gfx1200"]));
        Assert.Equal([0], RocmEnvironment.SelectDevices(["gfx1150"]));
        Assert.Equal([0, 1], RocmEnvironment.SelectDevices(["gfx1150", "gfx1200"], includeApu: true));
    }

    [Fact]
    public void BootstrapSetsWhatIsMissingAndNeverClobbersTheUser()
    {
        var report = RocmEnvironment.Bootstrap(
            new Dictionary<string, string> { ["HF_HUB_OFFLINE"] = "0", ["HOME"] = "/home/x" },
            ["gfx1150", "gfx1200"]);

        Assert.Equal("0", report.Environment["HF_HUB_OFFLINE"]);
        Assert.Equal("0", report.Preexisting["HF_HUB_OFFLINE"]);
        Assert.Equal("1", report.Applied["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"]);
        Assert.Equal("1", report.Environment["HIP_VISIBLE_DEVICES"]);
        Assert.Contains(report.Notes, n => n.Contains("0:gfx1150"));
        Assert.Equal("/home/x", report.Environment["HOME"]);
        Assert.Contains("export HIP_VISIBLE_DEVICES=1", RocmEnvironment.FixSnippet(report));
    }

    [Fact]
    public void AnExistingDeviceSelectionIsLeftAlone()
    {
        var report = RocmEnvironment.Bootstrap(
            new Dictionary<string, string> { ["HIP_VISIBLE_DEVICES"] = "0" }, ["gfx1150", "gfx1200"]);
        Assert.Equal("0", report.Environment["HIP_VISIBLE_DEVICES"]);
        Assert.DoesNotContain("HIP_VISIBLE_DEVICES", report.Applied.Keys);
        Assert.Contains(report.Notes, n => n.Contains("leaving it alone"));
    }
}

public class EstimatorTests
{
    private static RunConfig Config(int swap, bool fp8 = false, string model = "krea-2-raw") => new()
    {
        Name = "n", Dataset = "d", Model = model, Resolution = 1024, BlocksToSwap = swap, Fp8Base = fp8,
    };

    [Fact]
    public void SwappingBlocksAndFp8StorageBothLowerThePeak()
    {
        var all = Estimator.For(Config(0)).PeakGib;
        var swapped = Estimator.For(Config(14)).PeakGib;
        Assert.True(swapped < all);
        Assert.True(Estimator.For(Config(14, fp8: true)).PeakGib < swapped);
        // What leaves the card is pinned in host RAM instead.
        Assert.Equal(0, Estimator.For(Config(0)).HostGib);
        Assert.True(Estimator.For(Config(14)).HostGib > 0);
    }

    [Fact]
    public void EveryModelHasAnEstimateAndSaysItIsNotMeasured()
    {
        Assert.All(LoraleiVocabulary.Models, model =>
        {
            var cfg = Config(0, model: model);
            var estimate = Estimator.For(cfg);
            Assert.True(estimate.PeakGib > 0, model);
            Assert.Contains("extrapolated", Estimator.Format(estimate, cfg));
        });
    }

    [Fact]
    public void AnOverBudgetRunIsFlaggedWithTheWayOut()
    {
        var cfg = Config(0);
        var line = Estimator.Format(Estimator.For(cfg), cfg, deviceGib: 16);
        Assert.Contains(Estimator.OverBudgetMarker, line);
        Assert.Contains("--blocks-to-swap", line);

        var fits = Config(24, fp8: true);
        Assert.DoesNotContain(Estimator.OverBudgetMarker, Estimator.Format(Estimator.For(fits), fits, 16));
    }
}
