using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Loralei.Core;
using Loralei.Gui.ViewModels;
using Loralei.Tests;
using Xunit;

namespace Loralei.Gui.Tests;

/// <summary>
/// Drives the real view models against the real CLI. This is the end-to-end claim the whole
/// design rests on: the GUI writes loralei.yaml, spawns loralei, and the live readout comes back
/// from the files the trainer writes.
/// <para>
/// Every test runs in its own <see cref="TempWorkspace"/>, so nothing here can touch the
/// developer's loralei.yaml or runs directory - not even if a run is interrupted before its
/// cleanup gets to run.
/// </para>
/// <para>The runs that need the GPU skip themselves when the base weights are not on this machine.</para>
/// </summary>
public class TrainPageTests
{
    private static readonly LoraleiPaths Real = LoraleiPaths.Discover(null, null);
    private static bool Available => Real.Problem() is null && File.Exists(Real.Executable);

    /// <summary>
    /// The CLI is built beside these tests, so it is always there; what a machine may lack is
    /// the base weights. Resolved the way the CLI will resolve them, from the real workspace.
    /// </summary>
    private static bool WeightsAreLocal => Loralei.Core.Models.ModelResolver.Resolve(
        Loralei.Core.Models.ModelRegistry.QwenImage,
        Loralei.Core.Config.ConfigResolver.ResolveSources(projectConfig: Real.ProjectConfig)).Complete;

    private const string NoWeights = "the qwen-image base weights are not on this machine";

    private static TempWorkspace Workspace() => TempWorkspace.Create(Real);

    [AvaloniaFact]
    public void ShellExposesEveryCommandAsAPage()
    {
        using var workspace = Workspace();
        var shell = new MainWindowViewModel();

        Assert.Equal(
            ["Doctor", "Models", "Resize", "Caption", "Data", "Cache", "Train", "Sample", "Export", "Runs"],
            shell.Pages.Select(p => p.Title));
        Assert.Equal("Doctor", shell.SelectedPage?.Title);
    }

    [AvaloniaFact]
    public void PageArgumentOpensStraightOntoThatPage()
    {
        using var workspace = Workspace();
        var shell = new MainWindowViewModel();

        shell.SelectStartupPage(["--page", "runs"]);
        Assert.Equal("Runs", shell.SelectedPage?.Title);
    }

    [AvaloniaFact]
    public void TrainFormOnlySendsWhatWasTouched()
    {
        using var workspace = Workspace();
        var shell = new MainWindowViewModel();
        var train = shell.Pages.OfType<TrainViewModel>().Single();

        train.Dataset = "/tmp";
        train.Name = "preview-check";
        train.Rank = 8;
        train.RefreshPreviewCommand.Execute(null);

        Assert.Contains("train /tmp --name preview-check", train.CommandPreview);
        Assert.Contains("--rank 8", train.CommandPreview);
        Assert.DoesNotContain("--lr", train.CommandPreview);

        // A flagless setting must appear in the YAML preview instead of the command line.
        train.FlowShift = 2.5;
        train.RefreshPreviewCommand.Execute(null);
        Assert.DoesNotContain("--flow-shift", train.CommandPreview);
        Assert.Contains("flow_shift: 2.5", train.YamlPreview);
    }

    [AvaloniaFact]
    public async Task ChoosingALoraTypeFillsInTheRecommendedSettings()
    {
        using var workspace = Workspace();
        var shell = new MainWindowViewModel();
        var train = shell.Pages.OfType<TrainViewModel>().Single();
        var dataset = TinyDataset();
        try
        {
            train.Dataset = dataset;
            await train.RefreshImageCountAsync();
            Assert.Equal(4, train.ImageCount);
            Assert.Null(train.Rank);

            train.LoraKind = Recommender.Character;
            Assert.Equal(16, train.Rank);
            Assert.Equal(1000, train.Steps);
            Assert.Equal(250, train.Epochs);

            // A value typed by hand survives the next recommendation; the rest follow it.
            train.LearningRate = 3e-4;
            train.LoraKind = Recommender.Style;
            Assert.Equal(32, train.Rank);
            Assert.Equal(1500, train.Steps);
            Assert.Equal(3e-4, train.LearningRate);

            train.ApplyRecommendedCommand.Execute(null);
            Assert.NotEqual(3e-4, train.LearningRate);

            // Back to custom clears what was recommended, so the preset decides again.
            train.LoraKind = Recommender.Custom;
            Assert.Null(train.Rank);
            Assert.Null(train.Steps);
        }
        finally { Discard(dataset); }
    }

    [AvaloniaFact]
    public async Task EpochsAreTurnedIntoStepsForTheCommand()
    {
        using var workspace = Workspace();
        var shell = new MainWindowViewModel();
        var train = shell.Pages.OfType<TrainViewModel>().Single();
        var dataset = TinyDataset();
        try
        {
            train.Dataset = dataset;
            train.Name = "epochs";
            await train.RefreshImageCountAsync();

            train.Steps = 40;
            train.LengthUnit = "Epochs";
            Assert.Equal(10, train.Epochs);          // the length carries over with the unit

            train.Epochs = 25;
            train.RefreshPreviewCommand.Execute(null);
            Assert.Contains("--steps 100", train.CommandPreview);

            train.BatchSize = 2;
            train.RefreshPreviewCommand.Execute(null);
            Assert.Contains("--steps 50", train.CommandPreview);
        }
        finally { Discard(dataset); }
    }

    [AvaloniaFact]
    public async Task EpochsWithoutImagesAreRefusedRatherThanGuessed()
    {
        using var workspace = Workspace();
        var shell = new MainWindowViewModel();
        var train = shell.Pages.OfType<TrainViewModel>().Single();
        var empty = Directory.CreateTempSubdirectory("loralei-gui-empty").FullName;
        try
        {
            train.Dataset = empty;
            train.Name = "no-images";
            await train.RefreshImageCountAsync();
            train.LengthUnit = "Epochs";
            train.Epochs = 10;
            train.DryRunCommand.Execute(null);

            Assert.Contains("no images were found", train.Error);
            Assert.False(shell.IsBusy);
        }
        finally { Discard(empty); }
    }

    [AvaloniaFact]
    public void ValidationErrorsSurfaceBeforeAnythingIsSpawned()
    {
        using var workspace = Workspace();
        var shell = new MainWindowViewModel();
        var train = shell.Pages.OfType<TrainViewModel>().Single();

        train.Dataset = "/tmp";
        train.Name = "bad";
        train.Resolution = 1000;
        train.DryRunCommand.Execute(null);

        Assert.NotNull(train.Error);
        Assert.Contains("multiple of 16", train.Error);
        Assert.False(shell.IsBusy);
    }

    [AvaloniaFact]
    public void WarnsThatNothingIsRecoverableWhenSaveEveryIsZero()
    {
        using var workspace = Workspace();
        var shell = new MainWindowViewModel();
        var train = shell.Pages.OfType<TrainViewModel>().Single();

        Assert.False(train.CheckpointWarning);
        train.SaveEvery = 0;
        Assert.True(train.CheckpointWarning);
        train.SaveEvery = 5;
        Assert.False(train.CheckpointWarning);
    }

    /// <summary>
    /// The form must not eat settings it does not show. Its empty fields mean "inherit", and
    /// an earlier version wrote that emptiness back as a deletion, silently removing keys the
    /// user had put in loralei.yaml by hand.
    /// </summary>
    [AvaloniaFact]
    public void WritingTheProjectConfigPreservesSettingsTheFormNeverShowed()
    {
        using var workspace = Workspace();
        File.WriteAllText(workspace.ProjectConfig,
            """
            # keep me
            models_dir:
              - ~/Develop/ai-models/unsloth/hub

            sample_every: 0
            sample_prompts:
              - "a photo of sks pattern on a wooden table"

            """);

        var shell = new MainWindowViewModel();
        var train = shell.Pages.OfType<TrainViewModel>().Single();
        train.ActivateAsync();

        // Seeded from the file, so the user can see what the workspace already sets.
        Assert.Equal(0, train.SampleEvery);
        Assert.Equal("a photo of sks pattern on a wooden table", train.SamplePrompts);

        train.Rank = 16;
        train.RefreshPreviewCommand.Execute(null);

        Assert.Contains("# keep me", train.YamlPreview);
        Assert.Contains("~/Develop/ai-models/unsloth/hub", train.YamlPreview);
        Assert.Contains("sample_every: 0", train.YamlPreview);
        Assert.Contains("a photo of sks pattern on a wooden table", train.YamlPreview);

        // Rank has a CLI flag, so it belongs on the command line, not in the file.
        Assert.DoesNotContain("rank:", train.YamlPreview);
        Assert.Contains("--rank 16", train.CommandPreview);
    }

    /// <summary>
    /// The pre-flight card. <c>--dry-run</c> resolves the config, reports the cache state and
    /// prints the VRAM/ETA estimate without loading a model or touching the GPU.
    /// </summary>
    [AvaloniaFact]
    public async Task DryRunReportsTheEstimateWithoutTrainingAnything()
    {
        Assert.SkipUnless(Available, "loralei is not installed");
        Assert.SkipUnless(WeightsAreLocal, NoWeights);
        using var cli = CliLock.Acquire();
        using var workspace = Workspace();

        var dataset = TinyDataset();
        try
        {
            var shell = new MainWindowViewModel();
            var train = shell.Pages.OfType<TrainViewModel>().Single();

            train.Dataset = dataset;
            train.Name = "gui-dry";
            train.Model = "qwen-image";
            train.Steps = 500;
            train.Resolution = 256;
            train.Rank = 8;

            var run = Task.Run(() => train.DryRunCommand.ExecuteAsync(null));
            await PumpUntil(() => run.IsCompleted, TimeSpan.FromMinutes(5));
            await run;

            Assert.True(train.Error is null, train.Error);
            Assert.StartsWith("estimate:", train.Estimate);
            Assert.Contains("GiB VRAM", train.Estimate);
            Assert.Contains("500 steps", train.Estimate);
            Assert.StartsWith("cache:", train.CacheStatus);

            // It resolved the run, and stopped before training it.
            var resolved = new RunStore(shell.Paths).ReadResolvedConfig("gui-dry", shell.OutputRoot);
            Assert.NotNull(resolved);
            Assert.Equal(8, FlatYaml.Parse(resolved).Int("rank"));
            Assert.Equal(500, FlatYaml.Parse(resolved).Int("max_steps"));
            Assert.Equal("", train.RunState);
            Assert.Null(train.FinalAdapter);
        }
        finally { Discard(dataset); }
    }

    /// <summary>
    /// A real 12-step run of Qwen-Image at 256px, driven entirely from the view model.
    /// Takes about a minute and a half on a 9060 XT.
    /// </summary>
    [AvaloniaFact]
    public async Task TrainsEndToEndAndReportsLiveProgress()
    {
        Assert.SkipUnless(Available, "loralei is not installed");
        Assert.SkipUnless(WeightsAreLocal, NoWeights);
        using var cli = CliLock.Acquire();
        using var workspace = Workspace();

        var dataset = TinyDataset();
        try
        {
            var shell = new MainWindowViewModel();
            var train = shell.Pages.OfType<TrainViewModel>().Single();

            // output_root has no CLI flag, so setting it here also proves that a flagless
            // setting reaches loralei through the project config.
            var outputRoot = Path.Combine(workspace.Directory, "elsewhere");
            train.OutputRoot = outputRoot;

            train.Dataset = dataset;
            train.Name = "gui-e2e";
            train.Model = "qwen-image";
            train.Steps = 12;
            train.Resolution = 256;
            train.Rank = 4;
            train.Optimizer = "adamw";
            train.BlocksToSwap = 0;
            train.SaveEvery = 6;
            train.LogEvery = 2;

            var run = Task.Run(() => train.StartCommand.ExecuteAsync(null));

            // Drain as the poll timer would, and record that progress really did arrive
            // while the child was still running - not only when it finished.
            var sawProgressMidRun = false;
            await PumpUntil(
                () =>
                {
                    train.PollLiveState();
                    if (!run.IsCompleted && train.Step > 0) sawProgressMidRun = true;
                    return run.IsCompleted;
                },
                TimeSpan.FromMinutes(12), shell, "the run to finish");
            await run;

            Assert.True(sawProgressMidRun, "no step was reported until the run had already ended");
            Assert.True(train.Error is null, train.Error);
            Assert.Equal("done", train.RunState);
            Assert.Equal(12, train.Step);
            Assert.Equal(12, train.MaxSteps);
            Assert.Equal(1.0, train.Progress);

            // The live readout came from runs/<name>/logs/train.jsonl, not from scraping.
            Assert.NotEmpty(train.LossSeries);
            Assert.All(train.LossSeries, loss => Assert.True(loss > 0 && loss < 100, $"loss was {loss}"));
            Assert.True(train.VramGib > 0, "no VRAM was reported");
            Assert.NotEmpty(train.Checkpoints);
            Assert.EndsWith("gui-e2e.safetensors", train.FinalAdapter);
            Assert.True(File.Exists(Path.Combine(outputRoot, "gui-e2e", "gui-e2e.safetensors")));

            var resolved = new RunStore(shell.Paths).ReadResolvedConfig("gui-e2e", outputRoot);
            Assert.NotNull(resolved);
            Assert.Equal(2, FlatYaml.Parse(resolved).Int("log_every"));
            Assert.Equal(outputRoot, FlatYaml.Parse(resolved).Scalar("output_root"));

            // A finished run is resumable, because a checkpoint and its sidecar exist.
            Assert.True(train.CanResume);

            // And the Runs page sees it the same way.
            var runs = shell.Pages.OfType<RunsViewModel>().Single();
            runs.RefreshCommand.Execute(null);
            var entry = Assert.Single(runs.Entries, e => e.Name == "gui-e2e");
            Assert.Equal("done", entry.State);
            Assert.Equal(12, entry.Step);
        }
        finally { Discard(dataset); }
    }

    /// <summary>
    /// Interrupting must actually stop the child and leave the run marked failed, not hang
    /// the GUI. loralei catches the SIGINT, stamps manifest.json and re-raises.
    /// </summary>
    [AvaloniaFact]
    public async Task InterruptingStopsTheRunAndSaysWhatWasLost()
    {
        Assert.SkipUnless(Available, "loralei is not installed");
        Assert.SkipUnless(WeightsAreLocal, NoWeights);
        using var cli = CliLock.Acquire();
        using var workspace = Workspace();

        var dataset = TinyDataset();
        try
        {
            var shell = new MainWindowViewModel();
            var train = shell.Pages.OfType<TrainViewModel>().Single();

            train.Dataset = dataset;
            train.Name = "gui-cancel";
            train.Model = "qwen-image";
            train.Steps = 4000;          // long enough to still be running when we stop it
            train.Resolution = 256;
            train.Rank = 4;
            train.Optimizer = "adamw";
            train.BlocksToSwap = 0;
            train.SaveEvery = 0;         // nothing recoverable, which the page must say
            train.LogEvery = 1;

            var run = Task.Run(() => train.StartCommand.ExecuteAsync(null));

            // Wait until it is genuinely training before interrupting. The poll timer that
            // does this in the running app is a DispatcherTimer, and nothing ticks it under
            // the headless harness, so drive the same drain directly.
            await PumpUntil(
                () => { train.PollLiveState(); return train.Step > 0 || run.IsCompleted; },
                TimeSpan.FromMinutes(10), shell, "the first training step");
            Assert.False(run.IsCompleted, $"the run ended before it could be interrupted: {train.Error}");

            shell.CancelCommand.Execute(null);
            await PumpUntil(() => run.IsCompleted, TimeSpan.FromMinutes(5), shell, "the interrupt to land");
            await run;

            Assert.Equal("failed", train.RunState);
            Assert.Contains("cannot be resumed", train.Status);
            Assert.False(shell.IsBusy);

            // The trainer's own handler recorded the interruption.
            var manifest = new RunStore(shell.Paths).ReadManifest("gui-cancel", shell.OutputRoot);
            Assert.NotNull(manifest);
            Assert.Equal("failed", manifest.State);
        }
        finally { Discard(dataset); }
    }

    /// <summary>
    /// Runs the dispatcher while waiting, so DispatcherTimer ticks are delivered. On timeout
    /// it reports the tail of the child's output, which is the only way to tell a slow start
    /// from a stuck one.
    /// </summary>
    private static async Task PumpUntil(Func<bool> done, TimeSpan timeout,
        MainWindowViewModel? shell = null, string what = "the run to finish")
    {
        var deadline = DateTime.UtcNow + timeout;
        while (!done() && DateTime.UtcNow < deadline)
        {
            Dispatcher.UIThread.RunJobs();
            await Task.Delay(100);
        }
        Dispatcher.UIThread.RunJobs();

        var tail = shell is null ? "" : "\n" + string.Join('\n', shell.Log.Lines.TakeLast(25));
        Assert.True(done(), $"timed out after {timeout.TotalMinutes:F0} min waiting for {what}.{tail}");
    }

    private static void Discard(string directory)
    {
        try { Directory.Delete(directory, recursive: true); }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
    }

    /// <summary>
    /// Four images in two sizes. The dataset scan drops exact duplicates by content hash, so the
    /// pixels have to differ per file or half the set silently disappears.
    /// </summary>
    private static string TinyDataset()
    {
        var dir = Directory.CreateTempSubdirectory("loralei-gui-data").FullName;
        for (var i = 0; i < 4; i++)
        {
            Png.Write(Path.Combine(dir, $"img{i}.png"), 256 + i % 2 * 64, 256, seed: i);
            File.WriteAllText(Path.Combine(dir, $"img{i}.txt"), "a photo of sks pattern on a blue field");
        }
        return dir;
    }
}
