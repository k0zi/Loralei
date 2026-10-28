using Xunit;

namespace Loralei.Core.Tests;

public class CommandBuilderTests
{
    [Fact]
    public void TrainPassesOnlyTheFieldsTheUserSet()
    {
        var args = CommandBuilder.Train(new TrainRequest
        {
            Name = "my-style",
            Dataset = "./data",
            Model = "qwen-image",
            Rank = 32,
        });

        Assert.Equal(["train", "./data", "--name", "my-style", "--model", "qwen-image", "--rank", "32"], args);
        // Untouched fields must not be sent, or they would override the preset and profile.
        Assert.DoesNotContain("--lr", args);
        Assert.DoesNotContain("--steps", args);
    }

    [Fact]
    public void Fp8BaseIsTriState()
    {
        Assert.DoesNotContain("--fp8-base",
            CommandBuilder.Train(new TrainRequest { Name = "n", Dataset = "d" }));
        Assert.Contains("--fp8-base",
            CommandBuilder.Train(new TrainRequest { Name = "n", Dataset = "d", Fp8Base = true }));
        Assert.Contains("--no-fp8-base",
            CommandBuilder.Train(new TrainRequest { Name = "n", Dataset = "d", Fp8Base = false }));
    }

    [Fact]
    public void ZeroValuedOptionsAreStillSent()
    {
        // 0 is meaningful: save_every 0 means "only at the end", blocks_to_swap 0 means "none".
        var args = CommandBuilder.Train(new TrainRequest
        {
            Name = "n", Dataset = "d", SaveEvery = 0, BlocksToSwap = 0, Seed = 0,
        });
        Assert.Contains("--save-every", args);
        Assert.Equal("0", args[args.IndexOf("--save-every") + 1]);
        Assert.Contains("--blocks-to-swap", args);
        Assert.Contains("--seed", args);
    }

    [Fact]
    public void SeveralModelRootsGoThroughAsOnePathSeparatedValue()
    {
        // --models-dir is a single Path on the CLI; as_roots() splits it again on os.pathsep.
        var args = CommandBuilder.Train(new TrainRequest
        {
            Name = "n",
            Dataset = "d",
            ModelsDir = ["/a/hub", "/b/hub"],
        });
        var value = args[args.IndexOf("--models-dir") + 1];
        Assert.Equal($"/a/hub{Path.PathSeparator}/b/hub", value);
    }

    [Fact]
    public void EmptyModelRootsAreOmitted()
    {
        var args = CommandBuilder.Train(new TrainRequest { Name = "n", Dataset = "d", ModelsDir = ["", "  "] });
        Assert.DoesNotContain("--models-dir", args);
    }

    [Fact]
    public void LearningRateSurvivesAsAnExponent()
    {
        var args = CommandBuilder.Train(new TrainRequest { Name = "n", Dataset = "d", LearningRate = 5e-05 });
        Assert.Equal("5E-05", args[args.IndexOf("--lr") + 1]);
    }

    [Fact]
    public void DoctorJsonNeverCombinesWithFix()
    {
        // --fix prints a shell snippet after the JSON, which would make the output unparseable.
        var args = CommandBuilder.Doctor(json: true, thorough: true, fix: true);
        Assert.Equal(["doctor", "--json"], args);
    }

    [Fact]
    public void DoctorInstallIsItsOwnCommand()
    {
        Assert.Equal(["doctor", "--install"], CommandBuilder.DoctorInstall());
    }

    [Fact]
    public void RunsRmAlwaysSkipsTheInteractiveConfirmation()
    {
        Assert.Contains("--yes", CommandBuilder.RunsRm("smoke"));
    }

    [Fact]
    public void ModelsGcOnlyEverLists()
    {
        var args = CommandBuilder.ModelsGc(2.5);
        Assert.Contains("--dry-run", args);
        Assert.DoesNotContain("--no-dry-run", args);
    }

    [Fact]
    public void SeveralModelsArePulledByOneCommand()
    {
        Assert.Equal(["models", "pull", "krea-2-raw", "ltx-2.3", "--dry-run"],
            CommandBuilder.ModelsPull(["krea-2-raw", "ltx-2.3"], dryRun: true));
        Assert.Equal(["models", "pull", "qwen-image"], CommandBuilder.ModelsPull("qwen-image"));
    }

    [Fact]
    public void ListingsTheGuiReadsAskForJson()
    {
        Assert.Equal("--json", CommandBuilder.ModelsLs(json: true)[^1]);
        Assert.Equal("--json", CommandBuilder.ModelsGc(1.0, json: true)[^1]);
        Assert.Equal(["data", "check", "/d", "--resolution", "512", "--json"],
            CommandBuilder.DataCheck("/d", 512, json: true));
        // The form shown to the user stays the readable one.
        Assert.DoesNotContain("--json", CommandBuilder.ModelsLs());
    }

    [Fact]
    public void SamplePromptsRepeatTheFlag()
    {
        var args = CommandBuilder.Sample(["a cat", "a dog"], lora: null, model: null, outDir: null,
            steps: null, cfg: null, width: null, height: null, seed: null, gpu: null);
        Assert.Equal(2, args.Count(a => a == "--prompt"));
        Assert.Contains("a cat", args);
        Assert.Contains("a dog", args);
    }

    [Fact]
    public void PreviewQuotesOnlyWhatNeedsIt()
    {
        var preview = CommandBuilder.Preview("/v/bin/loralei",
            ["sample", "--prompt", "a photo of sks style"]);
        Assert.Equal("/v/bin/loralei sample --prompt 'a photo of sks style'", preview);
    }
}

public class TrainRequestValidationTests
{
    private static TrainRequest Valid() => new() { Name = "n", Dataset = Path.GetTempPath() };

    [Fact]
    public void AcceptsAMinimalRequest() => Assert.Empty(Valid().Validate());

    [Fact]
    public void RejectsAResolutionThatIsNotAMultipleOf16()
    {
        var request = Valid();
        request.Resolution = 1000;
        Assert.Contains(request.Validate(), e => e.Contains("multiple of 16"));
    }

    [Fact]
    public void RejectsARankBelowOne()
    {
        var request = Valid();
        request.Rank = 0;
        Assert.Contains(request.Validate(), e => e.Contains("Rank"));
    }

    [Fact]
    public void RejectsCaptionDropoutOutsideTheUnitInterval()
    {
        var request = Valid();
        request.CaptionDropout = 1.5;
        Assert.Contains(request.Validate(), e => e.Contains("Caption dropout"));
    }

    [Fact]
    public void RejectsSamplingOnTheTrainingCard()
    {
        var request = Valid();
        request.Gpu = 0;
        request.SampleGpu = 0;
        Assert.Contains(request.Validate(), e => e.Contains("Sample GPU"));
    }

    [Fact]
    public void RejectsAnUnknownOptimizer()
    {
        var request = Valid();
        request.Optimizer = "lion";
        Assert.Contains(request.Validate(), e => e.Contains("Optimizer"));
    }

    [Fact]
    public void RejectsAMissingDatasetFolder()
    {
        var request = Valid();
        request.Dataset = "/definitely/not/here";
        Assert.Contains(request.Validate(), e => e.Contains("does not exist"));
    }
}
