using Loralei.Core.Config;
using Xunit;

namespace Loralei.Core.Tests;

public class ConfigResolverTests : IDisposable
{
    private readonly DirectoryInfo _dir = Directory.CreateTempSubdirectory("loralei-config");

    public void Dispose() => _dir.Delete(recursive: true);

    /// <summary>A project config that is not there, so the developer's own loralei.yaml never leaks in.</summary>
    private string NoProject => Path.Combine(_dir.FullName, "absent.yaml");

    private string Project(string yaml)
    {
        var path = Path.Combine(_dir.FullName, "loralei.yaml");
        File.WriteAllText(path, yaml);
        return path;
    }

    private RunConfig Resolve(string model = "krea-2-raw", string profile = "balanced",
        Dictionary<string, object?>? overrides = null, string? project = null)
    {
        var merged = new Dictionary<string, object?> { ["name"] = "x", ["dataset"] = _dir.FullName };
        foreach (var (key, value) in overrides ?? []) merged[key] = value;
        return ConfigResolver.Resolve(model, profile, merged, project ?? NoProject);
    }

    [Fact]
    public void PresetSuppliesModelDefaults()
    {
        var cfg = Resolve();
        Assert.Equal("krea-2-raw", cfg.Model);
        Assert.Equal(2.5, cfg.FlowShift);
        Assert.Equal(14, cfg.BlocksToSwap);
        Assert.Equal(32, cfg.Rank);
        Assert.Equal(1e-4, cfg.LearningRate);
    }

    [Fact]
    public void QwenPresetDiffers()
    {
        var cfg = Resolve("qwen-image");
        Assert.Equal(2.2, cfg.FlowShift);
        Assert.True(cfg.Fp8Base);
        Assert.True(cfg.BlocksToSwap > 40);
        Assert.Equal(5e-5, cfg.LearningRate);
    }

    [Fact]
    public void FlagsBeatTheProfileWhichBeatsThePreset()
    {
        var cfg = Resolve(profile: "lowmem", overrides: new() { ["rank"] = 8 });
        Assert.Equal(8, cfg.Rank);
        Assert.True(cfg.Fp8Base);
        Assert.Equal(24, cfg.BlocksToSwap);
    }

    [Fact]
    public void FlagsThatWereNotPassedAreIgnored()
    {
        Assert.Equal(32, Resolve(overrides: new() { ["rank"] = null }).Rank);
    }

    [Fact]
    public void ProjectConfigBeatsThePresetAndFlagsBeatTheProjectConfig()
    {
        var project = Project("rank: 48\nwarmup_steps: 7\nsample_prompts:\n  - \"a cat\"\n  - a dog\n");
        var fromFile = Resolve(project: project);
        Assert.Equal(48, fromFile.Rank);
        Assert.Equal(7, fromFile.WarmupSteps);
        Assert.Equal(["a cat", "a dog"], fromFile.SamplePrompts);

        Assert.Equal(4, Resolve(overrides: new() { ["rank"] = 4 }, project: project).Rank);
    }

    [Fact]
    public void ArchitectureDefaultsComeFromTheRegistry()
    {
        var cfg = Resolve("qwen-image");
        Assert.StartsWith("^transformer_blocks", cfg.TargetRegex);
        Assert.Contains("_mod", cfg.ExcludeRegex);

        // Krea 2 names exactly what it adapts, and so has nothing left to exclude.
        Assert.Null(Resolve("krea-2-raw").ExcludeRegex);
    }

    [Fact]
    public void AlphaDefaultsToRank()
    {
        Assert.Equal(16.0, Resolve(overrides: new() { ["rank"] = 16 }).EffectiveAlpha);
    }

    [Fact]
    public void VaeDefaultsToCpu() => Assert.Equal("cpu", Resolve().VaeDevice);

    [Fact]
    public void ResolutionMustBePatchAligned()
    {
        var error = Assert.Throws<ConfigException>(() => Resolve(overrides: new() { ["resolution"] = 1000 }));
        Assert.Contains("multiple of 16", error.Message);
    }

    [Fact]
    public void AModelWithADeeperVaeNeedsACoarserGrid()
    {
        // 1040 is a multiple of 16, which is all Krea 2 asks...
        Assert.Equal(1040, Resolve(overrides: new() { ["resolution"] = 1040 }).Resolution);
        // ...but LTX 2.3 patches at 32 pixels.
        var error = Assert.Throws<ConfigException>(() => Resolve("ltx-2.3", overrides: new() { ["resolution"] = 1040 }));
        Assert.Contains("multiple of 32", error.Message);
        var step = Assert.Throws<ConfigException>(() =>
            Resolve("ltx-2.3", project: Project("bucket_step: 48\n")));
        Assert.Contains("bucket_step", step.Message);
    }

    [Fact]
    public void SampleGpuMustDiffer()
    {
        var error = Assert.Throws<ConfigException>(() =>
            Resolve(overrides: new() { ["gpu"] = 0 }, project: Project("sample_gpu: 0\n")));
        Assert.Contains("must differ", error.Message);
    }

    [Fact]
    public void AnUnknownKeyIsAnErrorNotAWarning()
    {
        // A typo would otherwise train with the default it was meant to replace.
        var error = Assert.Throws<ConfigException>(() => Resolve(project: Project("learing_rate: 1e-5\n")));
        Assert.Contains("learing_rate", error.Message);
    }

    [Theory]
    [InlineData("optimizer: sgd\n", "optimizer")]
    [InlineData("dtype: fp16\n", "dtype")]
    [InlineData("rank: many\n", "rank")]
    [InlineData("fp8_base: perhaps\n", "fp8_base")]
    public void BadValuesNameTheirKey(string yaml, string key)
    {
        var error = Assert.Throws<ConfigException>(() => Resolve(project: Project(yaml)));
        Assert.Contains(key, error.Message);
    }

    [Fact]
    public void UnknownProfileAndModelAreRefused()
    {
        Assert.Throws<ConfigException>(() => Resolve(profile: "turbo"));
        Assert.Throws<ConfigException>(() => Resolve("sdxl"));
        Assert.Throws<ConfigException>(() => Resolve("z-image"));
        Assert.Throws<ConfigException>(() => Resolve("wan2.2-ti2v-5b"));
    }

    [Fact]
    public void ModelRootsAcceptOnePathOrSeveral()
    {
        var one = Resolve(project: Project("models_dir: /a\n"));
        Assert.Equal(["/a"], one.ModelsDir);

        var flag = Resolve(overrides: new() { ["models_dir"] = $"/a{Path.PathSeparator}/b" });
        Assert.Equal(["/a", "/b"], flag.ModelsDir);

        var list = Resolve(project: Project("models_dir:\n  - ~/models\n  - /c\n"));
        Assert.Equal([Path.Combine(PathText.Home, "models"), "/c"], list.ModelsDir);
    }

    [Fact]
    public void SourcesResolveThroughTheSameLayersWithoutAFullConfig()
    {
        var project = Project("models_dir:\n  - /shared\nrank: 8\n");
        var sources = ConfigResolver.ResolveSources(projectConfig: project);
        Assert.Equal(["/shared"], sources.Roots);

        var overridden = ConfigResolver.ResolveSources(
            new Dictionary<string, object?> { ["model_path"] = "~/one" }, project);
        Assert.Equal(Path.Combine(PathText.Home, "one"), overridden.ModelPath);
    }

    [Fact]
    public void TheResolvedConfigRoundTripsThroughItsOwnYaml()
    {
        var cfg = Resolve(overrides: new() { ["trigger_word"] = "sks", ["learning_rate"] = 5e-5 },
            project: Project("sample_prompts:\n  - \"a: cat\"\n"));
        var yaml = cfg.ToYaml();

        var keys = yaml.Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Where(l => !l.StartsWith(' ')).Select(l => l[..l.IndexOf(':')]).ToList();
        Assert.Equal(keys.Order(StringComparer.Ordinal), keys);

        // Loading it back as a project config must reproduce the same run.
        var again = ConfigResolver.Resolve("krea-2-raw", projectConfig: Project(yaml));
        Assert.Equal(cfg.ToJson().ToJsonString(), again.ToJson().ToJsonString());

        // And the Train form reads the same file.
        var request = TrainRequest.FromResolved(yaml);
        Assert.Equal("sks", request.TriggerWord);
        Assert.Equal(5e-5, request.LearningRate);
        Assert.Equal(["a: cat"], request.SamplePrompts);
    }

    [Fact]
    public void EveryKeyTheTrainFormWritesIsOneTheConfigAccepts()
    {
        // The form writes these into loralei.yaml; a key the binder rejects would fail every run.
        var doc = FlatYaml.Parse("");
        new TrainRequest
        {
            OutputRoot = "runs", Alpha = 16, LoraDropout = 0.1, TargetRegex = ".*", ExcludeRegex = "x",
            LrScheduler = "cosine", WarmupSteps = 1, GradientAccumulationSteps = 2, MaxGradNorm = 1,
            TimestepSampling = "uniform", FlowShift = 2, WeightingScheme = "mode", LogitMean = 0, LogitStd = 1,
            ModeScale = 1.2, Dtype = "fp32", GradientCheckpointing = false, SwapPinMemory = false,
            SwapPrefetch = 0, SwapH2dOnly = false, Bucketing = false, BucketStep = 32,
            MinBucketResolution = 128, MaxBucketResolution = 1024, CaptionExtension = ".caption",
            CaptionDropout = 0.1, Repeats = 2, NumWorkers = 0, SampleEvery = 100, SamplePrompts = ["p"],
            SampleSteps = 4, SampleCfg = 1, SampleGpu = 1, LogEvery = 5,
        }.ApplyYamlOnly(doc);

        var cfg = Resolve(project: Project(doc.ToString()));
        Assert.Equal("cosine", cfg.LrScheduler);
        Assert.Equal(".caption", cfg.CaptionExtension);
        Assert.False(cfg.Bucketing);
        Assert.Equal(1, cfg.SampleGpu);
    }
}
