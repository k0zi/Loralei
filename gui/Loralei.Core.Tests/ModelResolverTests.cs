using Loralei.Core.Dto;
using Loralei.Core.Models;
using Xunit;

namespace Loralei.Core.Tests;

public class ModelResolverTests : IDisposable
{
    private readonly string _tmp = Directory.CreateTempSubdirectory("loralei-models").FullName;
    private static readonly ModelSpec Qwen = ModelRegistry.QwenImage;

    public void Dispose() => Directory.Delete(_tmp, recursive: true);

    private static string MakeComponent(string folder, string weights = "diffusion_pytorch_model.safetensors")
    {
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "config.json"), "{}");
        File.WriteAllText(Path.Combine(folder, weights), "not really weights, but the right shape of file");
        return folder;
    }

    private static string Populate(string @base)
    {
        foreach (var component in new[] { "transformer", "text_encoder", "vae" })
            MakeComponent(Path.Combine(@base, component));
        Directory.CreateDirectory(Path.Combine(@base, "tokenizer"));
        File.WriteAllText(Path.Combine(@base, "tokenizer", "tokenizer_config.json"), "{}");
        Directory.CreateDirectory(Path.Combine(@base, "scheduler"));
        File.WriteAllText(Path.Combine(@base, "scheduler", "scheduler_config.json"), "{}");
        return @base;
    }

    private string PlainModel(string root, string modelKey) => Populate(Path.Combine(_tmp, root, modelKey));

    private string HubModel(string root, string repo, string revision = "abc123") =>
        Populate(Path.Combine(_tmp, root, "models--" + repo.Replace("/", "--"), "snapshots", revision));

    private ModelSources Roots(params string[] roots) =>
        new(ModelsDir: roots.Select(r => Path.Combine(_tmp, r)).ToList());

    [Fact]
    public void PlainLayoutIsFound()
    {
        PlainModel("models", "qwen-image");
        var resolved = ModelResolver.Resolve(Qwen, Roots("models"));
        Assert.True(resolved.Complete);
        Assert.Equal(Path.Combine(_tmp, "models", "qwen-image", "transformer"), resolved.PathOf("transformer"));
    }

    [Fact]
    public void HuggingFaceCacheLayoutIsFound()
    {
        var snapshot = HubModel("hub", "unsloth/Qwen-Image-2512");
        var resolved = ModelResolver.Resolve(Qwen, Roots("hub"));
        Assert.True(resolved.Complete);
        Assert.Equal(Path.Combine(snapshot, "vae"), resolved.PathOf("vae"));
    }

    [Fact]
    public void HfHomeWithTheHubInsideIsFound()
    {
        HubModel(Path.Combine("hf", "hub"), "unsloth/Qwen-Image-2512");
        Assert.True(ModelResolver.Resolve(Qwen, Roots("hf")).Complete);
    }

    [Fact]
    public void AMirrorRepoCountsAsTheModel()
    {
        HubModel("hub", "unsloth/Krea-2-Raw");
        Assert.True(ModelResolver.Resolve(ModelRegistry.Krea2Raw, Roots("hub")).Complete);
    }

    [Fact]
    public void RootsAreCombinedComponentByComponent()
    {
        // One root has only the transformer, another has the rest - a real situation when
        // two tools each downloaded part of a model.
        MakeComponent(Path.Combine(_tmp, "a", "qwen-image", "transformer"));
        PlainModel("b", "qwen-image");

        var resolved = ModelResolver.Resolve(Qwen, Roots("a", "b"));
        Assert.True(resolved.Complete);
        Assert.StartsWith(Path.Combine(_tmp, "a"), resolved.PathOf("transformer"));
        Assert.StartsWith(Path.Combine(_tmp, "b"), resolved.PathOf("vae"));
    }

    [Fact]
    public void ModelPathWinsOverModelsDir()
    {
        PlainModel("shared", "qwen-image");
        var explicitPath = Path.Combine(_tmp, "explicit");
        MakeComponent(Path.Combine(explicitPath, "transformer"));

        var resolved = ModelResolver.Resolve(Qwen,
            new ModelSources(explicitPath, [Path.Combine(_tmp, "shared")]));
        Assert.Equal(Path.Combine(explicitPath, "transformer"), resolved.PathOf("transformer"));
        // Components the explicit folder lacks still resolve from the shared root.
        Assert.StartsWith(Path.Combine(_tmp, "shared"), resolved.PathOf("vae"));
    }

    [Fact]
    public void AnEmptyComponentFolderDoesNotCount()
    {
        var model = PlainModel("models", "qwen-image");
        foreach (var leftover in Directory.GetFiles(Path.Combine(model, "tokenizer"))) File.Delete(leftover);

        var resolved = ModelResolver.Resolve(Qwen, Roots("models"));
        Assert.False(resolved.Complete);
        Assert.Contains("empty", resolved.Components["tokenizer"].Reason);
    }

    [Fact]
    public void AWeightlessFolderIsReportedMissing()
    {
        var model = PlainModel("models", "qwen-image");
        foreach (var weights in Directory.GetFiles(Path.Combine(model, "transformer"), "*.safetensors"))
            File.Delete(weights);

        var resolved = ModelResolver.Resolve(Qwen, Roots("models"));
        Assert.False(resolved.Complete);
        Assert.Contains("only config files present", resolved.Components["transformer"].Reason);
        Assert.Throws<FileNotFoundException>(() => resolved.PathOf("transformer"));
        Assert.Equal(Qwen.Component("transformer").ApproxBytes, resolved.MissingBytes);
    }

    [Fact]
    public void AMissingShardOrADanglingBlobIsNotComplete()
    {
        var model = PlainModel("models", "qwen-image");
        var transformer = Path.Combine(model, "transformer");
        File.WriteAllText(Path.Combine(transformer, "diffusion_pytorch_model.safetensors.index.json"),
            """{"weight_map": {"a": "diffusion_pytorch_model.safetensors", "b": "part-00002.safetensors"}}""");
        Assert.Contains("shard(s) missing, e.g. part-00002.safetensors",
            ModelResolver.Resolve(Qwen, Roots("models")).Components["transformer"].Reason);

        // The cache's files are links into blobs/; a pruned blob leaves the link behind.
        var vae = Path.Combine(model, "vae", "diffusion_pytorch_model.safetensors");
        File.Delete(vae);
        File.CreateSymbolicLink(vae, Path.Combine(_tmp, "pruned-blob"));
        Assert.Contains("dangling blob", ModelResolver.Resolve(Qwen, Roots("models")).Components["vae"].Reason);
    }

    [Fact]
    public void EveryModelCanBeResolvedIntoARunConfiguration()
    {
        Assert.Contains(ModelRegistry.DefaultKey, LoraleiVocabulary.Models);
        Assert.Equal(["qwen-image", "krea-2-raw", "ltx-2.3"], LoraleiVocabulary.Models);
        Assert.All(ModelRegistry.All, spec =>
        {
            // A preset, a worker family, and a transformer among its own components.
            Assert.NotEmpty(Loralei.Core.Config.ConfigResolver.LoadPreset(spec.Key));
            Assert.Contains(spec.Family, new[] { "qwen-image", "krea2", "ltx2" });
            Assert.Contains("transformer", spec.Components.Select(c => c.Name));
            Assert.Contains(spec.BucketDivisor, new[] { 16, 32 });

            var cfg = Loralei.Core.Config.ConfigResolver.Resolve(spec.Key,
                overrides: new Dictionary<string, object?> { ["name"] = "n", ["dataset"] = _tmp },
                projectConfig: Path.Combine(_tmp, "absent.yaml"));
            Assert.Equal(spec.LoraTargetRegex, cfg.TargetRegex);
            Assert.Equal(0, cfg.Resolution % spec.BucketDivisor);
            // The target pattern has to be one .NET and Python both read the same way.
            _ = new System.Text.RegularExpressions.Regex(spec.LoraTargetRegex);
        });
    }

    [Fact]
    public void DownloadTargetIsTheFirstRoot()
    {
        Assert.Equal(Path.Combine(_tmp, "a", "qwen-image"), ModelResolver.DownloadTarget(Qwen, Roots("a", "b")));
        Assert.Null(ModelResolver.DownloadTarget(Qwen, new ModelSources()));
    }

    [Fact]
    public void TheListingSaysWhatIsReadyAndWhatItWouldTakeToFinish()
    {
        PlainModel("models", "qwen-image");
        var report = ModelsReport.Build(Roots("models"));

        Assert.Equal(LoraleiVocabulary.Models, report.Models.Select(m => m.Key));
        var qwen = report.Models.Single(m => m.Key == "qwen-image");
        Assert.Equal("ready", qwen.Status);
        Assert.Equal(0, qwen.MissingBytes);
        Assert.Single(qwen.FoundIn);
        Assert.True(report.FreeBytes > 0);

        // Krea 2 shares nothing with Qwen-Image on disk, so nothing here can satisfy it.
        var krea = report.Models.Single(m => m.Key == "krea-2-raw");
        Assert.NotEqual("ready", krea.Status);
        Assert.True(krea.MissingBytes > 0);
    }

    [Fact]
    public void DirSizeCountsALinkedTreeOnce()
    {
        var blobs = Directory.CreateDirectory(Path.Combine(_tmp, "repo", "blobs")).FullName;
        File.WriteAllBytes(Path.Combine(blobs, "one"), new byte[1000]);
        var snapshot = Directory.CreateDirectory(Path.Combine(_tmp, "repo", "snapshots", "rev")).FullName;
        File.CreateSymbolicLink(Path.Combine(snapshot, "weights"), Path.Combine(blobs, "one"));
        Directory.CreateSymbolicLink(Path.Combine(_tmp, "repo", "again"), blobs);

        Assert.Equal(1000, Disk.DirSize(Path.Combine(_tmp, "repo")));
        Assert.True(Disk.FreeBytes(Path.Combine(_tmp, "not", "there", "yet")) > 0);
        Assert.Throws<NotEnoughSpaceException>(() => Disk.EnsureSpace(_tmp, long.MaxValue / 2));
    }
}
