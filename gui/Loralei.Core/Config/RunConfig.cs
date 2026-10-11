using System.Text.Json.Nodes;

namespace Loralei.Core.Config;

/// <summary>Raised for anything wrong with a configuration: an unknown key, a bad value, a failed check.</summary>
public sealed class ConfigException(string message) : Exception(message);

/// <summary>
/// Every setting a run uses. Property names map to the snake_case keys of
/// <c>loralei.yaml</c> and <c>config.resolved.yaml</c>; <see cref="ConfigResolver"/> fills this
/// from the layers and <see cref="Validate"/> holds the cross-field rules.
/// </summary>
public sealed class RunConfig
{
    // --- identity -------------------------------------------------------------
    public string Name { get; set; } = "";
    public string Model { get; set; } = Models.ModelRegistry.DefaultKey;
    public string Dataset { get; set; } = "";
    public string OutputRoot { get; set; } = "runs";

    /// <summary>
    /// Roots to search for base weights, before the default Hugging Face cache. Each may be a
    /// Hugging Face cache directory or a plain <c>&lt;root&gt;/&lt;model key&gt;/&lt;component&gt;</c> tree.
    /// Also read from <c>$LORALEI_MODELS_DIR</c>. The first entry is where <c>models pull</c> writes.
    /// </summary>
    public List<string> ModelsDir { get; set; } = [];

    /// <summary>An explicit diffusers-format folder for this one model. Highest priority.</summary>
    public string? ModelPath { get; set; }

    // --- data -----------------------------------------------------------------
    public int Resolution { get; set; } = 1024;
    public bool Bucketing { get; set; } = true;
    public int BucketStep { get; set; } = 64;
    public int MinBucketResolution { get; set; } = 256;
    public int MaxBucketResolution { get; set; } = 2048;
    public string CaptionExtension { get; set; } = ".txt";

    /// <summary>
    /// Where the VAE runs when caching latents. CPU by default: on gfx1200 / ROCm 7.1 the VAE's
    /// convolutions return NaN for most passes at 1024-class shapes. See the README, "ROCm notes".
    /// </summary>
    public string VaeDevice { get; set; } = "cpu";
    public double CaptionDropout { get; set; }
    public string? TriggerWord { get; set; }
    public int Repeats { get; set; } = 1;

    // --- LoRA -----------------------------------------------------------------
    public int Rank { get; set; } = 32;

    /// <summary>Null means alpha == rank (scale 1.0).</summary>
    public double? Alpha { get; set; }
    public double LoraDropout { get; set; }
    public string? TargetRegex { get; set; }
    public string? ExcludeRegex { get; set; }

    // --- optimization ---------------------------------------------------------
    public string Optimizer { get; set; } = "adamw8bit";
    public double LearningRate { get; set; } = 1e-4;
    public string LrScheduler { get; set; } = "constant_with_warmup";
    public int WarmupSteps { get; set; } = 50;
    public int MaxSteps { get; set; } = 2000;
    public int BatchSize { get; set; } = 1;
    public int GradientAccumulationSteps { get; set; } = 1;
    public double MaxGradNorm { get; set; } = 1.0;
    public int Seed { get; set; } = 42;

    // --- objective ------------------------------------------------------------
    public string TimestepSampling { get; set; } = "shift";
    public double FlowShift { get; set; } = 3.0;
    public string WeightingScheme { get; set; } = "none";
    public double LogitMean { get; set; }
    public double LogitStd { get; set; } = 1.0;
    public double ModeScale { get; set; } = 1.29;

    // --- memory ---------------------------------------------------------------
    public string Dtype { get; set; } = "bf16";
    public bool GradientCheckpointing { get; set; } = true;
    public int BlocksToSwap { get; set; }
    public bool SwapPinMemory { get; set; } = true;

    /// <summary>Blocks to copy ahead on a side stream; 0 disables overlap.</summary>
    public int SwapPrefetch { get; set; } = 1;

    /// <summary>Frozen base weights never change, so copying them back to host is pure waste.</summary>
    public bool SwapH2dOnly { get; set; } = true;

    /// <summary>Store frozen base weights as float8_e4m3fn and dequantize per-block in forward.</summary>
    public bool Fp8Base { get; set; }

    // --- runtime --------------------------------------------------------------
    public int? Gpu { get; set; }
    public int? SampleGpu { get; set; }
    public int NumWorkers { get; set; } = 2;
    public int SaveEvery { get; set; } = 500;
    public int SampleEvery { get; set; }
    public List<string> SamplePrompts { get; set; } = [];
    public int SampleSteps { get; set; } = 20;
    public double SampleCfg { get; set; } = 4.0;
    public int LogEvery { get; set; } = 10;

    public double EffectiveAlpha => Alpha ?? Rank;

    public string RunDir => Path.Combine(OutputRoot, Name);

    /// <summary>The closed sets and cross-field rules. Throws on the first violation.</summary>
    public void Validate()
    {
        if (string.IsNullOrEmpty(Name)) throw new ConfigException("name is required");
        if (string.IsNullOrEmpty(Dataset)) throw new ConfigException("dataset is required");
        if (Resolution % 16 != 0)
            throw new ConfigException($"resolution must be a multiple of 16, got {Resolution}");
        if (Rank < 1) throw new ConfigException("rank must be >= 1");
        if (CaptionDropout is < 0 or > 1) throw new ConfigException("caption_dropout must be in [0, 1]");
        if (Gpu is not null && Gpu == SampleGpu)
            throw new ConfigException("sample_gpu must differ from gpu (or be left unset)");

        OneOf("optimizer", Optimizer, LoraleiVocabulary.Optimizers);
        OneOf("timestep_sampling", TimestepSampling, LoraleiVocabulary.TimestepSamplings);
        OneOf("weighting_scheme", WeightingScheme, LoraleiVocabulary.WeightingSchemes);
        OneOf("dtype", Dtype, LoraleiVocabulary.Dtypes);
        OneOf("vae_device", VaeDevice, LoraleiVocabulary.VaeDevices);
    }

    private static void OneOf(string key, string value, string[] allowed)
    {
        if (!allowed.Contains(value))
            throw new ConfigException($"{key} must be one of {string.Join(", ", allowed)}; got '{value}'");
    }

    /// <summary>
    /// The whole config as snake_case JSON. This one shape is what the worker receives, what
    /// <c>config.resolved.yaml</c> holds and what is stamped into the safetensors metadata.
    /// </summary>
    public JsonObject ToJson() => ConfigBinder.ToJson(this);

    /// <summary>
    /// <c>config.resolved.yaml</c>: every key, sorted, so a run can be reproduced from its own
    /// directory and two runs can be diffed line by line.
    /// </summary>
    public string ToYaml()
    {
        var doc = FlatYaml.Parse("");
        foreach (var (key, node) in ToJson().OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            switch (node)
            {
                case null: doc.Set(key, null); break;
                case JsonArray array: doc.Set(key, array.Select(n => n!.GetValue<string>()).ToList()); break;
                default:
                    var value = node.AsValue();
                    if (value.TryGetValue<bool>(out var b)) doc.Set(key, b);
                    else if (value.TryGetValue<int>(out var i)) doc.Set(key, i);
                    else if (value.TryGetValue<double>(out var d)) doc.Set(key, d);
                    else doc.Set(key, value.GetValue<string>());
                    break;
            }
        }
        // An empty document parses to one blank line; the file should start at its first key.
        return doc.ToString().TrimStart('\n');
    }

    /// <summary>Writes <c>config.resolved.yaml</c> into the run directory and returns its path.</summary>
    public string WriteResolved()
    {
        Directory.CreateDirectory(RunDir);
        var path = Path.Combine(RunDir, "config.resolved.yaml");
        File.WriteAllText(path, ToYaml());
        return path;
    }
}
