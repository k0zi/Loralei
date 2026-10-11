using System.Globalization;

namespace Loralei.Core;

/// <summary>
/// Everything the Train page can set. Null means "not set by the user" and is simply
/// omitted, so loralei's own layering (defaults &lt; preset &lt; profile &lt; user config &lt;
/// project config &lt; flags) still decides the value - exactly how the CLI treats its own
/// <c>None</c> defaults.
/// <para>
/// Fields are split by how they reach loralei. <c>train</c> has flags for only about a third
/// of <c>RunConfig</c>; the rest exist solely in <c>loralei.yaml</c>.
/// </para>
/// </summary>
public sealed class TrainRequest
{
    // --- required -------------------------------------------------------------
    public string Name { get; set; } = "";
    public string Dataset { get; set; } = "";

    // --- has a train flag -----------------------------------------------------
    public string? Model { get; set; }
    public string? Profile { get; set; }
    public int? Rank { get; set; }
    public int? Steps { get; set; }
    public int? Resolution { get; set; }
    public double? LearningRate { get; set; }
    public int? BatchSize { get; set; }
    public int? BlocksToSwap { get; set; }
    public bool? Fp8Base { get; set; }
    public string? Optimizer { get; set; }
    public string? TriggerWord { get; set; }
    public string? VaeDevice { get; set; }
    public List<string> ModelsDir { get; set; } = [];
    public string? ModelPath { get; set; }
    public int? Gpu { get; set; }
    public int? SaveEvery { get; set; }
    public int? Seed { get; set; }
    public bool Resume { get; set; }

    // --- loralei.yaml only ------------------------------------------------------
    public string? OutputRoot { get; set; }
    public double? Alpha { get; set; }
    public double? LoraDropout { get; set; }
    public string? TargetRegex { get; set; }
    public string? ExcludeRegex { get; set; }
    public string? LrScheduler { get; set; }
    public int? WarmupSteps { get; set; }
    public int? GradientAccumulationSteps { get; set; }
    public double? MaxGradNorm { get; set; }
    public string? TimestepSampling { get; set; }
    public double? FlowShift { get; set; }
    public string? WeightingScheme { get; set; }
    public double? LogitMean { get; set; }
    public double? LogitStd { get; set; }
    public double? ModeScale { get; set; }
    public string? Dtype { get; set; }
    public bool? GradientCheckpointing { get; set; }
    public bool? SwapPinMemory { get; set; }
    public int? SwapPrefetch { get; set; }
    public bool? SwapH2dOnly { get; set; }
    public bool? Bucketing { get; set; }
    public int? BucketStep { get; set; }
    public int? MinBucketResolution { get; set; }
    public int? MaxBucketResolution { get; set; }
    public string? CaptionExtension { get; set; }
    public double? CaptionDropout { get; set; }
    public int? Repeats { get; set; }
    public int? NumWorkers { get; set; }
    public int? SampleEvery { get; set; }
    public List<string> SamplePrompts { get; set; } = [];
    public int? SampleSteps { get; set; }
    public double? SampleCfg { get; set; }
    public int? SampleGpu { get; set; }
    public int? LogEvery { get; set; }

    /// <summary>
    /// The same checks as <c>RunConfig.Validate</c> plus the two required fields, run before
    /// spawning so a typo surfaces in the form rather than as a failed command.
    /// </summary>
    public IReadOnlyList<string> Validate()
    {
        var errors = new List<string>();
        if (string.IsNullOrWhiteSpace(Name)) errors.Add("Name is required.");
        if (string.IsNullOrWhiteSpace(Dataset)) errors.Add("Dataset is required.");
        else if (!Directory.Exists(Dataset)) errors.Add($"Dataset folder does not exist: {Dataset}");

        if (Resolution is { } res && res % 16 != 0)
            errors.Add($"Resolution must be a multiple of 16, got {res}.");
        if (Rank is { } rank && rank < 1) errors.Add("Rank must be >= 1.");
        if (CaptionDropout is { } drop && (drop < 0 || drop > 1))
            errors.Add("Caption dropout must be between 0 and 1.");
        if (Gpu is { } gpu && SampleGpu is { } sampleGpu && gpu == sampleGpu)
            errors.Add("Sample GPU must differ from the training GPU.");
        if (Steps is { } steps && steps < 1) errors.Add("Steps must be >= 1.");
        if (BatchSize is { } batch && batch < 1) errors.Add("Batch size must be >= 1.");

        foreach (var (value, label) in new[]
                 {
                     (Model, "Model"), (Profile, "Profile"), (Optimizer, "Optimizer"),
                     (VaeDevice, "VAE device"), (Dtype, "Dtype"),
                     (TimestepSampling, "Timestep sampling"), (WeightingScheme, "Weighting scheme"),
                 })
        {
            if (value is null) continue;
            var allowed = label switch
            {
                "Model" => LoraleiVocabulary.Models,
                "Profile" => LoraleiVocabulary.Profiles,
                "Optimizer" => LoraleiVocabulary.Optimizers,
                "VAE device" => LoraleiVocabulary.VaeDevices,
                "Dtype" => LoraleiVocabulary.Dtypes,
                "Timestep sampling" => LoraleiVocabulary.TimestepSamplings,
                _ => LoraleiVocabulary.WeightingSchemes,
            };
            if (!allowed.Contains(value)) errors.Add($"{label} must be one of: {string.Join(", ", allowed)}.");
        }
        return errors;
    }

    /// <summary>
    /// Applies the flagless settings to the workspace <c>loralei.yaml</c>, leaving every other
    /// line - comments, <c>models_dir</c>, anything a newer loralei added - untouched. A key the
    /// user cleared is removed rather than written as null, so the preset default comes back.
    /// </summary>
    public void ApplyYamlOnly(FlatYaml doc)
    {
        void Put(string key, object? value)
        {
            if (value is null) doc.Remove(key);
            else doc.Set(key, value);
        }

        Put("output_root", OutputRoot);
        Put("alpha", Alpha);
        Put("lora_dropout", LoraDropout);
        Put("target_regex", TargetRegex);
        Put("exclude_regex", ExcludeRegex);
        Put("lr_scheduler", LrScheduler);
        Put("warmup_steps", WarmupSteps);
        Put("gradient_accumulation_steps", GradientAccumulationSteps);
        Put("max_grad_norm", MaxGradNorm);
        Put("timestep_sampling", TimestepSampling);
        Put("flow_shift", FlowShift);
        Put("weighting_scheme", WeightingScheme);
        Put("logit_mean", LogitMean);
        Put("logit_std", LogitStd);
        Put("mode_scale", ModeScale);
        Put("dtype", Dtype);
        Put("gradient_checkpointing", GradientCheckpointing);
        Put("swap_pin_memory", SwapPinMemory);
        Put("swap_prefetch", SwapPrefetch);
        Put("swap_h2d_only", SwapH2dOnly);
        Put("bucketing", Bucketing);
        Put("bucket_step", BucketStep);
        Put("min_bucket_resolution", MinBucketResolution);
        Put("max_bucket_resolution", MaxBucketResolution);
        Put("caption_extension", CaptionExtension);
        Put("caption_dropout", CaptionDropout);
        Put("repeats", Repeats);
        Put("num_workers", NumWorkers);
        Put("sample_every", SampleEvery);
        Put("sample_steps", SampleSteps);
        Put("sample_cfg", SampleCfg);
        Put("sample_gpu", SampleGpu);
        Put("log_every", LogEvery);

        if (SamplePrompts.Count > 0) doc.Set("sample_prompts", SamplePrompts);
        else doc.Remove("sample_prompts");
    }

    /// <summary>Rebuilds a request from a run's <c>config.resolved.yaml</c>.</summary>
    public static TrainRequest FromResolved(string yaml)
    {
        var d = FlatYaml.Parse(yaml);
        string? S(string k) => d.Scalar(k) is { } v && v != "null" ? v : null;

        return new TrainRequest
        {
            Name = S("name") ?? "",
            Dataset = S("dataset") ?? "",
            Model = S("model"),
            Rank = d.Int("rank"),
            Steps = d.Int("max_steps"),
            Resolution = d.Int("resolution"),
            LearningRate = d.Double("learning_rate"),
            BatchSize = d.Int("batch_size"),
            BlocksToSwap = d.Int("blocks_to_swap"),
            Fp8Base = d.Bool("fp8_base"),
            Optimizer = S("optimizer"),
            TriggerWord = S("trigger_word"),
            VaeDevice = S("vae_device"),
            ModelsDir = d.Sequence("models_dir").ToList(),
            ModelPath = S("model_path"),
            Gpu = d.Int("gpu"),
            SaveEvery = d.Int("save_every"),
            Seed = d.Int("seed"),
            OutputRoot = S("output_root"),
            Alpha = d.Double("alpha"),
            LoraDropout = d.Double("lora_dropout"),
            TargetRegex = S("target_regex"),
            ExcludeRegex = S("exclude_regex"),
            LrScheduler = S("lr_scheduler"),
            WarmupSteps = d.Int("warmup_steps"),
            GradientAccumulationSteps = d.Int("gradient_accumulation_steps"),
            MaxGradNorm = d.Double("max_grad_norm"),
            TimestepSampling = S("timestep_sampling"),
            FlowShift = d.Double("flow_shift"),
            WeightingScheme = S("weighting_scheme"),
            LogitMean = d.Double("logit_mean"),
            LogitStd = d.Double("logit_std"),
            ModeScale = d.Double("mode_scale"),
            Dtype = S("dtype"),
            GradientCheckpointing = d.Bool("gradient_checkpointing"),
            SwapPinMemory = d.Bool("swap_pin_memory"),
            SwapPrefetch = d.Int("swap_prefetch"),
            SwapH2dOnly = d.Bool("swap_h2d_only"),
            Bucketing = d.Bool("bucketing"),
            BucketStep = d.Int("bucket_step"),
            MinBucketResolution = d.Int("min_bucket_resolution"),
            MaxBucketResolution = d.Int("max_bucket_resolution"),
            CaptionExtension = S("caption_extension"),
            CaptionDropout = d.Double("caption_dropout"),
            Repeats = d.Int("repeats"),
            NumWorkers = d.Int("num_workers"),
            SampleEvery = d.Int("sample_every"),
            SamplePrompts = d.Sequence("sample_prompts").ToList(),
            SampleSteps = d.Int("sample_steps"),
            SampleCfg = d.Double("sample_cfg"),
            SampleGpu = d.Int("sample_gpu"),
            LogEvery = d.Int("log_every"),
        };
    }

    internal static string Num(double value) => value.ToString("R", CultureInfo.InvariantCulture);
}
