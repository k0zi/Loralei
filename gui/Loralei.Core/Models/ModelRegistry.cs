namespace Loralei.Core.Models;

/// <summary>
/// One diffusers subfolder (transformer, text_encoder, vae, ...).
/// <para>
/// <see cref="Repos"/> is a preference order. The first entry is where downloads come from; the
/// rest are repos known to contain a byte-identical copy, so an existing local cache can be
/// reused instead of re-downloading tens of gigabytes.
/// </para>
/// </summary>
/// <param name="Weights">False for tokenizer/scheduler - config-only folders are complete without safetensors.</param>
public sealed record ComponentSpec(string Name, IReadOnlyList<string> Repos, long ApproxBytes, bool Weights = true);

/// <summary>
/// What a supported base model is made of, and the architecture facts the trainer needs.
/// Training hyperparameters live in <c>Presets/*.yaml</c>; this is about assets (which repo,
/// how many bytes) and structure (which submodule holds the swappable blocks, which Linears
/// LoRA should target, which worker family drives it).
/// </summary>
/// <param name="Family">The worker's name for the architecture: the key into <c>loralei/families.py</c>.</param>
/// <param name="BucketDivisor">
/// VAE downscale times transformer patch size: every image side must be a multiple of this.
/// </param>
public sealed record ModelSpec(
    string Key,
    string Label,
    string Family,
    string PipelineClass,
    string TransformerClass,
    IReadOnlyList<ComponentSpec> Components,
    string BlocksAttr,
    string LoraTargetRegex,
    string? LoraExcludeRegex,
    int MaxSequenceLength,
    int BucketDivisor,
    IReadOnlyList<string> Notes)
{
    public long DownloadBytes => Components.Sum(c => c.ApproxBytes);

    public ComponentSpec Component(string name) =>
        Components.FirstOrDefault(c => c.Name == name)
        ?? throw new KeyNotFoundException($"{Key} has no component '{name}'");
}

public static class ModelRegistry
{
    private const long Gib = 1024L * 1024 * 1024;
    private const long Mib = 1024L * 1024;

    /// <summary>
    /// Said of every model here. The objectives are checked against each model's own diffusers
    /// pipeline on tiny random-weight copies (the worker's <c>tests/test_families.py</c>), which
    /// proves the calls are the right calls. It does not prove a LoRA comes out good.
    /// </summary>
    private const string Unproven =
        "Not yet proven on real weights: the training calls match the model's own pipeline, " +
        "but no full run has been judged by eye. Check early samples before trusting a long run.";

    private const string OneFrame = "A video model, trained here on images as one-frame clips.";

    // Sizes are the repo listings as of 2026-10; they only feed the free-space guard.
    private static readonly string[] QwenUnslothFirst = ["unsloth/Qwen-Image-2512", "Qwen/Qwen-Image-2512"];

    public static readonly ModelSpec QwenImage = new(
        Key: "qwen-image",
        Label: "Qwen-Image 2512 (20B)",
        Family: "qwen-image",
        PipelineClass: "QwenImagePipeline",
        TransformerClass: "QwenImageTransformer2DModel",
        Components:
        [
            new("transformer", ["Qwen/Qwen-Image-2512", "unsloth/Qwen-Image-2512"], 41 * Gib),
            new("text_encoder", QwenUnslothFirst, 16 * Gib),
            new("tokenizer", QwenUnslothFirst, 12 * Mib, Weights: false),
            new("vae", QwenUnslothFirst, 260 * Mib),
            new("scheduler", QwenUnslothFirst, 1024, Weights: false),
        ],
        BlocksAttr: "transformer_blocks",
        LoraTargetRegex: @"^transformer_blocks\.\d+\..*",
        // The adaLN modulation projections are tiny and numerically sensitive.
        LoraExcludeRegex: @".*(_mod|modulation|embedder|norm_out|proj_out).*",
        MaxSequenceLength: 1024,
        BucketDivisor: 16,
        Notes:
        [
            "60 MMDiT blocks, ~20B params. Needs fp8 weight storage + aggressive block swap to fit a 16 GB card.",
            Unproven,
        ]);

    // unsloth's mirror is ungated and file-for-file the same as Krea's own repo.
    private static readonly string[] KreaRepo = ["krea/Krea-2-Raw", "unsloth/Krea-2-Raw"];

    public static readonly ModelSpec Krea2Raw = new(
        Key: "krea-2-raw",
        Label: "Krea 2 Raw (12B)",
        Family: "krea2",
        PipelineClass: "Krea2Pipeline",
        TransformerClass: "Krea2Transformer2DModel",
        Components:
        [
            new("transformer", KreaRepo, (long)(24.5 * Gib)),
            new("text_encoder", KreaRepo, (long)(8.3 * Gib)),
            new("tokenizer", KreaRepo, 16 * Mib, Weights: false),
            new("vae", KreaRepo, 480 * Mib),
            new("scheduler", KreaRepo, 1024, Weights: false),
        ],
        BlocksAttr: "transformer_blocks",
        // Attention projections and the gated MLP's two matrices; the gates themselves are left alone.
        LoraTargetRegex: @"^transformer_blocks\.\d+\.(attn\.to_(q|k|v|out\.0)|ff\.(up|down))$",
        LoraExcludeRegex: null,
        MaxSequenceLength: 512,
        BucketDivisor: 16,
        Notes:
        [
            "28 blocks, ~12.8B params. The undistilled checkpoint Krea publishes for fine-tuning.",
            "Gated: accept the licence on huggingface.co/krea/Krea-2-Raw, then set HF_TOKEN before pulling.",
            Unproven,
        ]);

    private static readonly string[] LtxRepo = ["diffusers/LTX-2.3-Diffusers"];

    public static readonly ModelSpec Ltx23 = new(
        Key: "ltx-2.3",
        Label: "LTX 2.3 dev (19B, video + audio)",
        Family: "ltx2",
        PipelineClass: "LTX2Pipeline",
        TransformerClass: "LTX2VideoTransformer3DModel",
        Components:
        [
            new("transformer", LtxRepo, (long)(35.4 * Gib)),
            new("text_encoder", LtxRepo, (long)(45.4 * Gib)),
            new("tokenizer", LtxRepo, 40 * Mib, Weights: false),
            new("connectors", LtxRepo, (long)(11.9 * Gib)),
            new("vae", LtxRepo, (long)(1.4 * Gib)),
            new("audio_vae", LtxRepo, 110 * Mib),
            new("vocoder", LtxRepo, 250 * Mib),
            new("scheduler", LtxRepo, 1024, Weights: false),
        ],
        BlocksAttr: "transformer_blocks",
        // The video stream only: with no audio in the dataset, the audio half gets no gradient.
        LoraTargetRegex: @"^transformer_blocks\.\d+\.(attn1|attn2|ff)\..*",
        LoraExcludeRegex: @".*to_gate_logits.*",
        MaxSequenceLength: 1024,
        BucketDivisor: 32,
        Notes:
        [
            "48 blocks, ~19B params, in the diffusers layout (not Lightricks' single-file checkpoint).",
            "Its Gemma 3 12B text encoder is 45 GiB and is run on the CPU when captions are cached.",
            "Denoises video and audio together; the audio stream is fed noise and only the video is trained.",
            OneFrame,
            Unproven,
        ]);

    /// <summary>Every supported model, in listing order.</summary>
    public static readonly IReadOnlyList<ModelSpec> All =
        [QwenImage, Krea2Raw, Ltx23];

    /// <summary>What a command uses when <c>--model</c> is not given.</summary>
    public const string DefaultKey = "qwen-image";

    public static ModelSpec Get(string key) =>
        All.FirstOrDefault(m => m.Key == key)
        ?? throw new Config.ConfigException(
            $"unknown model '{key}'; available: {string.Join(", ", All.Select(m => m.Key))}");
}
