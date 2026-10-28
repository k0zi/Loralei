namespace Loralei.Core;

/// <summary>
/// The closed sets a configuration accepts, in one place, so the forms offer exactly what
/// <c>RunConfig.Validate</c> will take. Anything else is a hard error, not a warning.
/// </summary>
public static class LoraleiVocabulary
{
    public static readonly string[] Models = Loralei.Core.Models.ModelRegistry.All.Select(m => m.Key).ToArray();
    public static readonly string[] Profiles = Config.ConfigResolver.Profiles.Keys.ToArray();
    public static readonly string[] Optimizers = ["adamw8bit", "adamw", "adafactor"];
    public static readonly string[] TimestepSamplings = ["shift", "uniform", "logit_normal"];
    public static readonly string[] WeightingSchemes = ["none", "sigma_sqrt", "logit_normal", "mode"];
    public static readonly string[] LrSchedulers = ["constant_with_warmup", "constant", "cosine", "linear"];
    public static readonly string[] Dtypes = ["bf16", "fp32"];
    public static readonly string[] VaeDevices = ["cpu", "cuda"];
    public static readonly string[] LoraKinds = [Recommender.Custom, Recommender.Style, Recommender.Character];
    public static readonly string[] ExportFormats = ["both", "comfyui", "diffusers"];
}
