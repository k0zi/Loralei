using Loralei.Core.Models;

namespace Loralei.Core.Config;

/// <summary>
/// Layered configuration. Precedence, lowest to highest:
/// <code>
/// built-in defaults &lt; $LORALEI_MODELS_DIR &lt; Presets/&lt;model&gt;.yaml &lt; profile
///                   &lt; ~/.config/loralei/config.yaml &lt; ./loralei.yaml &lt; command-line flags
/// </code>
/// The fully resolved config is written to <c>runs/&lt;name&gt;/config.resolved.yaml</c> and stamped
/// into the output safetensors metadata, so a run can always be reproduced from its own artifacts.
/// </summary>
public static class ConfigResolver
{
    /// <summary>Keys that also apply to commands which never build a full <see cref="RunConfig"/>.</summary>
    private static readonly string[] SourceKeys = ["models_dir", "model_path"];

    public const string ProjectConfigName = "loralei.yaml";

    public static string UserConfig => Path.Combine(PathText.Home, ".config", "loralei", "config.yaml");

    /// <summary>Profiles are named overlays applied after the preset, before the user config.</summary>
    public static readonly IReadOnlyDictionary<string, IReadOnlyDictionary<string, object?>> Profiles =
        new Dictionary<string, IReadOnlyDictionary<string, object?>>
        {
            ["balanced"] = new Dictionary<string, object?>(),
            ["quality"] = new Dictionary<string, object?> { ["rank"] = 64, ["fp8_base"] = false, ["blocks_to_swap"] = 20 },
            ["speed"] = new Dictionary<string, object?> { ["fp8_base"] = true, ["blocks_to_swap"] = 0, ["gradient_checkpointing"] = true },
            ["lowmem"] = new Dictionary<string, object?> { ["fp8_base"] = true, ["blocks_to_swap"] = 24, ["batch_size"] = 1 },
        };

    /// <summary>A YAML file as one config layer; a missing file is simply an empty layer.</summary>
    public static Dictionary<string, object?> LoadYaml(string path)
    {
        if (!File.Exists(path)) return [];
        return FromYaml(File.ReadAllText(path));
    }

    private static Dictionary<string, object?> FromYaml(string text)
    {
        var doc = FlatYaml.Parse(text);
        var layer = new Dictionary<string, object?>();
        foreach (var key in doc.Keys)
        {
            if (doc.IsSequence(key)) layer[key] = doc.Sequence(key).ToList();
            else layer[key] = doc.Scalar(key) is { } scalar && scalar is not ("null" or "~") ? scalar : null;
        }
        return layer;
    }

    public static Dictionary<string, object?> LoadPreset(string modelKey)
    {
        var assembly = typeof(ConfigResolver).Assembly;
        using var stream = assembly.GetManifestResourceStream($"Loralei.Core.Presets.{modelKey}.yaml");
        if (stream is null) return [];
        using var reader = new StreamReader(stream);
        return FromYaml(reader.ReadToEnd());
    }

    private static Dictionary<string, object?> SourceDefaults() =>
        Environment.GetEnvironmentVariable("LORALEI_MODELS_DIR") is { Length: > 0 } dir
            ? new Dictionary<string, object?> { ["models_dir"] = dir }
            : [];

    private static void Merge(Dictionary<string, object?> into, IReadOnlyDictionary<string, object?> layer,
        bool skipNulls = false, Func<string, bool>? only = null)
    {
        foreach (var (key, value) in layer)
        {
            if (skipNulls && value is null) continue;
            if (only is not null && !only(key)) continue;
            into[key] = value;
        }
    }

    /// <summary>
    /// Model locations for commands that do not build a full config - <c>models ls</c>,
    /// <c>pull</c>, <c>sample</c> and <c>export</c> - resolved through the same layers.
    /// </summary>
    public static ModelSources ResolveSources(IReadOnlyDictionary<string, object?>? overrides = null,
        string? projectConfig = null)
    {
        var merged = SourceDefaults();
        Merge(merged, LoadYaml(UserConfig), only: SourceKeys.Contains);
        Merge(merged, LoadYaml(projectConfig ?? ProjectConfigName), only: SourceKeys.Contains);
        Merge(merged, overrides ?? new Dictionary<string, object?>(), skipNulls: true);

        var cfg = new RunConfig();
        ConfigBinder.Apply(cfg, merged);
        return new ModelSources(cfg.ModelPath, PathText.AsRoots(cfg.ModelsDir));
    }

    /// <summary>Merges every layer and validates.</summary>
    /// <param name="overrides">Command-line flags; a null value means "not passed" and is ignored.</param>
    public static RunConfig Resolve(string modelKey, string profile = "balanced",
        IReadOnlyDictionary<string, object?>? overrides = null, string? projectConfig = null)
    {
        if (!Profiles.TryGetValue(profile, out var overlay))
            throw new ConfigException($"unknown profile '{profile}'; available: {string.Join(", ", Profiles.Keys)}");

        var merged = new Dictionary<string, object?> { ["model"] = modelKey };
        Merge(merged, SourceDefaults());
        Merge(merged, LoadPreset(modelKey));
        Merge(merged, overlay);
        Merge(merged, LoadYaml(UserConfig));
        Merge(merged, LoadYaml(projectConfig ?? ProjectConfigName));
        Merge(merged, overrides ?? new Dictionary<string, object?>(), skipNulls: true);

        // Architecture defaults come from the registry unless explicitly overridden.
        var spec = ModelRegistry.Get(merged["model"]?.ToString() ?? modelKey);
        merged.TryAdd("target_regex", spec.LoraTargetRegex);
        merged.TryAdd("exclude_regex", spec.LoraExcludeRegex);

        var cfg = new RunConfig();
        ConfigBinder.Apply(cfg, merged);
        cfg.ModelsDir = PathText.AsRoots(cfg.ModelsDir);
        cfg.Validate();

        // Image sides must divide into this model's latent patches - 16 px for some, 32 for others.
        var divisor = spec.BucketDivisor;
        if (cfg.Resolution % divisor != 0)
            throw new ConfigException($"resolution must be a multiple of {divisor} for {spec.Key}, got {cfg.Resolution}");
        if (cfg.Bucketing && (cfg.BucketStep <= 0 || cfg.BucketStep % divisor != 0))
            throw new ConfigException($"bucket_step must be a multiple of {divisor} for {spec.Key}, got {cfg.BucketStep}");
        return cfg;
    }
}
