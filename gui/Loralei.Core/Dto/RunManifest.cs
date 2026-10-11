using System.Text.Json;
using System.Text.Json.Serialization;

namespace Loralei.Core.Dto;

/// <summary>
/// <c>runs/&lt;name&gt;/manifest.json</c>, written by <c>src/loralei/run/layout.py:write_manifest</c>.
/// Extra keys are kept verbatim so the Runs page can show everything a run recorded,
/// including fields added by a newer loralei than this GUI knows about.
/// </summary>
public sealed class RunManifest
{
    [JsonPropertyName("state")] public string State { get; init; } = "unknown";
    [JsonPropertyName("step")] public int Step { get; init; }
    [JsonPropertyName("model")] public string? Model { get; init; }
    [JsonPropertyName("dataset")] public string? Dataset { get; init; }
    [JsonPropertyName("final")] public string? Final { get; init; }
    [JsonPropertyName("images")] public int? Images { get; init; }
    [JsonPropertyName("optimizer")] public string? Optimizer { get; init; }
    [JsonPropertyName("peak_vram_gib")] public double? PeakVramGib { get; init; }
    [JsonPropertyName("seconds")] public double? Seconds { get; init; }
    [JsonPropertyName("trainable_params")] public long? TrainableParams { get; init; }
    [JsonPropertyName("lora_targets")] public int? LoraTargets { get; init; }
    [JsonPropertyName("blocks_total")] public int? BlocksTotal { get; init; }
    [JsonPropertyName("blocks_swapped")] public int? BlocksSwapped { get; init; }
    [JsonPropertyName("updated_at")] public double? UpdatedAt { get; init; }

    [JsonExtensionData] public Dictionary<string, JsonElement> Extra { get; init; } = new();

    public bool IsRunning => State == "running";
    public bool IsDone => State == "done";
    public bool IsFailed => State == "failed";

    public DateTimeOffset? Updated =>
        UpdatedAt is null ? null : DateTimeOffset.FromUnixTimeMilliseconds((long)(UpdatedAt.Value * 1000));

    /// <summary>Every recorded key, flattened for display, sorted like `loralei runs show`.</summary>
    public IEnumerable<KeyValuePair<string, string>> AllFields()
    {
        var pairs = new Dictionary<string, string>();
        void Put(string k, object? v) { if (v is not null) pairs[k] = Convert.ToString(v, System.Globalization.CultureInfo.InvariantCulture) ?? ""; }
        Put("state", State); Put("step", Step); Put("model", Model); Put("dataset", Dataset);
        Put("final", Final); Put("images", Images); Put("optimizer", Optimizer);
        Put("peak_vram_gib", PeakVramGib); Put("seconds", Seconds);
        Put("trainable_params", TrainableParams); Put("lora_targets", LoraTargets);
        Put("blocks_total", BlocksTotal); Put("blocks_swapped", BlocksSwapped);
        Put("updated_at", UpdatedAt);
        foreach (var (k, v) in Extra) pairs[k] = v.ToString();
        return pairs.OrderBy(p => p.Key, StringComparer.Ordinal);
    }
}
