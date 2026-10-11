using System.Text.Json.Serialization;

namespace Loralei.Core.Dto;

/// <summary>
/// One line of <c>runs/&lt;name&gt;/logs/train.jsonl</c>. Metric lines carry
/// <see cref="Step"/>/<see cref="Loss"/>; lifecycle lines carry <see cref="Event"/>
/// (<c>start</c>, <c>save</c>, <c>done</c>).
/// </summary>
public sealed class TrainEvent
{
    [JsonPropertyName("t")] public double Timestamp { get; init; }
    [JsonPropertyName("event")] public string? Event { get; init; }
    [JsonPropertyName("step")] public int? Step { get; init; }
    [JsonPropertyName("epoch")] public int? Epoch { get; init; }
    [JsonPropertyName("loss")] public double? Loss { get; init; }
    [JsonPropertyName("vram_gib")] public double? VramGib { get; init; }
    [JsonPropertyName("max_steps")] public int? MaxSteps { get; init; }
    [JsonPropertyName("images")] public int? Images { get; init; }
    [JsonPropertyName("path")] public string? Path { get; init; }
    [JsonPropertyName("final")] public string? Final { get; init; }
    [JsonPropertyName("seconds")] public double? Seconds { get; init; }

    public bool IsMetric => Event is null && Step is not null;
}
