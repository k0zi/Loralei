using System.Text.Json;
using System.Text.Json.Serialization;

namespace Loralei.Core.Dto;

/// <summary>What <c>loralei doctor --json</c> prints; built by <c>Diagnostics/Doctor.cs</c>.</summary>
public sealed class DoctorReport
{
    [JsonPropertyName("loralei_version")] public string LoraleiVersion { get; init; } = "";
    [JsonPropertyName("ok")] public bool Ok { get; init; }
    [JsonPropertyName("env_applied")] public Dictionary<string, string> EnvApplied { get; init; } = new();
    [JsonPropertyName("notes")] public List<string> Notes { get; init; } = new();
    [JsonPropertyName("checks")] public List<DoctorCheck> Checks { get; init; } = new();
}

public sealed class DoctorCheck
{
    [JsonPropertyName("name")] public string Name { get; init; } = "";

    /// <summary>One of <c>ok</c>, <c>warn</c>, <c>fail</c>, <c>skip</c>.</summary>
    [JsonPropertyName("status")] public string Status { get; init; } = "";
    [JsonPropertyName("detail")] public string Detail { get; init; } = "";
    [JsonPropertyName("remedy")] public string Remedy { get; init; } = "";

    /// <summary>Measurements behind the verdict, where a check has any.</summary>
    [JsonPropertyName("data")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public JsonElement? Data { get; init; }
}
