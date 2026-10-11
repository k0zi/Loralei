namespace Loralei.Core.Dto;

/// <summary>A run directory under the output root, as <c>runs ls</c> lists it.</summary>
public sealed record RunEntry(
    string Name,
    string Model,
    int Step,
    string State,
    DateTimeOffset? Updated,
    string Path);
