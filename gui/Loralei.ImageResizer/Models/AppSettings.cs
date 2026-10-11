using System;

namespace Loralei.ImageResizer.Models;

/// <summary>Everything the app remembers between runs. Defaults match a first launch.</summary>
public sealed record AppSettings
{
    public int TargetSize { get; init; } = 1024;

    public ResizeMode DefaultMode { get; init; } = ResizeMode.Crop;

    public PadFill PadFill { get; init; } = PadFill.Blur;

    public int JpegQuality { get; init; } = 95;

    public string? LastFolder { get; init; }

    /// <summary>
    /// Brings hand-edited or outdated values back into range. A settings file is user-writable, so
    /// nothing in it is trusted to be sane.
    /// </summary>
    public AppSettings Sanitized() => this with
    {
        TargetSize = Array.IndexOf(ResizeSettings.TargetSizes, TargetSize) >= 0 ? TargetSize : 1024,
        DefaultMode = Enum.IsDefined(DefaultMode) ? DefaultMode : ResizeMode.Crop,
        PadFill = Enum.IsDefined(PadFill) ? PadFill : PadFill.Blur,
        JpegQuality = Math.Clamp(JpegQuality, 50, 100),
    };
}
