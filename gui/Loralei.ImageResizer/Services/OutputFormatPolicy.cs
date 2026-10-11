using System;
using SkiaSharp;

namespace Loralei.ImageResizer.Services;

/// <summary>The encoder chosen for one image, plus the file extension that goes with it.</summary>
public readonly record struct OutputFormat(SKEncodedImageFormat Format, int Quality, string Extension);

/// <summary>
/// Picks the output encoder. The rule is "keep the source format", with two exceptions:
/// a transparent fill forces PNG, and BMP/GIF become PNG because Skia cannot encode them.
/// </summary>
public static class OutputFormatPolicy
{
    /// <summary>Extensions the folder scanner accepts, lower-case, with the leading dot.</summary>
    public static readonly string[] SupportedExtensions =
        [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"];

    public static bool IsSupported(string extension) =>
        Array.Exists(SupportedExtensions, e => e.Equals(extension, StringComparison.OrdinalIgnoreCase));

    public static OutputFormat Choose(string sourceExtension, Models.PadFill fill, int jpegQuality, bool needsFill)
    {
        // Only an actually-used transparent fill matters; a pure crop covers the canvas either way.
        var transparent = needsFill && fill == Models.PadFill.Transparent;
        var ext = sourceExtension.ToLowerInvariant();

        if (transparent && ext is not ".webp")
            return new OutputFormat(SKEncodedImageFormat.Png, 100, ".png");

        return ext switch
        {
            ".jpg" or ".jpeg" => new OutputFormat(SKEncodedImageFormat.Jpeg, jpegQuality, ext),
            ".webp" => new OutputFormat(SKEncodedImageFormat.Webp, transparent ? 100 : jpegQuality, ".webp"),
            _ => new OutputFormat(SKEncodedImageFormat.Png, 100, ".png"),
        };
    }
}
