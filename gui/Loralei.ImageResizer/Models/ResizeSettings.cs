namespace Loralei.ImageResizer.Models;

/// <summary>The export settings shared by every image in the batch.</summary>
public sealed record ResizeSettings(
    int TargetSize,
    PadFill PadFill,
    int JpegQuality)
{
    /// <summary>The A values offered for the A x A output.</summary>
    public static readonly int[] TargetSizes = [512, 768, 1024, 1280, 2048];
}
