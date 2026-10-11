namespace Loralei.ImageResizer.Models;

/// <summary>How the square region to export is chosen for an image.</summary>
public enum ResizeMode
{
    /// <summary>Largest centred square fully inside the image. Crops the long edge away.</summary>
    Crop,

    /// <summary>Whole image scaled to fit, the remainder filled with <see cref="PadFill"/>.</summary>
    Pad,

    /// <summary>Square positioned by hand in the crop editor.</summary>
    Manual,
}
