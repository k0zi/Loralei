namespace Loralei.ImageResizer.Models;

/// <summary>What goes in the parts of the output canvas the source image does not cover.</summary>
public enum PadFill
{
    /// <summary>The image itself, scaled to cover the canvas and heavily blurred.</summary>
    Blur,
    White,
    Black,
    /// <summary>Alpha zero. Forces PNG output.</summary>
    Transparent,
}
