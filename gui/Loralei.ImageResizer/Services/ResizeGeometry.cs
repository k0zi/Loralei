using System;
using Loralei.ImageResizer.Models;

namespace Loralei.ImageResizer.Services;

/// <summary>Where the source image lands on the square output canvas, in output pixels.</summary>
public readonly record struct DestRect(double X, double Y, double Width, double Height);

/// <summary>
/// All of the resizing maths, as pure functions over plain numbers - no Skia, no Avalonia.
/// Crop, pad and manual placement all reduce to "map this <see cref="SquareSelection"/> onto the
/// whole A x A canvas", so there is exactly one code path to get right and it is unit-testable.
/// </summary>
public static class ResizeGeometry
{
    /// <summary>Smallest square the user may shrink a manual selection to.</summary>
    public const double MinSelectionSize = 16;

    /// <summary>Largest centred square that fits inside the image. Crops the long edge.</summary>
    public static SquareSelection Cover(int width, int height)
    {
        double side = Math.Min(width, height);
        return new SquareSelection((width - side) / 2, (height - side) / 2, side);
    }

    /// <summary>Smallest centred square that contains the whole image. Overflows the short axis.</summary>
    public static SquareSelection Contain(int width, int height)
    {
        double side = Math.Max(width, height);
        return new SquareSelection((width - side) / 2, (height - side) / 2, side);
    }

    /// <summary>The automatic selection for <paramref name="mode"/>. Manual has no automatic answer, so it falls back to <see cref="Cover"/>.</summary>
    public static SquareSelection ForMode(ResizeMode mode, int width, int height) => mode switch
    {
        ResizeMode.Pad => Contain(width, height),
        _ => Cover(width, height),
    };

    /// <summary>How much the source is scaled by: &gt; 1 means the output is being enlarged.</summary>
    public static double ScaleFactor(SquareSelection selection, int target) =>
        target / selection.Size;

    /// <summary>
    /// Where the full source image lands on the target canvas. The selection maps onto
    /// (0,0,target,target), so the image is placed relative to it.
    /// </summary>
    public static DestRect ImageDestination(SquareSelection selection, int width, int height, int target)
    {
        var scale = ScaleFactor(selection, target);
        return new DestRect(-selection.X * scale, -selection.Y * scale, width * scale, height * scale);
    }

    /// <summary>True when the selected region has fewer pixels than the output, i.e. detail is being invented.</summary>
    public static bool IsUpscaling(SquareSelection selection, int target) =>
        selection.Size < target;

    /// <summary>True when part of the canvas will not be covered by the source and needs the pad fill.</summary>
    public static bool NeedsFill(SquareSelection selection, int width, int height) =>
        !selection.IsInside(width, height);

    /// <summary>Re-sizes the selection while keeping the point (<paramref name="anchorX"/>, <paramref name="anchorY"/>) fixed. Used by wheel zoom.</summary>
    public static SquareSelection ResizeAbout(
        SquareSelection selection, double newSize, double anchorX, double anchorY, int width, int height)
    {
        newSize = ClampSize(newSize, width, height);
        var fx = (anchorX - selection.X) / selection.Size;
        var fy = (anchorY - selection.Y) / selection.Size;
        return new SquareSelection(anchorX - fx * newSize, anchorY - fy * newSize, newSize);
    }

    /// <summary>Keeps the selection's size but re-centres it on the image.</summary>
    public static SquareSelection Centered(SquareSelection selection, int width, int height) =>
        selection.MovedTo((width - selection.Size) / 2, (height - selection.Size) / 2);

    /// <summary>
    /// Keeps a manual selection usable: not smaller than <see cref="MinSelectionSize"/>, not
    /// absurdly larger than the image, and not dragged completely off it.
    /// </summary>
    public static SquareSelection Sanitize(SquareSelection selection, int width, int height)
    {
        var size = ClampSize(selection.Size, width, height);
        // Keep at least a quarter of the square overlapping the image so it can always be dragged back.
        var margin = size * 0.75;
        var x = Math.Clamp(selection.X, -margin, width - size + margin);
        var y = Math.Clamp(selection.Y, -margin, height - size + margin);
        return new SquareSelection(x, y, size);
    }

    public static double ClampSize(double size, int width, int height)
    {
        var max = Math.Max(width, height) * 2.0;
        return Math.Clamp(size, MinSelectionSize, Math.Max(MinSelectionSize, max));
    }
}
