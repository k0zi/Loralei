namespace Loralei.ImageResizer.Models;

/// <summary>
/// A square region in source-pixel coordinates. It may extend outside the image bounds -
/// whatever falls outside is filled with the chosen <see cref="PadFill"/>, which is what lets
/// crop, pad and manual placement share a single render path.
/// </summary>
public readonly record struct SquareSelection(double X, double Y, double Size)
{
    public double Right => X + Size;
    public double Bottom => Y + Size;

    /// <summary>True when the square lies wholly within a <paramref name="width"/> x <paramref name="height"/> image.</summary>
    public bool IsInside(int width, int height) =>
        X >= 0 && Y >= 0 && Right <= width && Bottom <= height;

    public SquareSelection MovedTo(double x, double y) => this with { X = x, Y = y };
}
