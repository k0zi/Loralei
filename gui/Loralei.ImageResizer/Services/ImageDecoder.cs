using System;
using System.IO;
using SkiaSharp;

namespace Loralei.ImageResizer.Services;

/// <summary>
/// Decoding that respects EXIF orientation. Skia hands back the raw encoded pixels and leaves the
/// orientation to the caller, so without this a photo shot in portrait comes out lying on its side.
/// </summary>
public static class ImageDecoder
{
    /// <summary>Reads the display dimensions without decoding pixels. Width/height are already swapped for sideways EXIF origins.</summary>
    public static bool TryReadInfo(string path, out int width, out int height)
    {
        width = height = 0;
        try
        {
            using var codec = SKCodec.Create(path);
            if (codec is null)
                return false;

            var info = codec.Info;
            (width, height) = SwapsAxes(codec.EncodedOrigin)
                ? (info.Height, info.Width)
                : (info.Width, info.Height);
            return width > 0 && height > 0;
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>
    /// Decodes <paramref name="path"/> upright. When <paramref name="maxLongEdge"/> is positive the
    /// codec is asked for a reduced-size decode, which is far cheaper than decoding full resolution
    /// and shrinking afterwards - this is what makes thumbnailing a big folder bearable.
    /// </summary>
    public static SKBitmap? Decode(string path, int maxLongEdge = 0)
    {
        using var codec = SKCodec.Create(path);
        if (codec is null)
            return null;

        var info = codec.Info;
        var targetInfo = new SKImageInfo(info.Width, info.Height, SKColorType.Bgra8888, SKAlphaType.Premul);

        if (maxLongEdge > 0)
        {
            var longEdge = Math.Max(info.Width, info.Height);
            if (longEdge > maxLongEdge)
            {
                var dims = codec.GetScaledDimensions((float)maxLongEdge / longEdge);
                if (dims.Width > 0 && dims.Height > 0)
                    targetInfo = new SKImageInfo(dims.Width, dims.Height, SKColorType.Bgra8888, SKAlphaType.Premul);
            }
        }

        var decoded = new SKBitmap(targetInfo);
        var result = codec.GetPixels(targetInfo, decoded.GetPixels());
        if (result is not (SKCodecResult.Success or SKCodecResult.IncompleteInput))
        {
            decoded.Dispose();
            return null;
        }

        decoded.NotifyPixelsChanged();
        return Orient(decoded, codec.EncodedOrigin);
    }

    /// <summary>Decodes to a PNG byte stream at thumbnail size, ready to hand to Avalonia's <c>Bitmap</c>.</summary>
    public static MemoryStream? DecodeToPngStream(string path, int maxLongEdge)
    {
        using var bitmap = Decode(path, maxLongEdge);
        if (bitmap is null)
            return null;

        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        return new MemoryStream(data.ToArray(), writable: false);
    }

    /// <summary>EXIF origins 5-8 mean the stored image is rotated a quarter turn, so the display axes are swapped.</summary>
    public static bool SwapsAxes(SKEncodedOrigin origin) => origin is
        SKEncodedOrigin.LeftTop or SKEncodedOrigin.RightTop or
        SKEncodedOrigin.RightBottom or SKEncodedOrigin.LeftBottom;

    /// <summary>
    /// Re-draws <paramref name="source"/> upright and disposes it. The canvas transforms below map
    /// encoded pixel (x, y) to its display position for each of the eight EXIF origins.
    /// </summary>
    private static SKBitmap Orient(SKBitmap source, SKEncodedOrigin origin)
    {
        if (origin is SKEncodedOrigin.TopLeft or SKEncodedOrigin.Default)
            return source;

        var swap = SwapsAxes(origin);
        int width = swap ? source.Height : source.Width;
        int height = swap ? source.Width : source.Height;

        var target = new SKBitmap(new SKImageInfo(width, height, source.ColorType, source.AlphaType));
        using (var canvas = new SKCanvas(target))
        {
            switch (origin)
            {
                case SKEncodedOrigin.TopRight: // mirrored horizontally
                    canvas.Translate(width, 0);
                    canvas.Scale(-1, 1);
                    break;
                case SKEncodedOrigin.BottomRight: // rotated 180
                    canvas.Translate(width, height);
                    canvas.Scale(-1, -1);
                    break;
                case SKEncodedOrigin.BottomLeft: // mirrored vertically
                    canvas.Translate(0, height);
                    canvas.Scale(1, -1);
                    break;
                case SKEncodedOrigin.LeftTop: // transposed
                    canvas.RotateDegrees(90);
                    canvas.Scale(1, -1);
                    break;
                case SKEncodedOrigin.RightTop: // rotated 90 clockwise
                    canvas.Translate(width, 0);
                    canvas.RotateDegrees(90);
                    break;
                case SKEncodedOrigin.RightBottom: // transverse
                    canvas.Translate(width, height);
                    canvas.Scale(-1, -1);
                    canvas.RotateDegrees(90);
                    canvas.Scale(1, -1);
                    break;
                case SKEncodedOrigin.LeftBottom: // rotated 90 counter-clockwise
                    canvas.Translate(0, height);
                    canvas.RotateDegrees(-90);
                    break;
            }

            canvas.DrawBitmap(source, 0, 0);
        }

        source.Dispose();
        return target;
    }
}
