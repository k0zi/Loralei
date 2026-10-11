using SkiaSharp;

namespace Loralei.ImageResizer.Tests;

/// <summary>Builds small on-disk images so the save pipeline can be tested for real.</summary>
internal static class ImageFixtures
{
    /// <summary>Writes an image with a distinct colour in each corner, so orientation is detectable.</summary>
    public static string Write(string folder, string name, int width, int height, SKEncodedImageFormat format)
    {
        using var surface = SKSurface.Create(new SKImageInfo(width, height));
        var canvas = surface.Canvas;
        canvas.Clear(new SKColor(0x20, 0x20, 0x28));

        var band = Math.Max(4, Math.Min(width, height) / 8);
        Fill(canvas, 0, 0, band, band, SKColors.Red);
        Fill(canvas, width - band, 0, band, band, SKColors.Lime);
        Fill(canvas, width - band, height - band, band, band, SKColors.Blue);
        Fill(canvas, 0, height - band, band, band, SKColors.Yellow);

        using var image = surface.Snapshot();
        using var data = image.Encode(format, 95);

        var path = Path.Combine(folder, name);
        using var stream = File.Create(path);
        data.SaveTo(stream);
        return path;
    }

    /// <summary>
    /// Rewrites a JPEG with an EXIF orientation tag by splicing an APP1 segment in after the SOI
    /// marker. Skia writes no EXIF of its own, and orientation handling is too easy to regress to
    /// leave untested.
    /// </summary>
    public static void SetJpegOrientation(string path, ushort orientation)
    {
        var original = File.ReadAllBytes(path);

        byte[] app1 =
        [
            0xFF, 0xE1,             // APP1 marker
            0x00, 0x22,             // segment length (34 = 2 + 6 + 26)
            0x45, 0x78, 0x69, 0x66, 0x00, 0x00, // "Exif\0\0"
            0x49, 0x49, 0x2A, 0x00, // TIFF header, little endian
            0x08, 0x00, 0x00, 0x00, // offset of IFD0
            0x01, 0x00,             // one entry
            0x12, 0x01,             // tag 0x0112 (Orientation)
            0x03, 0x00,             // type SHORT
            0x01, 0x00, 0x00, 0x00, // count
            (byte)orientation, 0x00, 0x00, 0x00, // value, padded to four bytes
            0x00, 0x00, 0x00, 0x00, // no next IFD
        ];

        using var stream = File.Create(path);
        stream.Write(original, 0, 2); // SOI
        stream.Write(app1, 0, app1.Length);
        stream.Write(original, 2, original.Length - 2);
    }

    private static void Fill(SKCanvas canvas, int x, int y, int w, int h, SKColor colour)
    {
        using var paint = new SKPaint { Color = colour };
        canvas.DrawRect(x, y, w, h, paint);
    }
}
