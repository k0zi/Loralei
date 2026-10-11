using System;
using Loralei.ImageResizer.Models;
using SkiaSharp;

namespace Loralei.ImageResizer.Services;

/// <summary>Renders a <see cref="SquareSelection"/> onto a square canvas and encodes it.</summary>
public static class ImageProcessor
{
    /// <summary>Mitchell is a good all-rounder: sharp enough downscaling without ringing, clean upscaling.</summary>
    private static readonly SKSamplingOptions HighQuality = new(SKCubicResampler.Mitchell);

    /// <summary>The blurred backdrop is drawn slightly larger than the canvas so the blur never samples past the edge.</summary>
    private const float BlurOverscan = 1.12f;

    /// <summary>Takes the edge off the backdrop so the foreground image stays dominant.</summary>
    private static readonly SKColor BlurScrim = new(0, 0, 0, 38);

    public static SKData Encode(SKBitmap source, SquareSelection selection, ResizeSettings settings, OutputFormat format)
    {
        using var image = Render(source, selection, settings);
        return image.Encode(format.Format, format.Quality);
    }

    /// <summary>Draws <paramref name="source"/> into a settings.TargetSize square, filling anything the selection does not cover.</summary>
    public static SKImage Render(SKBitmap source, SquareSelection selection, ResizeSettings settings)
    {
        var target = settings.TargetSize;
        var info = new SKImageInfo(target, target, SKColorType.Bgra8888, SKAlphaType.Premul);
        using var surface = SKSurface.Create(info);
        var canvas = surface.Canvas;

        using var skImage = SKImage.FromBitmap(source);
        var needsFill = ResizeGeometry.NeedsFill(selection, source.Width, source.Height);

        if (needsFill)
            DrawBackground(canvas, skImage, source.Width, source.Height, settings);
        else
            canvas.Clear(SKColors.Transparent);

        var dest = ResizeGeometry.ImageDestination(selection, source.Width, source.Height, target);
        canvas.DrawImage(skImage, ToRect(dest), HighQuality);

        return surface.Snapshot();
    }

    private static void DrawBackground(SKCanvas canvas, SKImage image, int width, int height, ResizeSettings settings)
    {
        switch (settings.PadFill)
        {
            case PadFill.White:
                canvas.Clear(SKColors.White);
                return;
            case PadFill.Black:
                canvas.Clear(SKColors.Black);
                return;
            case PadFill.Transparent:
                canvas.Clear(SKColors.Transparent);
                return;
        }

        var target = settings.TargetSize;
        canvas.Clear(SKColors.Black);

        // The image scaled to cover the canvas, blurred, and overscanned so the blur kernel
        // never reaches past the source and leaves washed-out borders.
        var cover = ResizeGeometry.ImageDestination(ResizeGeometry.Cover(width, height), width, height, target);
        var rect = ToRect(cover);
        var centre = target / 2f;
        rect = new SKRect(
            centre + (rect.Left - centre) * BlurOverscan,
            centre + (rect.Top - centre) * BlurOverscan,
            centre + (rect.Right - centre) * BlurOverscan,
            centre + (rect.Bottom - centre) * BlurOverscan);

        var sigma = target / 25f;
        using (var blur = SKImageFilter.CreateBlur(sigma, sigma, SKShaderTileMode.Clamp))
        using (var paint = new SKPaint { ImageFilter = blur })
        {
            canvas.DrawImage(image, rect, HighQuality, paint);
        }

        using var scrim = new SKPaint { Color = BlurScrim };
        canvas.DrawRect(0, 0, target, target, scrim);
    }

    public static SKRect ToRect(DestRect rect) => new(
        (float)rect.X,
        (float)rect.Y,
        (float)(rect.X + rect.Width),
        (float)(rect.Y + rect.Height));
}
