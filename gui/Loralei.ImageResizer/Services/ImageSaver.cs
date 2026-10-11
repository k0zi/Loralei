using System;
using System.IO;
using Loralei.ImageResizer.Models;
using SkiaSharp;

namespace Loralei.ImageResizer.Services;

/// <summary>Writes resized copies into the output folder. Originals are only ever read.</summary>
public static class ImageSaver
{
    /// <summary>Give up inventing "_2", "_3", ... suffixes after this many attempts.</summary>
    private const int MaxNameAttempts = 9999;

    public static string EnsureOutputFolder(string sourceFolder)
    {
        var folder = Path.Combine(sourceFolder, ImageScanner.OutputFolderName);
        Directory.CreateDirectory(folder);
        return folder;
    }

    /// <summary>Renders one image and writes it. Returns the path written.</summary>
    public static string Save(SourceImage source, SquareSelection selection, ResizeSettings settings, string outputFolder)
    {
        using var bitmap = ImageDecoder.Decode(source.Path)
            ?? throw new InvalidOperationException("The file could not be decoded.");

        var needsFill = ResizeGeometry.NeedsFill(selection, bitmap.Width, bitmap.Height);
        var format = OutputFormatPolicy.Choose(source.Extension, settings.PadFill, settings.JpegQuality, needsFill);

        using var data = ImageProcessor.Encode(bitmap, selection, settings, format);
        return WriteWithoutOverwriting(
            outputFolder, Path.GetFileNameWithoutExtension(source.Path), format.Extension, data);
    }

    /// <summary>
    /// Writes to "stem.ext", or "stem_2.ext", "stem_3.ext", ... if taken. FileMode.CreateNew makes
    /// the check-and-create atomic, so parallel saves cannot clobber each other.
    /// </summary>
    private static string WriteWithoutOverwriting(string folder, string stem, string extension, SKData data)
    {
        for (var attempt = 1; attempt <= MaxNameAttempts; attempt++)
        {
            var name = attempt == 1 ? stem + extension : $"{stem}_{attempt}{extension}";
            var path = Path.Combine(folder, name);

            try
            {
                using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                data.SaveTo(stream);
                return path;
            }
            catch (IOException) when (File.Exists(path))
            {
                // Name taken - try the next suffix.
            }
        }

        throw new IOException($"Could not find a free file name for '{stem}{extension}'.");
    }
}
