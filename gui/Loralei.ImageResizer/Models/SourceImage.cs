using System;
using System.IO;

namespace Loralei.ImageResizer.Models;

/// <summary>
/// An image file found in the chosen folder. <see cref="PixelWidth"/> and <see cref="PixelHeight"/>
/// are the dimensions <em>after</em> EXIF orientation is applied, so they match what the user sees
/// and what gets written out.
/// </summary>
public sealed record SourceImage(
    string Path,
    int PixelWidth,
    int PixelHeight,
    long FileSizeBytes)
{
    public string FileName => System.IO.Path.GetFileName(Path);

    public string Extension => System.IO.Path.GetExtension(Path);

    public string DimensionsText => $"{PixelWidth} × {PixelHeight}";

    public string FileSizeText => FormatBytes(FileSizeBytes);

    public static string FormatBytes(long bytes)
    {
        string[] units = ["B", "KB", "MB", "GB"];
        double value = bytes;
        var unit = 0;
        while (value >= 1024 && unit < units.Length - 1)
        {
            value /= 1024;
            unit++;
        }

        return unit == 0
            ? $"{bytes} B"
            : $"{value.ToString(value < 10 ? "0.0" : "0")} {units[unit]}";
    }
}
