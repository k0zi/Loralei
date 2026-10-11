using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Loralei.ImageResizer.Models;

namespace Loralei.ImageResizer.Services;

/// <summary>Finds the images directly inside a folder. Sub-folders are deliberately ignored.</summary>
public static class ImageScanner
{
    public const string OutputFolderName = "saved_resized";

    public static IReadOnlyList<SourceImage> Scan(string folder, CancellationToken token = default)
    {
        var files = Directory
            .EnumerateFiles(folder, "*", SearchOption.TopDirectoryOnly)
            .Where(f => OutputFormatPolicy.IsSupported(Path.GetExtension(f)))
            .OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase);

        var results = new List<SourceImage>();
        foreach (var file in files)
        {
            token.ThrowIfCancellationRequested();

            // A file Skia cannot open is not an image we can resize, so it never reaches the grid.
            if (!ImageDecoder.TryReadInfo(file, out var width, out var height))
                continue;

            results.Add(new SourceImage(file, width, height, new FileInfo(file).Length));
        }

        return results;
    }
}
