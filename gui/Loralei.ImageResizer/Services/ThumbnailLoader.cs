using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Media.Imaging;
using Avalonia.Threading;

namespace Loralei.ImageResizer.Services;

/// <summary>Decodes grid thumbnails off the UI thread, a few at a time.</summary>
public static class ThumbnailLoader
{
    /// <summary>Long edge of a grid thumbnail, in pixels. Keeps memory sane for big folders.</summary>
    public const int ThumbnailSize = 220;

    /// <summary>Long edge of the crop-editor preview - big enough to judge framing, small enough to decode fast.</summary>
    public const int PreviewSize = 1600;

    /// <summary>
    /// Decodes a thumbnail for each path and hands it back on the UI thread via
    /// <paramref name="onLoaded"/>. Concurrency is capped so a huge folder cannot swamp the machine.
    /// </summary>
    public static async Task LoadAllAsync<T>(
        IReadOnlyList<T> items,
        Func<T, string> pathOf,
        Action<T, Bitmap> onLoaded,
        CancellationToken token)
    {
        using var limit = new SemaphoreSlim(Math.Max(2, Environment.ProcessorCount / 2));

        var tasks = new List<Task>(items.Count);
        foreach (var item in items)
        {
            if (token.IsCancellationRequested)
                break;

            await limit.WaitAsync(token).ConfigureAwait(false);
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    var bitmap = LoadOne(pathOf(item), ThumbnailSize);
                    if (bitmap is null || token.IsCancellationRequested)
                    {
                        bitmap?.Dispose();
                        return;
                    }

                    Dispatcher.UIThread.Post(() =>
                    {
                        if (token.IsCancellationRequested)
                            bitmap.Dispose();
                        else
                            onLoaded(item, bitmap);
                    });
                }
                finally
                {
                    limit.Release();
                }
            }, token));
        }

        try
        {
            await Task.WhenAll(tasks).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            // Folder changed mid-load; the new scan takes over.
        }
    }

    /// <summary>Decodes a single upright bitmap at most <paramref name="maxLongEdge"/> px on its long side.</summary>
    public static Bitmap? LoadOne(string path, int maxLongEdge)
    {
        try
        {
            using var stream = ImageDecoder.DecodeToPngStream(path, maxLongEdge);
            return stream is null ? null : new Bitmap(stream);
        }
        catch (Exception)
        {
            return null;
        }
    }
}
