using System.Text;
using Loralei.DatasetMaker.Captioning;

namespace Loralei.DatasetMaker.Processing;

public sealed record ProcessorSettings(
    string SystemPrompt,
    string? TriggerWord,
    bool Overwrite,
    int MaxParallel,
    int MaxRetries,
    bool DryRun);

public sealed record ProcessingSummary(int Total, int Succeeded, int Skipped, int Failed);

public enum ItemStatus
{
    /// <summary>Dry run only: the image would be captioned.</summary>
    Todo,
    Ok,
    Skipped,
    Failed,

    /// <summary>A transient failure is being retried; the image will report again.</summary>
    Retrying,

    /// <summary>A request for the image is about to be sent; the image will report again.</summary>
    Started,
}

/// <summary>
/// One line of progress. These took the place of the tool's console output, and
/// <see cref="ToString"/> is still that line.
/// </summary>
public sealed record ProcessingEvent(
    int Index,
    int Total,
    ItemStatus Status,
    string Image,
    string? Detail = null,
    string? Caption = null)
{
    public override string ToString() =>
        $"[{Index}/{Total}] {Label,-5} {Image}" + (Detail is null ? "" : $" - {Detail}");

    private string Label => Status switch
    {
        ItemStatus.Todo => "TODO",
        ItemStatus.Ok => "OK",
        ItemStatus.Skipped => "SKIP",
        ItemStatus.Failed => "FAIL",
        ItemStatus.Started => "START",
        _ => "RETRY",
    };
}

/// <param name="report">Called for every event, from whichever thread produced it.</param>
public sealed class DatasetProcessor(
    OpenAiCaptionClient client, ProcessorSettings settings, Action<ProcessingEvent> report)
{
    private static readonly UTF8Encoding Utf8NoBom = new(encoderShouldEmitUTF8Identifier: false);

    public async Task<ProcessingSummary> RunAsync(string directory, CancellationToken ct)
    {
        var images = ImageScanner.FindImages(directory);
        var total = images.Count;

        int succeeded = 0, skipped = 0, failed = 0, index = 0;
        var claimedTargets = new HashSet<string>(StringComparer.Ordinal);
        var work = new List<(int Index, string Image, string Target)>();

        foreach (var image in images)
        {
            index++;
            var target = Path.ChangeExtension(image, ".txt");
            if (!claimedTargets.Add(target))
            {
                report(new(index, total, ItemStatus.Skipped, image,
                    $"another image already maps to {Path.GetFileName(target)}"));
                skipped++;
            }
            else if (!settings.Overwrite && File.Exists(target))
            {
                report(new(index, total, ItemStatus.Skipped, image, "caption exists"));
                skipped++;
            }
            else if (settings.DryRun)
            {
                report(new(index, total, ItemStatus.Todo, image));
            }
            else
            {
                work.Add((index, image, target));
            }
        }

        if (settings.DryRun)
            return new ProcessingSummary(total, 0, skipped, 0);

        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = Math.Max(1, settings.MaxParallel),
            CancellationToken = ct,
        };

        await Parallel.ForEachAsync(work, parallelOptions, async (item, token) =>
        {
            try
            {
                report(new(item.Index, total, ItemStatus.Started, item.Image));
                var raw = await CaptionWithRetryAsync(item.Index, total, item.Image, token);
                var caption = PromptBuilder.PostProcess(raw, settings.TriggerWord);
                await WriteAtomicAsync(item.Target, caption, token);
                Interlocked.Increment(ref succeeded);
                report(new(item.Index, total, ItemStatus.Ok, item.Image, Caption: caption));
            }
            catch (OperationCanceledException) when (token.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex)
            {
                Interlocked.Increment(ref failed);
                report(new(item.Index, total, ItemStatus.Failed, item.Image, ex.Message));
            }
        });

        return new ProcessingSummary(total, succeeded, skipped, failed);
    }

    private async Task<string> CaptionWithRetryAsync(int index, int total, string image, CancellationToken ct)
    {
        for (var attempt = 0; ; attempt++)
        {
            try
            {
                return await client.CaptionAsync(image, settings.SystemPrompt, ct);
            }
            catch (CaptionRequestException ex) when (ex.IsTransient && attempt < settings.MaxRetries)
            {
                var delay = TimeSpan.FromSeconds(Math.Pow(2, attempt + 1));
                report(new(index, total, ItemStatus.Retrying, image,
                    $"retry {attempt + 1}/{settings.MaxRetries} in {delay.TotalSeconds:0}s ({ex.Message})"));
                await Task.Delay(delay, ct);
            }
        }
    }

    private static async Task WriteAtomicAsync(string target, string content, CancellationToken ct)
    {
        var temp = target + ".tmp";
        await File.WriteAllTextAsync(temp, content, Utf8NoBom, ct);
        File.Move(temp, target, overwrite: true);
    }
}
