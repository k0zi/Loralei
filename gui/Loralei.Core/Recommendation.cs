using System.Globalization;
using Loralei.Core.Config;

namespace Loralei.Core;

/// <summary>
/// The settings the Train page fills in for a kind of LoRA. Null means "no opinion", which
/// leaves the field to the preset and profile like any other empty field.
/// </summary>
public sealed record Recommendation(
    int? Rank = null,
    double? LearningRate = null,
    int? Epochs = null,
    int? Steps = null,
    int? BatchSize = null,
    int? Repeats = null,
    int? WarmupSteps = null,
    int? SaveEvery = null,
    double? CaptionDropout = null,
    string Summary = "")
{
    public static readonly Recommendation None = new();
}

/// <summary>
/// Suggests starting values from what is being trained (a style or a character), on which
/// model, and from how many images.
/// <para>
/// These are rules of thumb, not measurements - like the presets they sit on top of. The
/// length of the run is what the image count decides: each image should be seen roughly the
/// same number of times whatever the size of the set, within a floor (a tiny set still needs
/// enough steps to converge) and a ceiling (a large one does not need them all).
/// </para>
/// </summary>
public static class Recommender
{
    public const string Custom = "custom";
    public const string Style = "style";
    public const string Character = "character";

    /// <param name="ViewsPerImage">How often each image is shown over the whole run.</param>
    private sealed record Shape(int Rank, int ViewsPerImage, int MinSteps, int MaxSteps, double? CaptionDropout);

    /// <summary>
    /// A character is one subject: a small adapter learns it, and it overfits quickly. A style
    /// spans the whole image, so it takes more capacity, a longer run at either end of the
    /// range, and some caption dropout so the look also holds when the prompt does not describe it.
    /// </summary>
    private static readonly Dictionary<string, Shape> Shapes = new()
    {
        [Character] = new(Rank: 16, ViewsPerImage: 100, MinSteps: 1000, MaxSteps: 3000, CaptionDropout: null),
        [Style] = new(Rank: 32, ViewsPerImage: 80, MinSteps: 1500, MaxSteps: 4000, CaptionDropout: 0.1),
    };

    /// <summary>How many checkpoints a recommended run leaves behind to choose between.</summary>
    private const int Checkpoints = 8;

    /// <summary>
    /// Steps in one pass over the dataset. The trainer counts every batch as a step and keeps
    /// the short last batch, so this is exact for one bucket and a lower bound for several.
    /// </summary>
    public static int StepsPerEpoch(int images, int repeats = 1, int batchSize = 1)
    {
        if (images <= 0) return 0;
        var samples = (long)images * Math.Max(1, repeats);
        var batch = Math.Max(1, batchSize);
        return (int)Math.Min(int.MaxValue, (samples + batch - 1) / batch);
    }

    public static int StepsForEpochs(int epochs, int images, int repeats = 1, int batchSize = 1) =>
        (int)Math.Min(int.MaxValue, (long)Math.Max(0, epochs) * StepsPerEpoch(images, repeats, batchSize));

    /// <summary>The whole epochs needed to cover <paramref name="steps"/>.</summary>
    public static int EpochsForSteps(int steps, int images, int repeats = 1, int batchSize = 1)
    {
        var perEpoch = StepsPerEpoch(images, repeats, batchSize);
        return perEpoch == 0 || steps <= 0 ? 0 : (int)(((long)steps + perEpoch - 1) / perEpoch);
    }

    public static Recommendation For(string model, string kind, int images)
    {
        if (!Shapes.TryGetValue(kind, out var shape)) return Recommendation.None;

        var learningRate = PresetLearningRate(model);
        var rec = new Recommendation(
            Rank: shape.Rank,
            LearningRate: learningRate,
            BatchSize: 1,
            Repeats: 1,
            CaptionDropout: shape.CaptionDropout);

        if (images <= 0)
            return rec with { Summary = $"{kind}: rank {shape.Rank}. Pick a dataset folder to size the run." };

        // Batch 1 and no repeats, so an epoch is one step per image.
        var target = Math.Clamp((long)images * shape.ViewsPerImage, shape.MinSteps, shape.MaxSteps);
        var epochs = (int)Math.Max(1, Math.Round((double)target / images, MidpointRounding.AwayFromZero));
        var steps = StepsForEpochs(epochs, images);
        var saveEvery = Math.Max(1, (int)Math.Round((double)epochs / Checkpoints, MidpointRounding.AwayFromZero)) * images;

        var summary = string.Create(CultureInfo.InvariantCulture,
            $"{images} images, {kind}: rank {shape.Rank}, {epochs} epochs = {steps} steps, a checkpoint every {saveEvery} steps.");
        if (kind == Character) summary += " Give a character a trigger word.";

        return rec with
        {
            Epochs = epochs,
            Steps = steps,
            WarmupSteps = Math.Clamp(steps / 20, 10, 100),
            SaveEvery = saveEvery,
            Summary = summary,
        };
    }

    /// <summary>The learning rate is the model's own: the presets already differ on it.</summary>
    private static double? PresetLearningRate(string model) =>
        ConfigResolver.LoadPreset(model).GetValueOrDefault("learning_rate") is string text &&
        double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
            ? value
            : null;
}
