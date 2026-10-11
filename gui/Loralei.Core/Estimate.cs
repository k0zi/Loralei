using System.Globalization;
using Loralei.Core.Config;

namespace Loralei.Core;

/// <param name="BlockGib">Resident cost of one transformer block in bf16.</param>
/// <param name="FixedGib">Everything that is not a swappable block: embedders, final layer, allocator slack.</param>
public sealed record ModelCost(double BlockGib, int Blocks, double FixedGib, double ActivationGibPerMpixel,
    double SecondsPerStepPerMpixel);

public sealed record Estimate(double PeakGib, double SecondsPerStep, double HostGib, bool Measured)
{
    public bool Fits(double deviceGib, double headroom = 0.9) => PeakGib <= deviceGib * headroom;
}

/// <summary>
/// Predicts peak VRAM and step time before a run starts.
/// <para>
/// The shape of the model is measured: on a Radeon RX 9060 XT (gfx1200, 16 GiB), a
/// block-swapped transformer cost a fixed part plus a constant per resident block, and
/// activations grew with megapixels. The per-model numbers below are not - they are the
/// size of each model's bf16 weights divided by its block count, with the fixed and
/// activation terms carried over. Every estimate is therefore flagged as extrapolated until
/// someone replaces a row with a measurement.
/// </para>
/// </summary>
public static class Estimator
{
    private static readonly Dictionary<string, ModelCost> Costs = new()
    {
        ["qwen-image"] = new(0.640, 60, 2.40, 1.90, 11.0),
        ["krea-2-raw"] = new(0.850, 28, 2.00, 1.70, 7.0),
        ["ltx-2.3"] = new(0.740, 48, 2.40, 0.80, 10.0),
    };

    /// <summary>Blocks kept resident beyond the one executing (prefetch window plus slack).</summary>
    private const int ResidencySlack = 4;

    /// <summary>
    /// fp8 storage halves the weight bytes of every quantizable Linear, which is most but not
    /// all of a block (norms, biases and skipped embedders stay bf16).
    /// </summary>
    private const double Fp8BlockFactor = 0.55;

    public static Estimate For(RunConfig cfg)
    {
        if (!Costs.TryGetValue(cfg.Model, out var cost)) return new Estimate(0, 0, 0, false);

        var swap = Math.Max(0, Math.Min(cfg.BlocksToSwap, cost.Blocks - 1));
        var resident = cost.Blocks - swap + Math.Min(swap, ResidencySlack);

        var blockGib = cost.BlockGib * (cfg.Fp8Base ? Fp8BlockFactor : 1.0);
        var megapixels = (double)cfg.Resolution * cfg.Resolution / 1_000_000 * cfg.BatchSize;

        var activations = cost.ActivationGibPerMpixel * megapixels;
        if (!cfg.GradientCheckpointing) activations *= 4.0;

        return new Estimate(
            PeakGib: cost.FixedGib + blockGib * resident + activations,
            SecondsPerStep: cost.SecondsPerStepPerMpixel * megapixels * (1.0 + 0.5 * swap / cost.Blocks),
            HostGib: cost.BlockGib * swap,
            Measured: false);
    }

    /// <summary>The marker <see cref="Format"/> appends when the run will not fit the card.</summary>
    public const string OverBudgetMarker = "[!] over the";

    public static string Format(Estimate est, RunConfig cfg, double? deviceGib = null)
    {
        static string F1(double v) => v.ToString("F1", CultureInfo.InvariantCulture);

        var totalMinutes = est.SecondsPerStep * cfg.MaxSteps / 60;
        var line = string.Join(" | ",
            $"~{F1(est.PeakGib)} GiB VRAM",
            est.HostGib > 0 ? $"~{F1(est.HostGib)} GiB pinned host RAM" : "no block swap",
            $"~{F1(est.SecondsPerStep)} s/step",
            $"~{F1(totalMinutes / 60)} h for {cfg.MaxSteps} steps");
        if (deviceGib is > 0 && !est.Fits(deviceGib.Value))
            line += $"  {OverBudgetMarker} {F1(deviceGib.Value)} GiB card - raise --blocks-to-swap or use --fp8-base";
        if (!est.Measured) line += "  (extrapolated, not measured on this model)";
        return line;
    }
}
