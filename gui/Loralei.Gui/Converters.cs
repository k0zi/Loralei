using System.Globalization;
using Avalonia;
using Avalonia.Data.Converters;
using Avalonia.Media;
using Avalonia.Styling;
using Material.Icons;

namespace Loralei.Gui;

public static class Converters
{
    /// <summary>
    /// Doctor check status, run state or model readiness to its colour.
    /// <para>
    /// Resolved from the active theme rather than from fixed <see cref="Brushes"/> values:
    /// the dark-mode greens and ambers wash out on a light background.
    /// </para>
    /// </summary>
    public static readonly IValueConverter StatusBrush =
        new FuncValueConverter<string?, IBrush>(status => Themed(status switch
        {
            "ok" or "done" or "ready" => "OkBrush",
            "warn" or "running" or "incomplete" or "needs disk" => "WarnBrush",
            "fail" or "failed" => "FailBrush",
            _ => "MutedBrush",
        }));

    /// <summary>
    /// Looks a brush up in the application's resources for the current variant. A converter
    /// has no visual parent to resolve against, so it goes to the Application directly.
    /// </summary>
    private static IBrush Themed(string key)
    {
        var app = Application.Current;
        var variant = app?.ActualThemeVariant ?? ThemeVariant.Dark;
        return app is not null && app.TryGetResource(key, variant, out var found) && found is IBrush brush
            ? brush
            : Brushes.Gray;
    }

    /// <summary>
    /// The same status as a glyph. Shape carries the meaning as well as colour, which the
    /// PASS/WARN/FAIL text alone left entirely to hue.
    /// </summary>
    public static readonly IValueConverter StatusIcon =
        new FuncValueConverter<string?, MaterialIconKind>(status => status switch
        {
            "ok" or "done" or "ready" => MaterialIconKind.CheckCircle,
            "warn" or "incomplete" or "needs disk" => MaterialIconKind.AlertCircle,
            "running" => MaterialIconKind.ProgressClock,
            "fail" or "failed" => MaterialIconKind.CloseCircle,
            _ => MaterialIconKind.MinusCircle,
        });

    /// <summary>The short marker the CLI prints for the same statuses.</summary>
    public static readonly IValueConverter StatusMark =
        new FuncValueConverter<string?, string>(status => status switch
        {
            "ok" => "PASS",
            "warn" => "WARN",
            "fail" => "FAIL",
            "skip" => "SKIP",
            _ => status?.ToUpperInvariant() ?? "",
        });

    public static readonly IValueConverter Percent =
        new FuncValueConverter<double, string>(value => $"{value * 100:F1}%");

    public static readonly IValueConverter FileName =
        new FuncValueConverter<string?, string>(path =>
            string.IsNullOrEmpty(path) ? "" : Path.GetFileName(path));

    /// <summary>
    /// Formats a loss for the live readout. Losses here sit around 0.2-2.5, so four
    /// decimals is what the CLI's own progress bar shows.
    /// </summary>
    public static readonly IValueConverter Loss =
        new FuncValueConverter<double, string>(value =>
            value == 0 ? "-" : value.ToString("F4", CultureInfo.InvariantCulture));

    public static readonly IValueConverter Vram =
        new FuncValueConverter<double, string>(value =>
            value == 0 ? "-" : $"{value.ToString("F2", CultureInfo.InvariantCulture)} GiB");
}
