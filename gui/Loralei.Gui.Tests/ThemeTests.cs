using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Media;
using Avalonia.Platform;
using Avalonia.Styling;
using Loralei.Core;
using Loralei.Gui.Controls;
using Loralei.Gui.Theming;
using Loralei.Gui.ViewModels;
using Loralei.Tests;
using Xunit;

namespace Loralei.Gui.Tests;

/// <summary>
/// The theme is only half Rice's doing. Rice supplies the control styles and the colour
/// themes; remembering System / Light / Dark is ours, and our own palette has to answer to
/// the variant too or light mode arrives half dark.
/// </summary>
public class ThemeTests
{
    private static readonly LoraleiPaths Real = LoraleiPaths.Discover(null, null);

    [Theory]
    [InlineData("light", ThemePreference.Light)]
    [InlineData("LIGHT", ThemePreference.Light)]
    [InlineData("dark", ThemePreference.Dark)]
    [InlineData("system", ThemePreference.System)]
    [InlineData("nonsense", ThemePreference.System)]
    [InlineData(null, ThemePreference.System)]
    public void PreferenceParsingFallsBackToSystem(string? stored, ThemePreference expected) =>
        Assert.Equal(expected, ThemeService.ParsePreference(stored));

    [AvaloniaFact]
    public void ChoosingAVariantAppliesItAndWritesItDown()
    {
        using var workspace = TempWorkspace.Create(Real);
        var shell = new MainWindowViewModel();
        shell.Theme.Initialise();

        shell.SelectedTheme = ThemePreference.Light;
        Assert.Equal(ThemeVariant.Light, Application.Current!.RequestedThemeVariant);

        shell.SelectedTheme = ThemePreference.Dark;
        Assert.Equal(ThemeVariant.Dark, Application.Current.RequestedThemeVariant);

        // Survives a restart.
        Assert.Equal("dark", GuiSettings.Load(workspace.SettingsPath).Theme);
    }

    /// <summary>System is Avalonia's own Default variant, which tracks the desktop live.</summary>
    [AvaloniaFact]
    public void SystemHandsTheVariantBackToTheDesktop()
    {
        using var workspace = TempWorkspace.Create(Real);
        var shell = new MainWindowViewModel();
        shell.Theme.Initialise();
        shell.SelectedTheme = ThemePreference.Dark;

        shell.SelectedTheme = ThemePreference.System;

        Assert.Equal(ThemeVariant.Default, Application.Current!.RequestedThemeVariant);
        Assert.Equal("system", GuiSettings.Load(workspace.SettingsPath).Theme);
    }

    /// <summary>
    /// The palette the pages and the loss chart draw from. If these did not change with the
    /// variant, light mode would show a dark chart and a dark console pane on a white page.
    /// </summary>
    [AvaloniaTheory]
    [InlineData("OkBrush")]
    [InlineData("WarnBrush")]
    [InlineData("FailBrush")]
    [InlineData("MutedBrush")]
    [InlineData("LogPaneBrush")]
    [InlineData("ProblemBrush")]
    [InlineData("ChartBackgroundBrush")]
    [InlineData("ChartGridBrush")]
    [InlineData("ChartSeriesBrush")]
    [InlineData("ChartTrendBrush")]
    public void ThePaletteAnswersToTheVariant(string key)
    {
        var app = Application.Current!;

        Assert.True(app.TryGetResource(key, ThemeVariant.Dark, out var darkValue), $"{key} missing in Dark");
        Assert.True(app.TryGetResource(key, ThemeVariant.Light, out var lightValue), $"{key} missing in Light");

        var dark = Assert.IsAssignableFrom<ISolidColorBrush>(darkValue).Color;
        var light = Assert.IsAssignableFrom<ISolidColorBrush>(lightValue).Color;
        Assert.True(dark != light, $"{key} is {dark} in both variants");
    }

    /// <summary>
    /// Goes through the chart's own lookup rather than the application's.
    /// <para>
    /// This is the test that was missing: asserting the palette differs per variant passed
    /// happily while the chart itself still resolved nothing and painted its dark fallback,
    /// because a control does not search the same resource chain the Application does.
    /// </para>
    /// </summary>
    [AvaloniaFact]
    public void TheChartResolvesItsOwnBrushesPerVariant()
    {
        var chart = new LossChart();
        var scope = new ThemeVariantScope { Child = chart };
        // Attached, or the control has no resource chain to search at all.
        var window = new Window { Content = scope };
        window.Show();

        try
        {
            var keys = new[]
            {
                "ChartBackgroundBrush", "ChartGridBrush", "ChartSeriesBrush",
                "ChartTrendBrush", "MutedBrush",
            };

            scope.RequestedThemeVariant = ThemeVariant.Dark;
            var dark = keys.ToDictionary(k => k, k => Colour(chart.Brush(k, "#FF00FF")));

            scope.RequestedThemeVariant = ThemeVariant.Light;
            var light = keys.ToDictionary(k => k, k => Colour(chart.Brush(k, "#FF00FF")));

            foreach (var key in keys)
            {
                // Magenta means the lookup failed and the fallback was painted.
                Assert.True(dark[key] != Colors.Magenta, $"{key} did not resolve in Dark");
                Assert.True(light[key] != Colors.Magenta, $"{key} did not resolve in Light");
                Assert.True(dark[key] != light[key], $"{key} is {dark[key]} in both variants");
            }
        }
        finally { window.Close(); }
    }

    private static Color Colour(IBrush brush) => Assert.IsAssignableFrom<ISolidColorBrush>(brush).Color;

    /// <summary>Rice's own brushes have to be reachable, since our cards bind to them.</summary>
    [AvaloniaTheory]
    [InlineData("RiceSurfaceBrush")]
    [InlineData("RiceBorderBrush")]
    [InlineData("RiceBackgroundBrush")]
    [InlineData("RiceForegroundBrush")]
    public void RiceBrushesAreAvailable(string key)
    {
        var app = Application.Current!;
        Assert.True(app.TryGetResource(key, ThemeVariant.Dark, out _), $"{key} missing in Dark");
        Assert.True(app.TryGetResource(key, ThemeVariant.Light, out _), $"{key} missing in Light");
    }
}
