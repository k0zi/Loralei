using System.Globalization;
using Avalonia.Headless.XUnit;
using Loralei.Gui;
using Loralei.Gui.ViewModels;
using Loralei.Tests;
using Material.Icons;
using Xunit;

namespace Loralei.Gui.Tests;

/// <summary>
/// Icons carry meaning here, not just decoration: the Doctor table's shape is what tells a
/// pass from a failure when colour alone is not enough to go on.
/// </summary>
public class IconTests
{
    private static readonly Loralei.Core.LoraleiPaths Real = Loralei.Core.LoraleiPaths.Discover(null, null);

    private static MaterialIconKind Icon(string? status) =>
        (MaterialIconKind)Converters.StatusIcon.Convert(
            status, typeof(MaterialIconKind), null, CultureInfo.InvariantCulture)!;

    [Theory]
    [InlineData("ok", MaterialIconKind.CheckCircle)]
    [InlineData("done", MaterialIconKind.CheckCircle)]
    [InlineData("ready", MaterialIconKind.CheckCircle)]
    [InlineData("warn", MaterialIconKind.AlertCircle)]
    [InlineData("needs disk", MaterialIconKind.AlertCircle)]
    [InlineData("running", MaterialIconKind.ProgressClock)]
    [InlineData("fail", MaterialIconKind.CloseCircle)]
    [InlineData("failed", MaterialIconKind.CloseCircle)]
    [InlineData("skip", MaterialIconKind.MinusCircle)]
    [InlineData("unknown", MaterialIconKind.MinusCircle)]
    [InlineData(null, MaterialIconKind.MinusCircle)]
    public void StatusMapsToATellingGlyph(string? status, MaterialIconKind expected) =>
        Assert.Equal(expected, Icon(status));

    /// <summary>A pass and a failure must not share a shape, or only hue distinguishes them.</summary>
    [Fact]
    public void PassAndFailureAreDifferentShapes()
    {
        Assert.NotEqual(Icon("ok"), Icon("fail"));
        Assert.NotEqual(Icon("ok"), Icon("warn"));
        Assert.NotEqual(Icon("warn"), Icon("fail"));
    }

    [AvaloniaFact]
    public void EveryPageHasItsOwnIcon()
    {
        using var workspace = TempWorkspace.Create(Real);
        var shell = new MainWindowViewModel();

        var icons = shell.Pages.Select(p => p.Icon).ToList();
        Assert.Equal(10, icons.Count);
        Assert.All(icons, i => Assert.True(Enum.IsDefined(i), $"{i} is not a real icon"));
        // Two pages sharing a glyph makes the side menu harder to scan, not easier.
        Assert.Equal(icons.Count, icons.Distinct().Count());
    }
}
