using Avalonia.Headless.XUnit;
using Loralei.Core;
using Loralei.Gui.ViewModels;
using Loralei.Tests;
using Xunit;

namespace Loralei.Gui.Tests;

/// <summary>
/// Run listings are read straight off disk rather than scraped, so every page that offers a
/// run to act on sees the same set - including runs produced by the CLI outside the GUI, and
/// runs that are still going.
/// </summary>
public class RunListingTests
{
    private static readonly LoraleiPaths Real = LoraleiPaths.Discover(null, null);

    [AvaloniaFact]
    public void ExportAndRunsPagesAgreeOnWhatExists()
    {
        using var workspace = TempWorkspace.Create(Real);

        // Two runs as loralei would leave them: one finished, one still going.
        var root = Path.Combine(workspace.Directory, "runs");
        WriteRun(root, "alpha", state: "done", step: 500, model: "qwen-image");
        WriteRun(root, "beta", state: "running", step: 120, model: "qwen-image");

        var shell = new MainWindowViewModel();

        var runs = shell.Pages.OfType<RunsViewModel>().Single();
        runs.ActivateAsync();
        Assert.Equal(["alpha", "beta"], runs.Entries.Select(e => e.Name).Order());
        Assert.Equal("done", runs.Entries.Single(e => e.Name == "alpha").State);
        Assert.Equal(120, runs.Entries.Single(e => e.Name == "beta").Step);
        Assert.Equal("qwen-image", runs.Entries.Single(e => e.Name == "beta").Model);

        // Export offers exactly the same set, and preselects one.
        var export = shell.Pages.OfType<ExportViewModel>().Single();
        export.ActivateAsync();
        Assert.Equal(["alpha", "beta"], export.Runs.Order());
        Assert.NotNull(export.Run);
    }

    [AvaloniaFact]
    public void ARunDirectoryWithNoManifestStillShowsUp()
    {
        using var workspace = TempWorkspace.Create(Real);
        Directory.CreateDirectory(Path.Combine(workspace.Directory, "runs", "half-written"));

        var shell = new MainWindowViewModel();
        var runs = shell.Pages.OfType<RunsViewModel>().Single();
        runs.ActivateAsync();

        var entry = Assert.Single(runs.Entries);
        Assert.Equal("half-written", entry.Name);
        Assert.Equal("unknown", entry.State);
    }

    [AvaloniaFact]
    public void SelectingARunLoadsWhatItRecorded()
    {
        using var workspace = TempWorkspace.Create(Real);
        var root = Path.Combine(workspace.Directory, "runs");
        WriteRun(root, "alpha", state: "done", step: 500, model: "qwen-image");

        var shell = new MainWindowViewModel();
        var runs = shell.Pages.OfType<RunsViewModel>().Single();
        runs.ActivateAsync();

        Assert.NotEmpty(runs.Manifest);
        Assert.Contains(runs.Manifest, f => f.Key == "state" && f.Value == "done");
        Assert.Contains(runs.Manifest, f => f.Key == "step" && f.Value == "500");
    }

    [AvaloniaFact]
    public void SelectingAPageActivatesIt()
    {
        using var workspace = TempWorkspace.Create(Real);
        var shell = new MainWindowViewModel();
        var export = shell.Pages.OfType<ExportViewModel>().Single();

        shell.SelectedPage = export;
        Assert.True(export.Activated);
    }

    /// <summary>Writes a manifest.json in the shape <c>run/layout.py:write_manifest</c> produces.</summary>
    private static void WriteRun(string root, string name, string state, int step, string model)
    {
        var dir = Directory.CreateDirectory(Path.Combine(root, name)).FullName;
        var updated = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        File.WriteAllText(Path.Combine(dir, "manifest.json"),
            $"{{\"model\": \"{model}\", \"state\": \"{state}\", \"step\": {step}, " +
            $"\"updated_at\": {updated}}}");
    }
}
