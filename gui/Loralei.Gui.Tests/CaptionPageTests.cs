using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Loralei.DatasetMaker.Config;
using Loralei.Gui.ViewModels;
using Loralei.Gui.Views;
using Loralei.Tests;
using Xunit;

namespace Loralei.Gui.Tests;

/// <summary>
/// The Caption page is Loralei.DatasetMaker's only front end now that it is not a command-line tool,
/// so what the command line used to take has to arrive through these fields.
/// </summary>
public class CaptionPageTests
{
    private static readonly Loralei.Core.LoraleiPaths Real = Loralei.Core.LoraleiPaths.Discover(null, null);

    private static string Dataset(TempWorkspace workspace, params string[] images)
    {
        var folder = Path.Combine(workspace.Directory, "dataset");
        Directory.CreateDirectory(folder);
        for (var i = 0; i < images.Length; i++) Png.Write(Path.Combine(folder, images[i]), 8, 8, seed: i);
        return folder;
    }

    private static CaptionViewModel Page(MainWindowViewModel shell, StubEndpoint endpoint, string dataset)
    {
        var page = shell.Pages.OfType<CaptionViewModel>().Single();
        page.Dataset = dataset;
        page.BaseUrl = endpoint.BaseUrl;
        page.Model = "test-vlm";
        page.ApiKey = "";
        page.MaxRetries = 0;
        return page;
    }

    [AvaloniaFact]
    public async Task CaptioningWritesTheFilesAndReportsItOnThePageAndInTheLog()
    {
        using var workspace = TempWorkspace.Create(Real);
        using var endpoint = new StubEndpoint();
        var dataset = Dataset(workspace, "a.png", "b.png", "c.png");
        File.WriteAllText(Path.Combine(dataset, "c.txt"), "written by hand");

        var shell = new MainWindowViewModel();
        var page = Page(shell, endpoint, dataset);
        page.TriggerWord = "ohwx";
        page.ExtraContext = "The cube is called Bob.";

        await page.CaptionCommand.ExecuteAsync(null);

        Assert.Null(page.Error);
        Assert.Equal("ohwx, A red cube on a table.", File.ReadAllText(Path.Combine(dataset, "a.txt")));
        Assert.Equal("ohwx, A red cube on a table.", File.ReadAllText(Path.Combine(dataset, "b.txt")));
        Assert.Equal("written by hand", File.ReadAllText(Path.Combine(dataset, "c.txt")));

        Assert.Equal((3, 3, 2, 1, 0), (page.Total, page.Done, page.Succeeded, page.Skipped, page.Failed));
        Assert.Equal("ohwx, A red cube on a table.", page.LastCaption);
        Assert.Equal("Done. Captioned 2, skipped 1, failed 0.", page.Status);
        Assert.False(shell.IsBusy);

        Assert.Equal(2, endpoint.Bodies.Count);
        Assert.All(endpoint.Bodies, body => Assert.Contains("The cube is called Bob.", body));
        // The run is told on the page; the output pane gets the start and the result, not a line per image.
        Assert.Equal("Done", page.Phase);
        Assert.Equal(1.0, page.Fraction);
        Assert.Empty(page.InFlight);
        Assert.False(page.IsRunning);
        Assert.NotNull(page.LastThumbnail);
        Assert.Contains(shell.Log.Lines, line => line.StartsWith("$ caption: "));
        Assert.Contains(page.Status, shell.Log.Lines);
        Assert.DoesNotContain(shell.Log.Lines, line => line.EndsWith(".png"));
    }

    [AvaloniaFact]
    public async Task PreviewListsTheWorkAndSendsNothing()
    {
        using var workspace = TempWorkspace.Create(Real);
        using var endpoint = new StubEndpoint();
        var dataset = Dataset(workspace, "a.png", "b.png");

        var shell = new MainWindowViewModel();
        var page = Page(shell, endpoint, dataset);

        await page.PreviewCommand.ExecuteAsync(null);

        Assert.Empty(endpoint.Bodies);
        Assert.Empty(Directory.GetFiles(dataset, "*.txt"));
        Assert.Equal("2 of 2 image(s) would be captioned; 0 skipped. Nothing was sent.", page.Status);
        Assert.Equal(["a.png", "b.png"], page.Pending);
    }

    [AvaloniaFact]
    public async Task AFailedImageIsNamedOnThePage()
    {
        using var workspace = TempWorkspace.Create(Real);
        using var endpoint = new StubEndpoint { Respond = _ => (400, "no vision here") };
        var dataset = Dataset(workspace, "a.png");

        var shell = new MainWindowViewModel();
        var page = Page(shell, endpoint, dataset);

        await page.CaptionCommand.ExecuteAsync(null);

        Assert.Equal(1, page.Failed);
        Assert.Contains("no vision here", Assert.Single(page.Failures));
        Assert.Contains(shell.Log.Lines, line => line.Contains(" FAIL ") && line.Contains("no vision here"));
        Assert.Equal("Done. Captioned 0, skipped 0, failed 1.", page.Status);
    }

    [AvaloniaTheory]
    [InlineData("not a url", "test-vlm")]
    [InlineData("ftp://example.com/v1", "test-vlm")]
    [InlineData("http://127.0.0.1:1/v1", " ")]
    public async Task ABadEndpointIsRefusedBeforeAnythingRuns(string baseUrl, string model)
    {
        using var workspace = TempWorkspace.Create(Real);
        using var endpoint = new StubEndpoint();
        var dataset = Dataset(workspace, "a.png");

        var shell = new MainWindowViewModel();
        var page = Page(shell, endpoint, dataset);
        page.BaseUrl = baseUrl;
        page.Model = model;

        await page.CaptionCommand.ExecuteAsync(null);

        Assert.NotNull(page.Error);
        Assert.Empty(endpoint.Bodies);
        Assert.Empty(Directory.GetFiles(dataset, "*.txt"));
    }

    [AvaloniaFact]
    public void LeavingThePageRemembersTheFieldsAndAnUntouchedPageWritesNothing()
    {
        using var workspace = TempWorkspace.Create(Real);
        var path = Path.Combine(workspace.Directory, "dataset-maker.json");
        Assert.Equal(path, CaptionViewModel.SettingsPath);

        var shell = new MainWindowViewModel();
        var window = new MainWindow { DataContext = shell };
        window.Show();

        try
        {
            var page = shell.Pages.OfType<CaptionViewModel>().Single();
            shell.SelectedPage = page;
            Dispatcher.UIThread.RunJobs();
            shell.SelectedPage = shell.Pages[0];
            Dispatcher.UIThread.RunJobs();
            Assert.False(File.Exists(path));

            shell.SelectedPage = page;
            Dispatcher.UIThread.RunJobs();
            page.Model = "another-vlm";
            page.MaxParallel = 5;
            page.Overwrite = true;
            shell.SelectedPage = shell.Pages[0];
            Dispatcher.UIThread.RunJobs();

            var saved = new ConfigStore(path).Load();
            Assert.Equal("another-vlm", saved.OpenAI.Model);
            Assert.Equal(5, saved.Caption.MaxParallel);

            // A second shell is a restart: the fields come back, the per-run switch does not.
            var again = new MainWindowViewModel().Pages.OfType<CaptionViewModel>().Single();
            Assert.Equal("another-vlm", again.Model);
            Assert.False(again.Overwrite);
        }
        finally { window.Close(); }
    }
}
