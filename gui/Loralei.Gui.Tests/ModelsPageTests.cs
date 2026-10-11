using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Loralei.Core.Models;
using Loralei.Gui.ViewModels;
using Loralei.Gui.Views;
using Loralei.Tests;
using Xunit;

namespace Loralei.Gui.Tests;

/// <summary>
/// The Models page against the real CLI: every model the registry knows is offered, each
/// with a tick, and the ticks are what a download acts on. Nothing here downloads anything.
/// </summary>
public class ModelsPageTests
{
    private static readonly Loralei.Core.LoraleiPaths Real = Loralei.Core.LoraleiPaths.Discover(null, null);

    private static async Task<(MainWindowViewModel Shell, ModelsViewModel Page)> Open()
    {
        var shell = new MainWindowViewModel();
        var page = shell.Pages.OfType<ModelsViewModel>().Single();
        var refresh = page.RefreshCommand.ExecuteAsync(null);
        var deadline = DateTime.UtcNow + TimeSpan.FromMinutes(1);
        while (!refresh.IsCompleted && DateTime.UtcNow < deadline)
        {
            Dispatcher.UIThread.RunJobs();
            await Task.Delay(50);
        }
        await refresh;
        return (shell, page);
    }

    [AvaloniaFact]
    public async Task EveryRegisteredModelIsOfferedForDownloadAndForTraining()
    {
        using var cli = CliLock.Acquire();
        using var workspace = TempWorkspace.Create(Real);
        var (_, page) = await Open();

        Assert.True(page.Error is null, page.Error);
        Assert.Equal(ModelRegistry.All.Select(m => m.Key), page.Models.Select(m => m.Row.Key));
        Assert.All(page.Models, m => Assert.False(m.IsChecked));

        // Whatever can be downloaded here can be trained on: the run forms offer the same list.
        var train = page.Shell.Pages.OfType<TrainViewModel>().Single();
        Assert.Equal(page.Models.Select(m => m.Row.Key), train.ModelOptions);
        Assert.Contains("krea-2-raw", train.ModelOptions);
        Assert.DoesNotContain(train.ModelOptions, key => key.Contains("z-image"));
    }

    [AvaloniaFact]
    public async Task DownloadActsOnTheTicksAndRefusesWhenThereAreNone()
    {
        using var cli = CliLock.Acquire();
        using var workspace = TempWorkspace.Create(Real);
        var (shell, page) = await Open();

        await page.PullCommand.ExecuteAsync(null);
        Assert.Contains("Tick at least one model", page.Error);
        // Refused in the page: no command was built, let alone run.
        Assert.DoesNotContain(shell.Log.Lines, l => l.Contains("models pull"));
    }

    [AvaloniaFact]
    public async Task EachRowCarriesATickBoxThatIsOffForACompleteModel()
    {
        using var cli = CliLock.Acquire();
        using var workspace = TempWorkspace.Create(Real);
        var shell = new MainWindowViewModel();
        var window = new MainWindow { DataContext = shell };
        window.Show();
        try
        {
            var page = shell.Pages.OfType<ModelsViewModel>().Single();
            shell.SelectedPage = page;
            var deadline = DateTime.UtcNow + TimeSpan.FromMinutes(1);
            while ((shell.IsBusy || page.Models.Count == 0) && DateTime.UtcNow < deadline)
            {
                Dispatcher.UIThread.RunJobs();
                await Task.Delay(50);
            }
            Dispatcher.UIThread.RunJobs();

            var view = window.GetVisualDescendants().OfType<ModelsView>().Single();
            var boxes = view.GetVisualDescendants().OfType<CheckBox>().ToList();
            Assert.Equal(page.Models.Count, boxes.Count);

            // Ticking in the view is ticking in the view model, which is what Download reads.
            var target = page.Models.First(m => !m.Row.Ready);
            var box = boxes.Single(b => ReferenceEquals(b.DataContext, target));
            Assert.True(box.IsEnabled);
            box.IsChecked = true;
            Assert.True(target.IsChecked);

            Assert.All(boxes, b => Assert.Equal(!((ModelChoice)b.DataContext!).Row.Ready, b.IsEnabled));
        }
        finally { window.Close(); }
    }
}
