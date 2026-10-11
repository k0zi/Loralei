using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Loralei.ImageResizer.Services;
using Loralei.Gui.ViewModels;
using Loralei.Gui.Views;
using Loralei.Tests;
using Xunit;

namespace Loralei.Gui.Tests;

/// <summary>
/// The Resize page is the Loralei.ImageResizer project's own view on one of our pages. Its behaviour
/// has its own test project; what is ours to get right is the hosting.
/// </summary>
public class ResizePageTests
{
    private static readonly Loralei.Core.LoraleiPaths Real = Loralei.Core.LoraleiPaths.Discover(null, null);

    [AvaloniaFact]
    public void ThePageHostsTheResizerOutsideThePageScroller()
    {
        using var workspace = TempWorkspace.Create(Real);
        var shell = new MainWindowViewModel();
        var window = new MainWindow { DataContext = shell };
        window.Show();

        try
        {
            var page = shell.Pages.OfType<ResizeViewModel>().Single();
            shell.SelectedPage = page;
            Dispatcher.UIThread.RunJobs();

            var view = window.GetVisualDescendants().OfType<Loralei.ImageResizer.Views.MainView>().Single();
            Assert.Same(page.Resizer, view.DataContext);

            // Inside a scroller the grid would be given endless height, never scroll itself, and
            // push the Save bar off the bottom of a long folder.
            Assert.Empty(view.GetVisualAncestors().OfType<ScrollViewer>());
            Assert.True(view.Bounds.Height > 0);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public void LeavingThePageWritesTheResizerSettingsBesideOurs()
    {
        using var workspace = TempWorkspace.Create(Real);
        var shell = new MainWindowViewModel();
        var window = new MainWindow { DataContext = shell };
        window.Show();

        try
        {
            var page = shell.Pages.OfType<ResizeViewModel>().Single();
            shell.SelectedPage = page;
            Dispatcher.UIThread.RunJobs();
            page.Resizer.SelectedTargetSize = 2048;

            shell.SelectedPage = shell.Pages[0];
            Dispatcher.UIThread.RunJobs();

            var path = Path.Combine(workspace.Directory, "image-resizer.json");
            Assert.Equal(path, ResizeViewModel.SettingsPath);
            Assert.Equal(2048, new AppSettingsStore(path).Load().TargetSize);
        }
        finally { window.Close(); }
    }
}
