using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Loralei.Gui.ViewModels;
using Loralei.Gui.Views;

namespace Loralei.Gui;

public partial class App : Application
{
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var viewModel = new MainWindowViewModel();
            // Before the window exists, so it opens in the right variant rather than flashing.
            viewModel.Theme.Initialise();
            viewModel.SelectStartupPage(desktop.Args);
            var window = new MainWindow { DataContext = viewModel };
            desktop.MainWindow = window;
            desktop.ShutdownRequested += (_, _) => viewModel.PersistSettings(window);
        }
        base.OnFrameworkInitializationCompleted();
    }
}
