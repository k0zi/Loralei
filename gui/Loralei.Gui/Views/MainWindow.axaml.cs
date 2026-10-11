using System.Collections.Specialized;
using Avalonia.Interactivity;
using Avalonia.Threading;
using KD.Avalonia.Rice.Controls;
using Loralei.Gui.ViewModels;

namespace Loralei.Gui.Views;

public partial class MainWindow : RiceWindow
{
    public MainWindow()
    {
        InitializeComponent();
        DataContextChanged += (_, _) => HookLog();
    }

    private async void OnSettingsRequested(object? sender, RoutedEventArgs e) =>
        await new SettingsWindow { DataContext = DataContext }.ShowDialog(this);

    private void HookLog()
    {
        if (DataContext is not MainWindowViewModel vm) return;

        vm.Log.Lines.CollectionChanged += (_, e) =>
        {
            if (!vm.Log.AutoScroll || e.Action == NotifyCollectionChangedAction.Remove) return;
            // Let the ItemsControl lay the new line out before scrolling to it.
            Dispatcher.UIThread.Post(() => LogScroller.ScrollToEnd(), DispatcherPriority.Background);
        };
    }
}
