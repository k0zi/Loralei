using Avalonia;
using Avalonia.Controls;
using Loralei.ImageResizer.ViewModels;

namespace Loralei.ImageResizer.Views;

/// <summary>
/// The whole resizer as one embeddable view. It used to be the application's main window; the
/// host now owns the window, so what the window's open and close did hangs off attach and detach.
/// </summary>
public partial class MainView : UserControl
{
    // Held on to rather than read back from DataContext: an inherited DataContext can already be
    // gone by the time the view is detached, which is exactly when the settings need writing.
    private MainViewModel? _viewModel;
    private bool _attached;

    public MainView()
    {
        InitializeComponent();
    }

    protected override void OnDataContextChanged(EventArgs e)
    {
        base.OnDataContextChanged(e);

        if (DataContext is MainViewModel viewModel)
            _viewModel = viewModel;
        Start();
    }

    protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
    {
        base.OnAttachedToVisualTree(e);

        _attached = true;
        Start();
    }

    protected override void OnDetachedFromVisualTree(VisualTreeAttachmentEventArgs e)
    {
        base.OnDetachedFromVisualTree(e);

        _attached = false;
        // One write on the way out - leaving the page or closing the window - captures the
        // settings that are not worth saving on every keystroke.
        _viewModel?.PersistSettings();
    }

    // Loading the remembered folder waits until the view is up, so the first paint is not delayed.
    private void Start()
    {
        if (_attached && _viewModel is { } viewModel)
            _ = viewModel.InitializeAsync();
    }
}
