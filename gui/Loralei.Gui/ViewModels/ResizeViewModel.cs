using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Loralei.ImageResizer.Services;
using Loralei.ImageResizer.Views;
using Loralei.Core;
using Material.Icons;

namespace Loralei.Gui.ViewModels;

/// <summary>
/// The Image Resizer (the <c>Loralei.ImageResizer</c> project), hosted as a page. It is the one page
/// that runs no loralei command: it works on the image files directly, so it neither takes
/// the command lock nor writes to the output pane.
/// </summary>
public sealed class ResizeViewModel : PageViewModel
{
    public override string Title => "Resize";
    public override MaterialIconKind Icon => MaterialIconKind.ImageSizeSelectLarge;
    public override string Blurb =>
        "Turn a folder of images into exact squares. Originals are untouched; results go to saved_resized inside it.";

    public override bool FillsPage => true;

    public Loralei.ImageResizer.ViewModels.MainViewModel Resizer { get; }

    /// <summary>
    /// The resizer's own preferences, kept beside ours so <c>LORALEI_GUI_SETTINGS</c> moves
    /// both and the tests stay out of the real ones.
    /// </summary>
    public static string SettingsPath =>
        Path.Combine(Path.GetDirectoryName(GuiSettings.DefaultPath) ?? "", "image-resizer.json");

    public ResizeViewModel(MainWindowViewModel shell) : base(shell)
    {
        var store = new AppSettingsStore(SettingsPath);
        // The view models are built before the window is, so the dialogs find their owner late.
        var dialogs = new DialogService(() =>
            (Application.Current?.ApplicationLifetime as IClassicDesktopStyleApplicationLifetime)?.MainWindow);
        Resizer = new Loralei.ImageResizer.ViewModels.MainViewModel(dialogs, store, store.Load());
    }
}
