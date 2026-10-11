using Avalonia;
using Avalonia.Controls;
using Avalonia.Platform.Storage;

namespace Loralei.Gui.Controls;

/// <summary>A text box for a path plus a Browse button, for folders or for one file type.</summary>
public partial class PathPicker : UserControl
{
    public static readonly StyledProperty<string?> PathProperty =
        AvaloniaProperty.Register<PathPicker, string?>(nameof(Path), defaultBindingMode: Avalonia.Data.BindingMode.TwoWay);

    public static readonly StyledProperty<bool> PickFolderProperty =
        AvaloniaProperty.Register<PathPicker, bool>(nameof(PickFolder), defaultValue: true);

    public static readonly StyledProperty<string> TitleProperty =
        AvaloniaProperty.Register<PathPicker, string>(nameof(Title), defaultValue: "Select");

    public static readonly StyledProperty<string?> WatermarkProperty =
        AvaloniaProperty.Register<PathPicker, string?>(nameof(PlaceholderText));

    /// <summary>Extensions offered when <see cref="PickFolder"/> is false, e.g. "safetensors".</summary>
    public static readonly StyledProperty<string?> ExtensionProperty =
        AvaloniaProperty.Register<PathPicker, string?>(nameof(Extension));

    public string? Path { get => GetValue(PathProperty); set => SetValue(PathProperty, value); }
    public bool PickFolder { get => GetValue(PickFolderProperty); set => SetValue(PickFolderProperty, value); }
    public string Title { get => GetValue(TitleProperty); set => SetValue(TitleProperty, value); }
    public string? PlaceholderText { get => GetValue(WatermarkProperty); set => SetValue(WatermarkProperty, value); }
    public string? Extension { get => GetValue(ExtensionProperty); set => SetValue(ExtensionProperty, value); }

    public PathPicker()
    {
        InitializeComponent();
    }

    private async void Browse(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var storage = TopLevel.GetTopLevel(this)?.StorageProvider;
        if (storage is null) return;

        var start = await SuggestedStart(storage);

        if (PickFolder)
        {
            var folders = await storage.OpenFolderPickerAsync(new FolderPickerOpenOptions
            {
                Title = Title,
                AllowMultiple = false,
                SuggestedStartLocation = start,
            });
            if (folders.Count > 0) Path = folders[0].Path.LocalPath;
            return;
        }

        var files = await storage.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = Title,
            AllowMultiple = false,
            SuggestedStartLocation = start,
            FileTypeFilter = Extension is { Length: > 0 } extension
                ?
                [
                    new FilePickerFileType(extension) { Patterns = [$"*.{extension}"] },
                    FilePickerFileTypes.All,
                ]
                : null,
        });
        if (files.Count > 0) Path = files[0].Path.LocalPath;
    }

    /// <summary>Reopens where the current value points, so repeated picks stay close by.</summary>
    private async Task<IStorageFolder?> SuggestedStart(IStorageProvider storage)
    {
        if (Path is not { Length: > 0 } current) return null;
        try
        {
            var directory = Directory.Exists(current) ? current : System.IO.Path.GetDirectoryName(current);
            return directory is { Length: > 0 } && Directory.Exists(directory)
                ? await storage.TryGetFolderFromPathAsync(directory)
                : null;
        }
        catch (Exception e) when (e is IOException or ArgumentException) { return null; }
    }
}
