using System.Runtime.InteropServices;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Styling;
using Avalonia.VisualTree;
using Loralei.ImageResizer.Models;
using Loralei.ImageResizer.Services;
using Loralei.ImageResizer.ViewModels;
using Loralei.ImageResizer.Views;
using KD.Avalonia.Rice.Controls;
using KD.Avalonia.Rice.Theming;
using SkiaSharp;

namespace Loralei.ImageResizer.Tests;

public sealed class ThemeTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("imageresizer-theme").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private sealed class StubDialogs(string folder) : IDialogService
    {
        public Task<string?> PickFolderAsync(string? startAt) => Task.FromResult<string?>(folder);

        public Task<bool> ShowCropEditorAsync(CropEditorViewModel viewModel) => Task.FromResult(false);
    }

    [AvaloniaFact]
    public void Settings_are_restored_into_the_view_model()
    {
        var store = new AppSettingsStore(Path.Combine(_folder, "settings.json"));
        var saved = new AppSettings
        {
            TargetSize = 2048,
            DefaultMode = ResizeMode.Pad,
            PadFill = PadFill.White,
            JpegQuality = 80,
            LastFolder = _folder,
        };
        var model = new MainViewModel(new StubDialogs(_folder), store, saved);

        Assert.Equal(2048, model.SelectedTargetSize);
        Assert.Equal(ResizeMode.Pad, model.SelectedModeOption.Value);
        Assert.Equal(PadFill.White, model.SelectedPadFillOption.Value);
        Assert.Equal(80, model.JpegQuality);
        Assert.Equal(_folder, model.FolderPath);
    }

    [AvaloniaFact]
    public void A_remembered_folder_that_no_longer_exists_is_ignored()
    {
        var store = new AppSettingsStore(Path.Combine(_folder, "settings.json"));

        var model = new MainViewModel(
            new StubDialogs(_folder), store,
            new AppSettings { LastFolder = Path.Combine(_folder, "deleted-last-week") });

        Assert.Null(model.FolderPath);
    }

    [AvaloniaFact]
    public void The_view_actually_repaints_when_the_theme_changes()
    {
        // Guards against a surface painted with a brush that does not follow the variant: such a
        // window looks right in light mode and then renders identically in dark.
        ImageFixtures.Write(_folder, "a.jpg", 400, 300, SKEncodedImageFormat.Jpeg);

        var app = Application.Current!;
        var store = new AppSettingsStore(Path.Combine(_folder, "settings.json"));
        app.RequestedThemeVariant = ThemeVariant.Light;
        try
        {
            var window = new RiceWindow
            {
                Width = 1100,
                Height = 720,
                Content = new MainView
                {
                    DataContext = new MainViewModel(new StubDialogs(_folder), store, new AppSettings()),
                },
            };
            window.Show();

            var light = Pixels(window);
            // The same call the title bar's light/dark button makes.
            ThemeManager.Instance.ToggleVariant();
            var dark = Pixels(window);

            Assert.True(ThemeManager.Instance.IsDark);
            Assert.NotEqual(light, dark);
            window.Close();
        }
        finally
        {
            app.RequestedThemeVariant = ThemeVariant.Default;
        }
    }

    private static byte[] Pixels(Window window)
    {
        using var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);

        var size = frame.PixelSize;
        var stride = size.Width * 4;
        var buffer = new byte[stride * size.Height];

        var pinned = GCHandle.Alloc(buffer, GCHandleType.Pinned);
        try
        {
            frame.CopyPixels(new PixelRect(size), pinned.AddrOfPinnedObject(), buffer.Length, stride);
        }
        finally
        {
            pinned.Free();
        }

        return buffer;
    }
}
