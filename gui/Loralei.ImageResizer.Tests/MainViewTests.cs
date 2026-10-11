using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Loralei.ImageResizer.Models;
using Loralei.ImageResizer.Services;
using Loralei.ImageResizer.ViewModels;
using Loralei.ImageResizer.Views;
using SkiaSharp;

namespace Loralei.ImageResizer.Tests;

/// <summary>Smoke tests over the real view and view model, so the XAML and bindings are exercised.</summary>
public sealed class MainViewTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("imageresizer-ui").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string SettingsPath => Path.Combine(_folder, "settings.json");

    /// <summary>A view model wired to a throwaway settings file, so tests never touch the real one.</summary>
    private MainViewModel CreateModel(AppSettings? settings = null)
    {
        var store = new AppSettingsStore(SettingsPath);
        settings ??= new AppSettings();
        return new MainViewModel(new StubDialogs(_folder), store, settings);
    }

    /// <summary>A bare window around the view, standing in for the page the real host puts it on.</summary>
    private static Window Host(MainViewModel model) =>
        new() { Content = new MainView { DataContext = model }, Width = 1100, Height = 720 };

    private sealed class StubDialogs(string folder) : IDialogService
    {
        public Task<string?> PickFolderAsync(string? startAt) => Task.FromResult<string?>(folder);

        public Task<bool> ShowCropEditorAsync(CropEditorViewModel viewModel) => Task.FromResult(false);
    }

    [AvaloniaFact]
    public async Task Choosing_a_folder_puts_one_card_in_the_grid_per_image()
    {
        ImageFixtures.Write(_folder, "a.jpg", 400, 300, SKEncodedImageFormat.Jpeg);
        ImageFixtures.Write(_folder, "b.png", 300, 400, SKEncodedImageFormat.Png);
        File.WriteAllText(Path.Combine(_folder, "readme.txt"), "ignored");

        var model = CreateModel();
        var window = Host(model);
        window.Show();

        await model.BrowseCommand.ExecuteAsync(null);
        Dispatcher.UIThread.RunJobs();

        var list = window.GetVisualDescendants().OfType<ListBox>().Single(l => l.Name == "ImageGrid");
        Assert.Equal(2, list.ItemCount);
        Assert.Equal(2, model.IncludedCount);
        Assert.Equal("2 images loaded.", model.StatusText);
    }

    [AvaloniaFact]
    public async Task Changing_the_default_mode_reselects_every_square()
    {
        ImageFixtures.Write(_folder, "wide.jpg", 1600, 900, SKEncodedImageFormat.Jpeg);

        var model = CreateModel();
        await model.BrowseCommand.ExecuteAsync(null);

        var item = Assert.Single(model.Images);
        Assert.Equal(900, item.Selection.Size, 6); // crop is the default

        model.SelectedModeOption = model.ModeOptions.Single(o => o.Value == ResizeMode.Pad);

        Assert.Equal(1600, item.Selection.Size, 6);
        Assert.True(ResizeGeometry.NeedsFill(item.Selection, 1600, 900));
    }

    [AvaloniaFact]
    public async Task A_hand_placed_square_survives_a_change_of_default_mode()
    {
        ImageFixtures.Write(_folder, "wide.jpg", 1600, 900, SKEncodedImageFormat.Jpeg);

        var model = CreateModel();
        await model.BrowseCommand.ExecuteAsync(null);

        var item = Assert.Single(model.Images);
        item.ApplyManualSelection(new SquareSelection(120, 60, 500));

        model.SelectedModeOption = model.ModeOptions.Single(o => o.Value == ResizeMode.Pad);

        Assert.Equal(ResizeMode.Manual, item.Mode);
        Assert.Equal(new SquareSelection(120, 60, 500), item.Selection);

        // "Apply mode to all" is the explicit override that does reset it.
        model.ApplyDefaultToAllCommand.Execute(null);
        Assert.Equal(ResizeMode.Pad, item.Mode);
        Assert.Equal(1600, item.Selection.Size, 6);
    }

    [AvaloniaFact]
    public async Task Changing_the_output_size_updates_the_upscale_warning()
    {
        ImageFixtures.Write(_folder, "small.jpg", 600, 600, SKEncodedImageFormat.Jpeg);

        var model = CreateModel();
        await model.BrowseCommand.ExecuteAsync(null);

        var item = Assert.Single(model.Images);
        Assert.True(item.IsUpscaling); // 600 px source into the default 1024 px output

        model.SelectedTargetSize = 512;

        Assert.False(item.IsUpscaling);
    }

    [AvaloniaFact]
    public async Task Saving_writes_every_included_image_and_reports_it()
    {
        ImageFixtures.Write(_folder, "a.jpg", 1200, 800, SKEncodedImageFormat.Jpeg);
        ImageFixtures.Write(_folder, "b.png", 500, 900, SKEncodedImageFormat.Png);

        var model = CreateModel();
        await model.BrowseCommand.ExecuteAsync(null);
        model.SelectedTargetSize = 512;

        await model.SaveCommand.ExecuteAsync(null);

        var written = Directory.GetFiles(Path.Combine(_folder, ImageScanner.OutputFolderName));
        Assert.Equal(2, written.Length);
        Assert.All(model.Images, i => Assert.Equal(ItemStatus.Saved, i.Status));
        Assert.Contains("Saved 2 images", model.StatusText);

        foreach (var file in written)
        {
            Assert.True(ImageDecoder.TryReadInfo(file, out var w, out var h));
            Assert.Equal((512, 512), (w, h));
        }
    }

    [AvaloniaFact]
    public async Task A_remembered_folder_is_loaded_at_startup()
    {
        ImageFixtures.Write(_folder, "a.jpg", 400, 300, SKEncodedImageFormat.Jpeg);
        ImageFixtures.Write(_folder, "b.png", 300, 400, SKEncodedImageFormat.Png);

        var model = CreateModel(new AppSettings { LastFolder = _folder });
        await model.InitializeAsync();

        Assert.Equal(_folder, model.FolderPath);
        Assert.Equal(2, model.Images.Count);
    }

    [AvaloniaFact]
    public async Task Initialising_twice_does_not_load_the_folder_twice()
    {
        ImageFixtures.Write(_folder, "a.jpg", 400, 300, SKEncodedImageFormat.Jpeg);

        var model = CreateModel(new AppSettings { LastFolder = _folder });
        var first = model.InitializeAsync();
        var second = model.InitializeAsync();
        await Task.WhenAll(first, second);

        // Two concurrent scans would each clear and then append, leaving the grid doubled.
        Assert.Single(model.Images);
    }

    [AvaloniaFact]
    public async Task Closing_the_window_around_the_view_writes_the_settings_out()
    {
        var model = CreateModel();
        var window = Host(model);
        window.Show();

        await model.BrowseCommand.ExecuteAsync(null);
        model.SelectedTargetSize = 2048;
        model.JpegQuality = 61;

        window.Close();

        var saved = new AppSettingsStore(SettingsPath).Load();
        Assert.Equal(2048, saved.TargetSize);
        Assert.Equal(61, saved.JpegQuality);
        Assert.Equal(_folder, saved.LastFolder);
    }

    [AvaloniaFact]
    public async Task Excluded_images_are_not_written()
    {
        ImageFixtures.Write(_folder, "keep.jpg", 800, 800, SKEncodedImageFormat.Jpeg);
        ImageFixtures.Write(_folder, "skip.jpg", 800, 800, SKEncodedImageFormat.Jpeg);

        var model = CreateModel();
        await model.BrowseCommand.ExecuteAsync(null);
        model.Images.Single(i => i.Source.FileName == "skip.jpg").IsIncluded = false;

        await model.SaveCommand.ExecuteAsync(null);

        var written = Directory.GetFiles(Path.Combine(_folder, ImageScanner.OutputFolderName))
            .Select(f => Path.GetFileName(f))
            .ToArray();
        Assert.Equal(["keep.jpg"], written);
    }
}
