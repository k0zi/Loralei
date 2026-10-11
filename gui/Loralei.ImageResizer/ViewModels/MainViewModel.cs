using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Loralei.ImageResizer.Models;
using Loralei.ImageResizer.Services;

namespace Loralei.ImageResizer.ViewModels;

public partial class MainViewModel : ViewModelBase
{
    private readonly IDialogService _dialogs;
    private readonly AppSettingsStore _settingsStore;
    private bool _initialized;
    private CancellationTokenSource? _thumbnailCts;
    private CancellationTokenSource? _saveCts;

    public MainViewModel(IDialogService dialogs, AppSettingsStore settingsStore, AppSettings settings)
    {
        _dialogs = dialogs;
        _settingsStore = settingsStore;

        SelectedTargetSize = settings.TargetSize;
        SelectedModeOption = ModeOptions.First(o => o.Value == settings.DefaultMode);
        SelectedPadFillOption = PadFillOptions.First(o => o.Value == settings.PadFill);
        JpegQuality = settings.JpegQuality;

        // A folder that has since been deleted or unmounted just means we start empty.
        if (!string.IsNullOrEmpty(settings.LastFolder) && Directory.Exists(settings.LastFolder))
            FolderPath = settings.LastFolder;
    }

    /// <summary>
    /// Starts loading the remembered folder, if there was one. Called once the view is up, and
    /// guarded so a second call cannot run a scan concurrently with the first and double the grid.
    /// </summary>
    public Task InitializeAsync()
    {
        if (_initialized || !HasFolder)
            return Task.CompletedTask;

        _initialized = true;
        return LoadFolderAsync();
    }

    public ObservableCollection<ImageItemViewModel> Images { get; } = [];

    public IReadOnlyList<int> TargetSizes { get; } = ResizeSettings.TargetSizes;

    public IReadOnlyList<EnumOption<ResizeMode>> ModeOptions { get; } =
    [
        new(ResizeMode.Crop, "Crop — fill the square"),
        new(ResizeMode.Pad, "Pad — fit the whole image"),
        new(ResizeMode.Manual, "Manual — edit per image"),
    ];

    public IReadOnlyList<EnumOption<PadFill>> PadFillOptions { get; } =
    [
        new(PadFill.Blur, "Blurred image"),
        new(PadFill.White, "White"),
        new(PadFill.Black, "Black"),
        new(PadFill.Transparent, "Transparent (PNG)"),
    ];

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasFolder))]
    [NotifyCanExecuteChangedFor(nameof(ReloadCommand), nameof(SaveCommand))]
    public partial string? FolderPath { get; set; }

    public bool HasFolder => !string.IsNullOrEmpty(FolderPath);

    [ObservableProperty]
    public partial int SelectedTargetSize { get; set; }

    [ObservableProperty]
    public partial EnumOption<ResizeMode> SelectedModeOption { get; set; } = null!;

    [ObservableProperty]
    public partial EnumOption<PadFill> SelectedPadFillOption { get; set; } = null!;

    [ObservableProperty]
    public partial int JpegQuality { get; set; } = 95;

    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(BrowseCommand), nameof(ReloadCommand), nameof(SaveCommand), nameof(CancelCommand))]
    public partial bool IsBusy { get; set; }

    [ObservableProperty]
    public partial double ProgressValue { get; set; }

    [ObservableProperty]
    public partial bool IsProgressVisible { get; set; }

    [ObservableProperty]
    public partial string StatusText { get; set; } = "Choose a folder to get started.";

    [ObservableProperty]
    public partial ImageItemViewModel? SelectedImage { get; set; }

    /// <summary>The current state of every remembered setting.</summary>
    public AppSettings CurrentSettings => new()
    {
        TargetSize = SelectedTargetSize,
        DefaultMode = SelectedModeOption.Value,
        PadFill = SelectedPadFillOption.Value,
        JpegQuality = JpegQuality,
        LastFolder = FolderPath,
    };

    /// <summary>Writes the settings out. Called after a folder loads and when the view goes away.</summary>
    public void PersistSettings() => _settingsStore.Save(CurrentSettings);

    public ResizeSettings Settings =>
        new(SelectedTargetSize, SelectedPadFillOption.Value, JpegQuality);

    public int IncludedCount => Images.Count(i => i.IsIncluded);

    public string SaveButtonText => IncludedCount == 1 ? "Save 1 image" : $"Save {IncludedCount} images";

    partial void OnSelectedTargetSizeChanged(int value)
    {
        foreach (var image in Images)
            image.UpdateTargetSize(value);
    }

    partial void OnSelectedModeOptionChanged(EnumOption<ResizeMode> value)
    {
        // Images the user has hand-placed keep their square; everything else follows the default.
        foreach (var image in Images.Where(i => !i.HasManualSelection))
            image.SetMode(value.Value);
    }

    [RelayCommand(CanExecute = nameof(CanInteract))]
    private async Task BrowseAsync()
    {
        var folder = await _dialogs.PickFolderAsync(FolderPath);
        if (string.IsNullOrEmpty(folder))
            return;

        FolderPath = folder;
        await LoadFolderAsync();
    }

    [RelayCommand(CanExecute = nameof(CanReload))]
    private Task ReloadAsync() => LoadFolderAsync();

    private bool CanInteract() => !IsBusy;

    private bool CanReload() => !IsBusy && HasFolder;

    private bool CanSave() => !IsBusy && HasFolder && Images.Any(i => i.IsIncluded);

    private async Task LoadFolderAsync()
    {
        if (string.IsNullOrEmpty(FolderPath))
            return;

        await CancelThumbnailsAsync();
        ClearImages();

        IsBusy = true;
        StatusText = "Scanning folder…";
        try
        {
            var folder = FolderPath;
            var found = await Task.Run(() => ImageScanner.Scan(folder));

            var mode = SelectedModeOption.Value;
            foreach (var source in found)
            {
                var item = new ImageItemViewModel(source, mode, SelectedTargetSize);
                item.PropertyChanged += OnItemPropertyChanged;
                Images.Add(item);
            }

            StatusText = found.Count switch
            {
                0 => "No supported images in that folder.",
                1 => "1 image loaded.",
                _ => $"{found.Count} images loaded.",
            };
        }
        catch (Exception ex)
        {
            StatusText = $"Could not read the folder: {ex.Message}";
        }
        finally
        {
            IsBusy = false;
            NotifySelectionCounts();
        }

        // Checkpoint the folder here as well as on close, so an unclean exit still remembers it.
        PersistSettings();

        _thumbnailCts = new CancellationTokenSource();
        var token = _thumbnailCts.Token;
        var items = Images.ToList();
        _ = ThumbnailLoader.LoadAllAsync(
            items,
            i => i.Source.Path,
            (i, bitmap) => i.Thumbnail = bitmap,
            token);
    }

    private void OnItemPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(ImageItemViewModel.IsIncluded))
            NotifySelectionCounts();
    }

    [RelayCommand(CanExecute = nameof(CanReload))]
    private void ApplyDefaultToAll()
    {
        var mode = SelectedModeOption.Value;
        foreach (var image in Images)
            image.ForceMode(mode);

        StatusText = $"Applied '{SelectedModeOption.Label}' to every image.";
    }

    [RelayCommand]
    private void SelectAll()
    {
        foreach (var image in Images)
            image.IsIncluded = true;
    }

    [RelayCommand]
    private void SelectNone()
    {
        foreach (var image in Images)
            image.IsIncluded = false;
    }

    [RelayCommand]
    private async Task EditAsync(ImageItemViewModel? item)
    {
        if (item is null || IsBusy)
            return;

        var preview = await Task.Run(() =>
            ThumbnailLoader.LoadOne(item.Source.Path, ThumbnailLoader.PreviewSize));

        var editor = new CropEditorViewModel(item.Source, preview, item.Selection, Settings);
        var accepted = await _dialogs.ShowCropEditorAsync(editor);

        if (accepted)
        {
            item.ApplyManualSelection(editor.Selection);
            StatusText = $"Updated the crop for {item.Source.FileName}.";
        }

        preview?.Dispose();
    }

    [RelayCommand(CanExecute = nameof(CanSave))]
    private async Task SaveAsync()
    {
        if (string.IsNullOrEmpty(FolderPath))
            return;

        var items = Images.Where(i => i.IsIncluded).ToList();
        if (items.Count == 0)
            return;

        foreach (var item in items)
            item.ClearStatus();

        var settings = Settings;
        _saveCts = new CancellationTokenSource();
        var token = _saveCts.Token;

        IsBusy = true;
        IsProgressVisible = true;
        ProgressValue = 0;
        StatusText = $"Saving {items.Count} images…";

        // Created here on the UI thread so Report marshals back to it for us.
        IProgress<double> progress = new Progress<double>(value => ProgressValue = value);

        var saved = 0;
        var failed = 0;
        var done = 0;

        try
        {
            var outputFolder = ImageSaver.EnsureOutputFolder(FolderPath);

            await Parallel.ForEachAsync(
                items,
                new ParallelOptions
                {
                    MaxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount - 1),
                    CancellationToken = token,
                },
                async (item, ct) =>
                {
                    try
                    {
                        ct.ThrowIfCancellationRequested();
                        ImageSaver.Save(item.Source, item.Selection, settings, outputFolder);
                        Interlocked.Increment(ref saved);
                        await SetStatusAsync(item, ItemStatus.Saved, null).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException)
                    {
                        throw;
                    }
                    catch (Exception ex)
                    {
                        // One unreadable file must not take the rest of the batch down with it.
                        Interlocked.Increment(ref failed);
                        await SetStatusAsync(item, ItemStatus.Failed, ex.Message).ConfigureAwait(false);
                    }
                    finally
                    {
                        var completed = Interlocked.Increment(ref done);
                        progress.Report(100.0 * completed / items.Count);
                    }
                });

            StatusText = failed == 0
                ? $"Saved {saved} images to {ImageScanner.OutputFolderName}."
                : $"Saved {saved} · Failed {failed}. See the badges on the failed cards.";
        }
        catch (OperationCanceledException)
        {
            StatusText = $"Cancelled after {saved} images.";
        }
        catch (Exception ex)
        {
            StatusText = $"Saving failed: {ex.Message}";
        }
        finally
        {
            _saveCts.Dispose();
            _saveCts = null;
            IsBusy = false;
            IsProgressVisible = false;
        }
    }

    [RelayCommand(CanExecute = nameof(IsBusy))]
    private void Cancel() => _saveCts?.Cancel();

    private static Task SetStatusAsync(ImageItemViewModel item, ItemStatus status, string? message) =>
        Avalonia.Threading.Dispatcher.UIThread.InvokeAsync(() =>
        {
            item.Status = status;
            item.StatusMessage = message;
        }).GetTask();

    private async Task CancelThumbnailsAsync()
    {
        if (_thumbnailCts is null)
            return;

        await _thumbnailCts.CancelAsync();
        _thumbnailCts.Dispose();
        _thumbnailCts = null;
    }

    private void ClearImages()
    {
        foreach (var image in Images)
        {
            image.PropertyChanged -= OnItemPropertyChanged;
            image.DisposeThumbnail();
        }

        Images.Clear();
        NotifySelectionCounts();
    }

    private void NotifySelectionCounts()
    {
        OnPropertyChanged(nameof(IncludedCount));
        OnPropertyChanged(nameof(SaveButtonText));
        SaveCommand.NotifyCanExecuteChanged();
    }
}
