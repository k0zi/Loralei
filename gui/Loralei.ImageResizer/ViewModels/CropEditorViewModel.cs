using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Loralei.ImageResizer.Models;
using Loralei.ImageResizer.Services;

namespace Loralei.ImageResizer.ViewModels;

/// <summary>Backs the manual crop window: one image, one square, and the presets to move it around.</summary>
public partial class CropEditorViewModel : ViewModelBase
{
    private readonly SquareSelection _original;

    public CropEditorViewModel(SourceImage source, Bitmap? preview, SquareSelection selection, ResizeSettings settings)
    {
        Source = source;
        Preview = preview;
        Settings = settings;
        _original = selection;
        Selection = selection;
    }

    public SourceImage Source { get; }

    public Bitmap? Preview { get; }

    public ResizeSettings Settings { get; }

    public string Title => $"Crop — {Source.FileName}";

    /// <summary>Set by the window when the user confirms; the caller reads it to decide whether to apply.</summary>
    public bool Accepted { get; private set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ReadoutText), nameof(IsUpscaling))]
    public partial SquareSelection Selection { get; set; }

    public bool IsUpscaling => ResizeGeometry.IsUpscaling(Selection, Settings.TargetSize);

    public string ReadoutText
    {
        get
        {
            var size = (int)System.Math.Round(Selection.Size);
            var scale = ResizeGeometry.ScaleFactor(Selection, Settings.TargetSize);
            var direction = scale > 1 ? "upscale" : "downscale";
            return $"Output {Settings.TargetSize} × {Settings.TargetSize}  —  source region {size} × {size}  ({scale:0.00}× {direction})";
        }
    }

    [RelayCommand]
    private void Fill() => Selection = ResizeGeometry.Cover(Source.PixelWidth, Source.PixelHeight);

    [RelayCommand]
    private void Fit() => Selection = ResizeGeometry.Contain(Source.PixelWidth, Source.PixelHeight);

    [RelayCommand]
    private void Center() => Selection = ResizeGeometry.Centered(Selection, Source.PixelWidth, Source.PixelHeight);

    [RelayCommand]
    private void Reset() => Selection = _original;

    [RelayCommand]
    private void Accept() => Accepted = true;
}
