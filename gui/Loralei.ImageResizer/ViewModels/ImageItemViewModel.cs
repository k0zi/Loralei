using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using Loralei.ImageResizer.Models;
using Loralei.ImageResizer.Services;

namespace Loralei.ImageResizer.ViewModels;

/// <summary>One card in the grid: the source file plus the square that will be exported from it.</summary>
public partial class ImageItemViewModel : ViewModelBase
{
    private int _targetSize;

    public ImageItemViewModel(SourceImage source, ResizeMode mode, int targetSize)
    {
        Source = source;
        _targetSize = targetSize;
        Mode = mode;
        Selection = ResizeGeometry.ForMode(mode, source.PixelWidth, source.PixelHeight);
    }

    public SourceImage Source { get; }

    /// <summary>True once the user has positioned the square by hand, which protects it from "apply to all".</summary>
    public bool HasManualSelection { get; private set; }

    [ObservableProperty]
    public partial Bitmap? Thumbnail { get; set; }

    [ObservableProperty]
    public partial bool IsIncluded { get; set; } = true;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsCrop), nameof(IsPad), nameof(IsManual))]
    public partial ResizeMode Mode { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsUpscaling), nameof(SelectionText))]
    public partial SquareSelection Selection { get; set; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasStatusMessage), nameof(IsSaved), nameof(IsFailed))]
    public partial ItemStatus Status { get; set; } = ItemStatus.Pending;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasStatusMessage))]
    public partial string? StatusMessage { get; set; }

    public bool HasStatusMessage => !string.IsNullOrEmpty(StatusMessage);

    public bool IsSaved => Status == ItemStatus.Saved;

    public bool IsFailed => Status == ItemStatus.Failed;

    /// <summary>True when the selected region has fewer pixels than the output, so the result is interpolated.</summary>
    public bool IsUpscaling => ResizeGeometry.IsUpscaling(Selection, _targetSize);

    public string SelectionText => $"{(int)Selection.Size} px → {_targetSize} px";

    // The three mode buttons bind to these rather than to grouped radio buttons, which keeps the
    // grouping explicit instead of depending on the visual tree.
    public bool IsCrop
    {
        get => Mode == ResizeMode.Crop;
        set
        {
            if (value)
                SetMode(ResizeMode.Crop);
            else
                OnPropertyChanged(); // clicking the active toggle must not turn it off
        }
    }

    public bool IsPad
    {
        get => Mode == ResizeMode.Pad;
        set
        {
            if (value)
                SetMode(ResizeMode.Pad);
            else
                OnPropertyChanged();
        }
    }

    public bool IsManual
    {
        get => Mode == ResizeMode.Manual;
        set
        {
            if (value)
                SetMode(ResizeMode.Manual);
            else
                OnPropertyChanged();
        }
    }

    /// <summary>Switches mode and recomputes the automatic square. Picking "Manual" keeps the current square as a starting point.</summary>
    public void SetMode(ResizeMode mode)
    {
        if (Mode == mode)
            return;

        Mode = mode;
        if (mode == ResizeMode.Manual)
        {
            HasManualSelection = true;
        }
        else
        {
            HasManualSelection = false;
            Selection = ResizeGeometry.ForMode(mode, Source.PixelWidth, Source.PixelHeight);
        }

        ClearStatus();
    }

    /// <summary>Resets the item to <paramref name="mode"/> even if it was hand-placed. Used by "Apply to all".</summary>
    public void ForceMode(ResizeMode mode)
    {
        Mode = mode;
        HasManualSelection = mode == ResizeMode.Manual;
        if (mode != ResizeMode.Manual)
            Selection = ResizeGeometry.ForMode(mode, Source.PixelWidth, Source.PixelHeight);

        ClearStatus();
    }

    /// <summary>Applies a square chosen in the crop editor.</summary>
    public void ApplyManualSelection(SquareSelection selection)
    {
        Selection = ResizeGeometry.Sanitize(selection, Source.PixelWidth, Source.PixelHeight);
        HasManualSelection = true;
        Mode = ResizeMode.Manual;
        ClearStatus();
    }

    /// <summary>Called when the target size changes - the upscale warning and readout depend on it.</summary>
    public void UpdateTargetSize(int targetSize)
    {
        _targetSize = targetSize;
        OnPropertyChanged(nameof(IsUpscaling));
        OnPropertyChanged(nameof(SelectionText));
    }

    public void ClearStatus()
    {
        Status = ItemStatus.Pending;
        StatusMessage = null;
    }

    public void DisposeThumbnail()
    {
        Thumbnail?.Dispose();
        Thumbnail = null;
    }
}
