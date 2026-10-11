using System.Threading.Tasks;
using Loralei.ImageResizer.ViewModels;

namespace Loralei.ImageResizer.Services;

/// <summary>The window-level interactions the main view model needs, kept behind an interface so it stays testable.</summary>
public interface IDialogService
{
    Task<string?> PickFolderAsync(string? startAt);

    /// <summary>Shows the crop editor modally. Returns true when the user accepted the new square.</summary>
    Task<bool> ShowCropEditorAsync(CropEditorViewModel viewModel);
}
