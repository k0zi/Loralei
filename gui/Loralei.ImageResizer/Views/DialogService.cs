using System;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Platform.Storage;
using Loralei.ImageResizer.Services;
using Loralei.ImageResizer.ViewModels;

namespace Loralei.ImageResizer.Views;

/// <summary>
/// Window-owning implementation of <see cref="IDialogService"/>. The owner is looked up on each
/// call, because the host builds its view models before its window exists.
/// </summary>
public sealed class DialogService(Func<Window?> owner) : IDialogService
{
    public async Task<string?> PickFolderAsync(string? startAt)
    {
        var storage = owner()?.StorageProvider;
        if (storage is not { CanOpen: true })
            return null;

        IStorageFolder? start = null;
        if (!string.IsNullOrEmpty(startAt))
        {
            try
            {
                start = await storage.TryGetFolderFromPathAsync(startAt);
            }
            catch (Exception)
            {
                // A folder that has since been removed or renamed just means "no starting point".
            }
        }

        var folders = await storage.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Choose the folder containing the images",
            AllowMultiple = false,
            SuggestedStartLocation = start,
        });

        return folders.FirstOrDefault()?.TryGetLocalPath();
    }

    public async Task<bool> ShowCropEditorAsync(CropEditorViewModel viewModel)
    {
        if (owner() is not { } parent)
            return false;

        var window = new CropEditorWindow { DataContext = viewModel };
        return await window.ShowDialog<bool>(parent);
    }
}
