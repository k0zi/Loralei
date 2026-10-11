using Avalonia.Interactivity;
using KD.Avalonia.Rice.Controls;

namespace Loralei.ImageResizer.Views;

public partial class CropEditorWindow : RiceWindow
{
    public CropEditorWindow()
    {
        InitializeComponent();
    }

    private void OnAccept(object? sender, RoutedEventArgs e) => Close(true);

    private void OnCancel(object? sender, RoutedEventArgs e) => Close(false);
}
