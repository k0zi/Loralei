using Avalonia;
using KD.Avalonia.Rice;
using KD.Avalonia.Rice.Options;
using KD.Avalonia.Rice.Theming;

namespace Loralei.Gui;

internal static class Program
{
    [STAThread]
    public static void Main(string[] args) => BuildAvaloniaApp()
        .StartWithClassicDesktopLifetime(args);

    // Also used by the XAML designer; do not inline.
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>()
        .UsePlatformDetect()
        .WithInterFont()
        .LogToTrace()
        .UseRice(options =>
        {
            // The AppId is the folder the chosen colour theme is saved under, so rice.json
            // lands next to our own settings.json in ~/.config/loralei-gui.
            options.AppInfo = new RiceAppInfo
            {
                AppId = "loralei-gui",
                Name = "Loralei",
                Description = "A desktop front end for the loralei LoRA trainer.",
                IconUri = "avares://Loralei.Gui/Assets/loralei.png",
            };
            options.DefaultThemeId = BuiltInThemes.Nord.Id;
        });
}
