using Avalonia;
using Avalonia.Headless;
using Loralei.Gui;

[assembly: AvaloniaTestApplication(typeof(Loralei.Gui.Tests.TestApp))]
// One Application for the whole run: Rice's ThemeManager is a process-wide singleton whose
// brush dictionary can only ever be parented by a single RiceStyles, so a fresh App per
// test fails with "The ResourceDictionary already has a parent".
[assembly: AvaloniaTestIsolation(AvaloniaTestIsolationLevel.PerAssembly)]

namespace Loralei.Gui.Tests;

/// <summary>Boots the real App on the headless platform, so the tests drive production code.</summary>
public static class TestApp
{
    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>().UseHeadless(new AvaloniaHeadlessPlatformOptions());
}
