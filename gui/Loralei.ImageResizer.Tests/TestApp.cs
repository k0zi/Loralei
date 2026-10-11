using Avalonia;
using Avalonia.Headless;
using Loralei.ImageResizer.Tests;
using KD.Avalonia.Rice;

[assembly: AvaloniaTestApplication(typeof(TestApp))]
// One application for the whole run: KD.Avalonia.Rice's ThemeManager is a process-wide singleton whose
// brush dictionary can only be attached to a single Application.
[assembly: AvaloniaTestIsolation(AvaloniaTestIsolationLevel.PerAssembly)]

namespace Loralei.ImageResizer.Tests;

/// <summary>
/// The library has no Application of its own any more - Loralei.Gui hosts it - so the tests bring
/// the least that the views need: Rice's styles, which carry FluentTheme and the Rice brushes.
/// </summary>
public sealed class TestApp : Application
{
    public override void Initialize() => Styles.Add(new RiceStyles());

    // Real Skia drawing rather than the no-op headless renderer, so Render() actually runs.
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<TestApp>()
        .UseSkia()
        .UseHeadless(new AvaloniaHeadlessPlatformOptions { UseHeadlessDrawing = false });
}
