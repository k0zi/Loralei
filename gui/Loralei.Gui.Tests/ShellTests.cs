using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.VisualTree;
using KD.Avalonia.Rice.Controls;
using Loralei.Gui.Theming;
using Loralei.Gui.ViewModels;
using Loralei.Gui.Views;
using Loralei.Tests;
using Material.Icons;
using Xunit;

namespace Loralei.Gui.Tests;

/// <summary>
/// The window's own wiring. The side menu is bound to view models, and the Application has
/// a ViewLocator that turns any page view model into a whole page - so a menu entry that
/// fell back to default content would render the page inside the menu.
/// </summary>
public class ShellTests
{
    private static readonly Loralei.Core.LoraleiPaths Real = Loralei.Core.LoraleiPaths.Discover(null, null);

    [AvaloniaFact]
    public void TheSideMenuShowsTitlesAndIconsAndDrivesThePage()
    {
        using var workspace = TempWorkspace.Create(Real);
        var shell = new MainWindowViewModel();
        var window = new MainWindow { DataContext = shell };
        window.Show();

        try
        {
            var menu = window.GetVisualDescendants().OfType<RiceSideMenu>().Single();
            var entries = menu.GetVisualDescendants().OfType<RiceSideMenuItem>().ToList();

            Assert.Equal(shell.Pages.Select(p => (object)p.Title), entries.Select(e => e.Content));
            Assert.Equal(shell.Pages.Select(p => p.Icon), entries.Select(e => (MaterialIconKind)e.Icon!));
            Assert.Same(shell.Pages[0], menu.SelectedItem);

            menu.SelectedIndex = 2;
            Assert.Same(shell.Pages[2], shell.SelectedPage);

            // The title bar carries Rice's own buttons, not pickers of ours.
            var titleBar = window.GetVisualDescendants().OfType<RiceTitleBar>().Single();
            Assert.Empty(titleBar.GetVisualDescendants().OfType<ComboBox>());
            Assert.True(window.ShowThemeToggle);
            Assert.True(window.ShowSettingsButton);
            Assert.True(window.ShowAboutButton);
        }
        finally { window.Close(); }
    }

    [AvaloniaFact]
    public void TheSettingsWindowRemembersTheStartUpVariantAndOffersTheColourThemes()
    {
        using var workspace = TempWorkspace.Create(Real);
        var shell = new MainWindowViewModel();
        var settings = new SettingsWindow { DataContext = shell };
        settings.Show();

        try
        {
            Assert.Single(settings.GetVisualDescendants().OfType<ThemeSelector>());

            var picker = settings.GetVisualDescendants().OfType<ComboBox>().Single();
            picker.SelectedItem = ThemePreference.Light;

            Assert.Equal(ThemePreference.Light, shell.SelectedTheme);
            Assert.Equal("light", Loralei.Core.GuiSettings.Load(workspace.SettingsPath).Theme);
        }
        finally { settings.Close(); }
    }
}
