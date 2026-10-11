using Avalonia;
using Avalonia.Styling;
using Loralei.Core;

namespace Loralei.Gui.Theming;

/// <summary>What the user asked for, which is not the same as the variant finally applied.</summary>
public enum ThemePreference
{
    System,
    Light,
    Dark,
}

/// <summary>
/// Owns the light/dark variant the application starts in.
/// <para>
/// Rice's title-bar toggle only flips the variant for the session, so the System / Light /
/// Dark choice made in the settings window is written to <c>settings.json</c> here and
/// applied at start-up. "System" is simply <see cref="ThemeVariant.Default"/>, which
/// Avalonia follows live.
/// </para>
/// <para>
/// The colour theme is not ours at all: Rice's <c>ThemeSelector</c> applies it and saves it
/// to its own <c>rice.json</c>.
/// </para>
/// </summary>
public sealed class ThemeService
{
    private readonly GuiSettings _settings;

    public ThemeService(GuiSettings settings)
    {
        _settings = settings;
        _preference = ParsePreference(settings.Theme);
    }

    private ThemePreference _preference;

    public ThemePreference Preference
    {
        get => _preference;
        set
        {
            if (_preference == value) return;
            _preference = value;
            _settings.Theme = value.ToString().ToLowerInvariant();
            _settings.Save();
            Apply();
        }
    }

    /// <summary>The variant to request; <see cref="ThemeVariant.Default"/> follows the desktop.</summary>
    public ThemeVariant Requested => _preference switch
    {
        ThemePreference.Light => ThemeVariant.Light,
        ThemePreference.Dark => ThemeVariant.Dark,
        _ => ThemeVariant.Default,
    };

    /// <summary>Call once the Avalonia application exists, before the main window is shown.</summary>
    public void Initialise() => Apply();

    private void Apply()
    {
        if (Application.Current is { } app) app.RequestedThemeVariant = Requested;
    }

    public static ThemePreference ParsePreference(string? value) => value?.ToLowerInvariant() switch
    {
        "light" => ThemePreference.Light,
        "dark" => ThemePreference.Dark,
        _ => ThemePreference.System,
    };
}
