using CommunityToolkit.Mvvm.ComponentModel;
using Material.Icons;

namespace Loralei.Gui.ViewModels;

/// <summary>One entry in the left-hand navigation and the page it shows.</summary>
public abstract partial class PageViewModel(MainWindowViewModel shell) : ObservableObject
{
    public MainWindowViewModel Shell { get; } = shell;

    public abstract string Title { get; }

    /// <summary>The glyph for this page in the side menu.</summary>
    public abstract MaterialIconKind Icon { get; }

    /// <summary>One line under the page heading explaining what the page does.</summary>
    public abstract string Blurb { get; }

    /// <summary>
    /// True for a page that lays out its own height and scrolls its own content, so the shared
    /// chrome gives it the remaining space instead of wrapping it in the page scroller.
    /// </summary>
    public virtual bool FillsPage => false;

    /// <summary>
    /// Called when the page's view goes away - another page was chosen, or the window closed -
    /// for a page with something of its own to write down.
    /// </summary>
    public virtual void Persist() { }

    [ObservableProperty] private string? _error;
    [ObservableProperty] private string? _status;

    /// <summary>
    /// Called the first time the page is shown, so start-up stays fast. Returns false when
    /// the work could not run because something else held the command lock, which marks the
    /// page for another attempt rather than leaving it permanently blank.
    /// </summary>
    public virtual Task<bool> ActivateAsync() => Task.FromResult(true);

    /// <summary>Whether <see cref="ActivateAsync"/> has run and succeeded for this page.</summary>
    public bool Activated { get; internal set; }

    protected void Clear()
    {
        Error = null;
        Status = null;
    }

    /// <summary>
    /// Turns a finished command into page state. loralei exits 1 for handled errors and dumps
    /// a traceback for the rest, so the tail of stderr is the most useful thing to show.
    /// </summary>
    protected bool Report(Loralei.Core.LoraleiResult result, string okMessage)
    {
        if (result.Cancelled) { Status = "Cancelled."; return false; }
        if (result.Ok) { Status = okMessage; Error = null; return true; }

        var detail = result.StdErrTail.Count > 0
            ? string.Join('\n', result.StdErrTail.TakeLast(12))
            : "See the log below for details.";
        Error = $"loralei exited with code {result.ExitCode}.\n{detail}";
        Status = null;
        return false;
    }
}
