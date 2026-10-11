using System.Collections.ObjectModel;
using Avalonia.Controls;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Loralei.Core;
using Loralei.Gui.Theming;

namespace Loralei.Gui.ViewModels;

public sealed partial class MainWindowViewModel : ObservableObject
{
    public GuiSettings Settings { get; }
    public LoraleiPaths Paths { get; private set; }
    public LoraleiClient Client { get; private set; }
    public RunStore Runs { get; private set; }
    public LogViewModel Log { get; } = new();
    public ThemeService Theme { get; }

    public ThemePreference[] ThemeOptions { get; } =
        [ThemePreference.System, ThemePreference.Light, ThemePreference.Dark];

    /// <summary>Bound to the picker in the settings window; the service does the applying and saving.</summary>
    public ThemePreference SelectedTheme
    {
        get => Theme.Preference;
        set
        {
            if (Theme.Preference == value) return;
            Theme.Preference = value;
            OnPropertyChanged();
        }
    }

    public ObservableCollection<PageViewModel> Pages { get; } = [];

    [ObservableProperty] private PageViewModel? _selectedPage;

    /// <summary>What is running right now, or null when idle.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsBusy))]
    private string? _busyWith;

    [ObservableProperty] private string? _setupProblem;

    public bool IsBusy => BusyWith is not null;

    private CancellationTokenSource? _cancellation;

    public MainWindowViewModel()
    {
        Settings = GuiSettings.Load();
        Theme = new ThemeService(Settings);
        Paths = LoraleiPaths.Discover(Settings.Executable, Settings.Workspace);
        Client = new LoraleiClient(Paths);
        Runs = new RunStore(Paths);
        SetupProblem = Paths.Problem();

        Pages.Add(new DoctorViewModel(this));
        Pages.Add(new ModelsViewModel(this));
        Pages.Add(new ResizeViewModel(this));
        Pages.Add(new CaptionViewModel(this));
        Pages.Add(new DataViewModel(this));
        Pages.Add(new CacheViewModel(this));
        Pages.Add(new TrainViewModel(this));
        Pages.Add(new SampleViewModel(this));
        Pages.Add(new ExportViewModel(this));
        Pages.Add(new RunsViewModel(this));

        SelectedPage = Pages[0];
        Log.AppendNote($"loralei: {Paths.Executable}");
        Log.AppendNote($"workspace: {Paths.Workspace}");
        if (SetupProblem is { } problem) Log.AppendNote($"! {problem}");
    }

    /// <summary>
    /// Where runs actually land. <c>train</c> has no <c>--output-root</c> flag, so the
    /// workspace loralei.yaml is the only thing that can move it - which makes that file, not
    /// the GUI's own preferences, the authority every page has to read.
    /// </summary>
    public string OutputRoot
    {
        get
        {
            try
            {
                if (File.Exists(Paths.ProjectConfig) &&
                    FlatYaml.Parse(File.ReadAllText(Paths.ProjectConfig)).Scalar("output_root")
                        is { Length: > 0 } configured && configured != "null")
                    return configured;
            }
            catch (IOException) { /* fall through to the default */ }
            return "runs";
        }
    }

    /// <summary>Honours <c>--page &lt;title&gt;</c>, which opens straight onto one page.</summary>
    public void SelectStartupPage(IReadOnlyList<string>? args)
    {
        if (args is null) return;
        var flag = args.ToList().IndexOf("--page");
        if (flag < 0 || flag + 1 >= args.Count) return;

        var wanted = args[flag + 1];
        var page = Pages.FirstOrDefault(p =>
            p.Title.Equals(wanted, StringComparison.OrdinalIgnoreCase));
        if (page is not null) SelectedPage = page;
    }

    partial void OnSelectedPageChanged(PageViewModel? value) => _ = ActivateAsync(value);

    private async Task ActivateAsync(PageViewModel? page)
    {
        if (page is null || page.Activated) return;
        page.Activated = true;
        if (!await page.ActivateAsync()) page.Activated = false;
    }

    /// <summary>
    /// Runs one loralei command at a time. There is a single box with a fixed number of
    /// cards behind all of this, so two training runs - or a run and a sample render on the
    /// same card - would simply fight over VRAM. Serialising is the honest behaviour.
    /// </summary>
    public async Task<bool> RunExclusiveAsync(string what, Func<CancellationToken, Task> work)
    {
        if (IsBusy)
        {
            Log.AppendNote($"! busy with {BusyWith}; {what} was not started");
            return false;
        }

        _cancellation = new CancellationTokenSource();
        BusyWith = what;
        try
        {
            await work(_cancellation.Token);
            return true;
        }
        finally
        {
            BusyWith = null;
            _cancellation.Dispose();
            _cancellation = null;
            // A page shown while this was running had its own start-up work refused.
            await ActivateAsync(SelectedPage);
        }
    }

    [RelayCommand]
    public void Cancel()
    {
        if (_cancellation is null) return;
        Log.AppendNote("! interrupting (SIGINT)");
        _cancellation.Cancel();
    }

    /// <summary>Runs a command and mirrors its output into the shared log pane.</summary>
    public Action<ConsoleLine> Sink(string preview)
    {
        Log.AppendCommand(preview);
        return Log.Append;
    }

    public void PersistSettings(Window window)
    {
        Settings.Executable = Paths.Executable;
        Settings.Workspace = Paths.Workspace;
        Settings.WindowWidth = window.Width;
        Settings.WindowHeight = window.Height;
        Settings.Save();
    }

    /// <summary>Re-points the GUI at a different CLI or workspace, rebuilding the services.</summary>
    public void Rebind(string? executable, string? workspace)
    {
        Paths = LoraleiPaths.Discover(executable, workspace);
        Client = new LoraleiClient(Paths);
        Runs = new RunStore(Paths);
        SetupProblem = Paths.Problem();
        Settings.Executable = Paths.Executable;
        Settings.Workspace = Paths.Workspace;
        Settings.Save();

        Log.AppendNote($"loralei: {Paths.Executable}");
        Log.AppendNote($"workspace: {Paths.Workspace}");
        foreach (var page in Pages) page.Activated = false;
        if (SelectedPage is { } selected)
        {
            selected.Activated = true;
            _ = selected.ActivateAsync();
        }
    }
}
