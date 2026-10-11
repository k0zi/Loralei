using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Loralei.Core;
using Loralei.Core.Dto;

using Material.Icons;

namespace Loralei.Gui.ViewModels;

public sealed partial class RunsViewModel(MainWindowViewModel shell) : PageViewModel(shell)
{
    public override string Title => "Runs";
    public override MaterialIconKind Icon => MaterialIconKind.FormatListBulletedSquare;
    public override string Blurb => "Every run under the output root, and what it recorded.";

    public ObservableCollection<RunEntry> Entries { get; } = [];
    public ObservableCollection<KeyValuePair<string, string>> Manifest { get; } = [];
    public ObservableCollection<string> Files { get; } = [];

    [ObservableProperty] private RunEntry? _selected;
    [ObservableProperty] private bool _confirmingDelete;
    [ObservableProperty] private string? _resolvedConfig;

    public override Task<bool> ActivateAsync()
    {
        Refresh();
        return Task.FromResult(true);
    }

    /// <summary>
    /// Read straight from the run directories rather than by scraping `runs ls`: the same
    /// manifest.json the CLI reads, so a run started from a terminal shows up here too.
    /// </summary>
    [RelayCommand]
    private void Refresh()
    {
        var previous = Selected?.Name;
        Entries.Clear();
        foreach (var entry in Shell.Runs.List(Shell.OutputRoot)) Entries.Add(entry);

        Selected = Entries.FirstOrDefault(e => e.Name == previous) ?? Entries.FirstOrDefault();
        Status = Entries.Count == 0 ? $"No runs under {Shell.Runs.Root(Shell.OutputRoot)}." : null;
    }

    partial void OnSelectedChanged(RunEntry? value)
    {
        ConfirmingDelete = false;
        Manifest.Clear();
        Files.Clear();
        ResolvedConfig = null;
        if (value is null) return;

        var manifest = Shell.Runs.ReadManifest(value.Name, Shell.OutputRoot);
        if (manifest is not null)
            foreach (var field in manifest.AllFields()) Manifest.Add(field);

        foreach (var file in Shell.Runs.Adapters(value.Name, Shell.OutputRoot))
            Files.Add($"{file.Name}  ({file.Length / (1024.0 * 1024.0):F1} MiB)");

        var checkpoints = Shell.Runs.ResumableCheckpoints(value.Name, Shell.OutputRoot).Count;
        if (checkpoints > 0) Files.Add($"{checkpoints} resumable checkpoint(s)");

        ResolvedConfig = Shell.Runs.ReadResolvedConfig(value.Name, Shell.OutputRoot);
    }

    [RelayCommand]
    private void LoadIntoTrainForm()
    {
        if (ResolvedConfig is not { } yaml)
        {
            Error = "This run has no config.resolved.yaml to load.";
            return;
        }

        var train = Shell.Pages.OfType<TrainViewModel>().First();
        train.LoadFrom(TrainRequest.FromResolved(yaml));
        Shell.SelectedPage = train;
    }

    [RelayCommand]
    private void UseForSampling()
    {
        if (Selected is not { } run) return;
        var adapter = Shell.Runs.Adapters(run.Name, Shell.OutputRoot)
            .FirstOrDefault(f => !f.Name.Contains("_comfy"));
        if (adapter is null)
        {
            Error = "This run has no adapter yet.";
            return;
        }

        var sample = Shell.Pages.OfType<SampleViewModel>().First();
        sample.UseAdapter(adapter.FullName, run.Model);
        Shell.SelectedPage = sample;
    }

    [RelayCommand]
    private void AskDelete() => ConfirmingDelete = Selected is not null;

    [RelayCommand]
    private void CancelDelete() => ConfirmingDelete = false;

    /// <summary>
    /// Confirmation happens here, then `runs rm --yes` runs non-interactively - the CLI's
    /// own prompt would be answered "no" by this GUI's closed stdin.
    /// </summary>
    [RelayCommand]
    private async Task ConfirmDelete()
    {
        ConfirmingDelete = false;
        if (Selected is not { } run) return;
        Clear();

        await Shell.RunExclusiveAsync($"runs rm {run.Name}", async ct =>
        {
            var sink = Shell.Sink(Shell.Client.Preview(CommandBuilder.RunsRm(run.Name)));
            var result = await Shell.Client.RunsRmAsync(run.Name, onOutput: sink, ct: ct);
            Report(result, $"Deleted {run.Name}.");
        });

        Refresh();
    }
}
