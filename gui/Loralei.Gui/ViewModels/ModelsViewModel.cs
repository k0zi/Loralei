using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Loralei.Core;

using Material.Icons;

namespace Loralei.Gui.ViewModels;

/// <summary>One row of the model table, with the tick that puts it in the next download.</summary>
public sealed partial class ModelChoice(ModelRow row) : ObservableObject
{
    public ModelRow Row { get; } = row;

    [ObservableProperty] private bool _isChecked;
}

public sealed partial class ModelsViewModel(MainWindowViewModel shell) : PageViewModel(shell)
{
    public override string Title => "Models";
    public override MaterialIconKind Icon => MaterialIconKind.CubeOutline;
    public override string Blurb => "Check which base weights are on this machine, tick what you want, and fetch it.";

    public ObservableCollection<ModelChoice> Models { get; } = [];

    [ObservableProperty] private string _freeSpace = "";
    [ObservableProperty] private string _roots = "";
    [ObservableProperty] private string _rawListing = "";
    [ObservableProperty] private double _minGib = 1.0;
    [ObservableProperty] private string _reclaimable = "";

    public override Task<bool> ActivateAsync() => Refresh();

    [RelayCommand]
    private async Task<bool> Refresh()
    {
        Clear();
        return await Shell.RunExclusiveAsync("models ls", async ct =>
        {
            var sink = Shell.Sink(Shell.Client.Preview(CommandBuilder.ModelsLs()));
            var listing = await Shell.Client.ModelsLsAsync(onOutput: sink, ct: ct);

            // A refresh after a download must not forget what else was ticked.
            var ticked = Models.Where(m => m.IsChecked).Select(m => m.Row.Key).ToHashSet();
            Models.Clear();
            foreach (var model in listing.Models)
                Models.Add(new ModelChoice(model) { IsChecked = !model.Ready && ticked.Contains(model.Key) });

            FreeSpace = listing.FreeSpace;
            Roots = string.Join('\n', listing.Roots);
            RawListing = listing.RawText;

            if (listing.Models.Count == 0)
                Error = "Could not read the model table; the raw output is below.";
            else
                Status = $"{listing.Models.Count(m => m.Ready)} of {listing.Models.Count} models ready.";
        });
    }

    /// <summary>
    /// Downloads every ticked model that is not complete yet, one after another in a single
    /// command. Space is checked first: `models pull` refuses rather than filling the
    /// filesystem, and finding that out before a multi-gigabyte download starts is worth one
    /// extra call.
    /// </summary>
    [RelayCommand]
    private async Task Pull()
    {
        var wanted = Models.Where(m => m.IsChecked && !m.Row.Ready).Select(m => m.Row.Key).ToList();
        Clear();
        if (wanted.Count == 0)
        {
            Error = "Tick at least one model that is not ready yet.";
            return;
        }

        var label = string.Join(", ", wanted);
        await Shell.RunExclusiveAsync($"models pull {label}", async ct =>
        {
            var check = await Shell.Client.ModelsPullAsync(wanted, dryRun: true,
                onOutput: Shell.Sink(Shell.Client.Preview(
                    CommandBuilder.ModelsPull(wanted, dryRun: true))), ct: ct);
            if (!Report(check, "")) return;

            var result = await Shell.Client.ModelsPullAsync(wanted,
                onOutput: Shell.Sink(Shell.Client.Preview(CommandBuilder.ModelsPull(wanted))), ct: ct);
            Report(result, wanted.Count == 1 ? $"{label} is ready." : $"Downloaded: {label}.");
        });

        await Refresh();
    }

    [RelayCommand]
    private async Task ListReclaimable()
    {
        Clear();
        await Shell.RunExclusiveAsync("models gc", async ct =>
        {
            var sink = Shell.Sink(Shell.Client.Preview(CommandBuilder.ModelsGc(MinGib)));
            var result = await Shell.Client.ModelsGcAsync(MinGib, sink, ct);
            Reclaimable = result.RawText;
            // Deleting asks for a confirmation per entry on stdin, which this GUI deliberately closes.
            Status = "Listed only. Delete these from a terminal with `loralei models gc --no-dry-run`.";
        });
    }
}
