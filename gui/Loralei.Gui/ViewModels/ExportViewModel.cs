using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Loralei.Core;

using Material.Icons;

namespace Loralei.Gui.ViewModels;

public sealed partial class ExportViewModel(MainWindowViewModel shell) : PageViewModel(shell)
{
    public override string Title => "Export";
    public override MaterialIconKind Icon => MaterialIconKind.FileExportOutline;
    public override string Blurb =>
        "Write the LoRA in diffusers and ComfyUI formats, and prove both load.";

    public string[] FormatOptions => LoraleiVocabulary.ExportFormats;

    public ObservableCollection<string> Runs { get; } = [];
    public ObservableCollection<string> Produced { get; } = [];

    [ObservableProperty] private string? _run;
    [ObservableProperty] private string _format = "both";
    [ObservableProperty] private bool _verify = true;

    public override Task<bool> ActivateAsync()
    {
        Refresh();
        return Task.FromResult(true);
    }

    [RelayCommand]
    private void Refresh()
    {
        var names = Shell.Runs.List(Shell.OutputRoot).Select(r => r.Name).ToList();
        Runs.Clear();
        foreach (var name in names) Runs.Add(name);
        Run ??= names.FirstOrDefault();
    }

    [RelayCommand]
    private async Task Export()
    {
        Clear();
        if (Run is not { Length: > 0 } run)
        {
            Error = "Pick a run to export.";
            return;
        }

        await Shell.RunExclusiveAsync("export", async ct =>
        {
            if (Verify)
            {
                // The check loads the whole base transformer on the CPU in bf16.
                Status = "Exporting, then loading the base transformer to verify the adapter attaches...";
            }

            var args = CommandBuilder.Export(run, Format, verify: Verify);
            var sink = Shell.Sink(Shell.Client.Preview(args));
            var result = await Shell.Client.ExportAsync(run, Format, verify: Verify, onOutput: sink, ct: ct);

            if (Report(result, "Export complete."))
            {
                Produced.Clear();
                foreach (var file in Shell.Runs.Adapters(run, Shell.OutputRoot))
                    Produced.Add($"{file.Name}  ({file.Length / (1024.0 * 1024.0):F1} MiB)");
            }
        });
    }
}
