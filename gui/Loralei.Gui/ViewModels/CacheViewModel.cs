using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Loralei.Core;

using Material.Icons;

namespace Loralei.Gui.ViewModels;

public sealed partial class CacheViewModel(MainWindowViewModel shell) : PageViewModel(shell)
{
    public override string Title => "Cache";
    public override MaterialIconKind Icon => MaterialIconKind.DatabaseArrowDownOutline;
    public override string Blurb =>
        "Pre-compute VAE latents and text embeddings. Training does this automatically; " +
        "run it here to pay the cost up front.";

    public string[] ModelOptions => LoraleiVocabulary.Models;
    public string[] VaeDeviceOptions => LoraleiVocabulary.VaeDevices;

    [ObservableProperty] private string _dataset = "";
    [ObservableProperty] private string _model = Loralei.Core.Models.ModelRegistry.DefaultKey;
    [ObservableProperty] private int? _resolution;
    [ObservableProperty] private int? _gpu;
    [ObservableProperty] private string _vaeDevice = "cpu";
    [ObservableProperty] private bool _force;

    public override Task<bool> ActivateAsync()
    {
        if (Dataset.Length == 0) Dataset = Shell.Settings.LastDataset ?? "";
        return Task.FromResult(true);
    }

    [RelayCommand]
    private async Task Run()
    {
        Clear();
        if (!Directory.Exists(Dataset))
        {
            Error = "Pick a dataset folder first.";
            return;
        }

        await Shell.RunExclusiveAsync("cache", async ct =>
        {
            var args = CommandBuilder.Cache(Dataset, Model, Resolution, Gpu, VaeDevice, force: Force);
            var sink = Shell.Sink(Shell.Client.Preview(args));
            var result = await Shell.Client.CacheAsync(Dataset, Model, Resolution, Gpu, VaeDevice,
                force: Force, onOutput: sink, ct: ct);
            Report(result, "Latents and text embeddings are cached.");
        });
    }
}
