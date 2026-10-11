using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Loralei.Core;

using Material.Icons;

namespace Loralei.Gui.ViewModels;

public sealed partial class DataViewModel(MainWindowViewModel shell) : PageViewModel(shell)
{
    public override string Title => "Data";
    public override MaterialIconKind Icon => MaterialIconKind.ImageMultipleOutline;
    public override string Blurb => "Pair images with captions and preview the aspect-ratio buckets.";

    public ObservableCollection<BucketRow> Buckets { get; } = [];
    public ObservableCollection<string> Problems { get; } = [];

    [ObservableProperty] private string _dataset = "";
    [ObservableProperty] private int _resolution = 1024;
    [ObservableProperty] private string _captionExtension = ".txt";
    [ObservableProperty] private int _pairs;
    [ObservableProperty] private string _captions = "";

    public override Task<bool> ActivateAsync()
    {
        Dataset = Shell.Settings.LastDataset ?? "";
        return Task.FromResult(true);
    }

    [RelayCommand]
    private async Task Check()
    {
        Clear();
        if (!Directory.Exists(Dataset))
        {
            Error = "Pick a dataset folder first.";
            return;
        }

        Shell.Settings.LastDataset = Dataset;
        Shell.Settings.Save();

        await Shell.RunExclusiveAsync("data check", async ct =>
        {
            // The scan hashes every image, and its JSON arrives only when it is done - say
            // so rather than looking hung.
            Status = "Scanning (this reads every image)...";

            var sink = Shell.Sink(Shell.Client.Preview(
                CommandBuilder.DataCheck(Dataset, Resolution, CaptionExtension)));
            var check = await Shell.Client.DataCheckAsync(Dataset, Resolution, CaptionExtension, sink, ct);

            Buckets.Clear();
            foreach (var bucket in check.Buckets) Buckets.Add(bucket);

            Problems.Clear();
            foreach (var problem in check.Problems) Problems.Add(problem);

            Pairs = check.Pairs;
            Captions = check.Captions;

            if (!check.Ok) Report(check.Process, "");
            else Status = $"{check.Pairs} usable pair(s) across {check.Buckets.Count} bucket(s).";
        });
    }

    /// <summary>Hands the dataset to the pages that consume it, so it is typed once.</summary>
    [RelayCommand]
    private void UseForTraining()
    {
        foreach (var page in Shell.Pages)
        {
            if (page is TrainViewModel train) train.Dataset = Dataset;
            if (page is CacheViewModel cache) cache.Dataset = Dataset;
        }
        Status = "Dataset copied to the Cache and Train pages.";
    }
}
