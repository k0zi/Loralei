using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Loralei.Core;

using Material.Icons;

namespace Loralei.Gui.ViewModels;

public sealed partial class SampleViewModel(MainWindowViewModel shell) : PageViewModel(shell)
{
    public override string Title => "Sample";
    public override MaterialIconKind Icon => MaterialIconKind.ImageSearchOutline;
    public override string Blurb => "Render images from a base model, optionally with a LoRA applied.";

    public string[] ModelOptions => LoraleiVocabulary.Models;

    public ObservableCollection<string> Images { get; } = [];

    [ObservableProperty] private string _prompts = "";
    [ObservableProperty] private string? _lora;
    [ObservableProperty] private string _model = Loralei.Core.Models.ModelRegistry.DefaultKey;
    [ObservableProperty] private string _outDir = "samples";
    [ObservableProperty] private int _steps = 30;
    [ObservableProperty] private double _cfg = 4.0;
    [ObservableProperty] private int _width = 1024;
    [ObservableProperty] private int _height = 1024;
    [ObservableProperty] private int _seed;
    [ObservableProperty] private int? _gpu;

    private List<string> PromptList() => Prompts
        .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
        .ToList();

    [RelayCommand]
    private async Task Render()
    {
        Clear();
        var prompts = PromptList();
        if (prompts.Count == 0)
        {
            Error = "Enter at least one prompt, one per line.";
            return;
        }
        if (!string.IsNullOrWhiteSpace(Lora) && !File.Exists(Lora))
        {
            Error = $"LoRA file not found: {Lora}";
            return;
        }

        await Shell.RunExclusiveAsync("sample", async ct =>
        {
            var args = CommandBuilder.Sample(prompts, Lora, Model, OutDir, Steps, Cfg,
                Width, Height, Seed, Gpu);
            var sink = Shell.Sink(Shell.Client.Preview(args));

            var result = await Shell.Client.SampleAsync(prompts, Lora, Model, OutDir, Steps, Cfg,
                Width, Height, Seed, Gpu, onOutput: sink, ct: ct);

            if (Report(result, $"Rendered {prompts.Count} image(s).")) LoadImages();
        });
    }

    private void LoadImages()
    {
        // --out is resolved against the child's working directory, which is the workspace.
        var dir = Path.IsPathRooted(OutDir) ? OutDir : Path.Combine(Shell.Paths.Workspace, OutDir);
        Images.Clear();
        if (!Directory.Exists(dir)) return;

        try
        {
            foreach (var file in new DirectoryInfo(dir).GetFiles("*.png")
                         .OrderByDescending(f => f.LastWriteTimeUtc).Take(24))
                Images.Add(file.FullName);
        }
        catch (IOException) { /* the gallery is a convenience */ }
    }

    /// <summary>Pre-fills from a finished run, which is the common way in to this page.</summary>
    public void UseAdapter(string path, string model)
    {
        Lora = path;
        Model = LoraleiVocabulary.Models.Contains(model) ? model : Model;
        Status = $"Using {Path.GetFileName(path)}.";
    }
}
