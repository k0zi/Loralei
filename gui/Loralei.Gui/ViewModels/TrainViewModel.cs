using System.Collections.ObjectModel;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Loralei.Core;
using Loralei.Core.Dto;

using Material.Icons;

namespace Loralei.Gui.ViewModels;

public sealed partial class TrainViewModel : PageViewModel
{
    private readonly TrainRequest _request = new();
    private TrainEventTailer? _tailer;
    private DispatcherTimer? _poll;
    private DateTimeOffset _startedAt;

    public TrainViewModel(MainWindowViewModel shell) : base(shell)
    {
        LossSeries = [];
    }

    public override string Title => "Train";
    public override MaterialIconKind Icon => MaterialIconKind.Tune;
    public override string Blurb => "Train a LoRA on a folder of images and captions.";

    public string[] ModelOptions => LoraleiVocabulary.Models;
    public string[] ProfileOptions => LoraleiVocabulary.Profiles;
    public string[] OptimizerOptions => LoraleiVocabulary.Optimizers;
    public string[] VaeDeviceOptions => LoraleiVocabulary.VaeDevices;
    public string[] DtypeOptions => LoraleiVocabulary.Dtypes;
    public string[] SchedulerOptions => LoraleiVocabulary.LrSchedulers;
    public string[] TimestepOptions => LoraleiVocabulary.TimestepSamplings;
    public string[] WeightingOptions => LoraleiVocabulary.WeightingSchemes;

    // --- basics ---------------------------------------------------------------
    [ObservableProperty] private string _dataset = "";
    [ObservableProperty] private string _name = "";
    [ObservableProperty] private string _model = Loralei.Core.Models.ModelRegistry.DefaultKey;
    [ObservableProperty] private string _profile = "balanced";
    [ObservableProperty] private int? _rank;
    [ObservableProperty] private int? _steps;
    [ObservableProperty] private int? _resolution;
    [ObservableProperty] private double? _learningRate;
    [ObservableProperty] private int? _batchSize;
    [ObservableProperty] private string? _optimizer;
    [ObservableProperty] private int? _seed;
    [ObservableProperty] private string? _triggerWord;
    [ObservableProperty] private int? _gpu;
    [ObservableProperty] private int? _saveEvery;

    // --- memory ---------------------------------------------------------------
    [ObservableProperty] private int? _blocksToSwap;
    [ObservableProperty] private bool? _fp8Base;
    [ObservableProperty] private string? _dtype;
    [ObservableProperty] private bool? _gradientCheckpointing;
    [ObservableProperty] private bool? _swapPinMemory;
    [ObservableProperty] private int? _swapPrefetch;
    [ObservableProperty] private string? _vaeDevice;

    // --- objective ------------------------------------------------------------
    [ObservableProperty] private double? _alpha;
    [ObservableProperty] private double? _loraDropout;
    [ObservableProperty] private string? _targetRegex;
    [ObservableProperty] private string? _excludeRegex;
    [ObservableProperty] private string? _lrScheduler;
    [ObservableProperty] private int? _warmupSteps;
    [ObservableProperty] private int? _gradientAccumulationSteps;
    [ObservableProperty] private double? _maxGradNorm;
    [ObservableProperty] private string? _timestepSampling;
    [ObservableProperty] private double? _flowShift;
    [ObservableProperty] private string? _weightingScheme;

    // --- data -----------------------------------------------------------------
    [ObservableProperty] private bool? _bucketing;
    [ObservableProperty] private int? _bucketStep;
    [ObservableProperty] private string? _captionExtension;
    [ObservableProperty] private double? _captionDropout;
    [ObservableProperty] private int? _repeats;
    [ObservableProperty] private int? _numWorkers;
    [ObservableProperty] private int? _logEvery;

    // --- validation renders ---------------------------------------------------
    [ObservableProperty] private int? _sampleEvery;
    [ObservableProperty] private string _samplePrompts = "";
    [ObservableProperty] private int? _sampleSteps;
    [ObservableProperty] private double? _sampleCfg;
    [ObservableProperty] private int? _sampleGpu;

    [ObservableProperty] private string? _outputRoot;

    // --- preview / live -------------------------------------------------------
    [ObservableProperty] private string _commandPreview = "";
    [ObservableProperty] private string _yamlPreview = "";
    [ObservableProperty] private string _estimate = "";
    [ObservableProperty] private string _cacheStatus = "";
    [ObservableProperty] private bool _overBudget;

    [ObservableProperty] private int _step;
    [ObservableProperty] private int _maxSteps;
    [ObservableProperty] private double _loss;
    [ObservableProperty] private double _vramGib;
    [ObservableProperty] private string _elapsed = "";
    [ObservableProperty] private string _eta = "";
    [ObservableProperty] private string _runState = "";
    [ObservableProperty] private bool _canResume;
    [ObservableProperty] private string? _finalAdapter;

    public ObservableCollection<double> LossSeries { get; }
    public ObservableCollection<string> Checkpoints { get; } = [];
    public ObservableCollection<string> SampleImages { get; } = [];

    public double Progress => MaxSteps > 0 ? Math.Clamp((double)Step / MaxSteps, 0, 1) : 0;

    /// <summary>
    /// Cancelling or crashing loses the adapter unless a checkpoint exists: the final save
    /// happens inside the try block, before the handler that stamps the manifest.
    /// </summary>
    public bool CheckpointWarning => SaveEvery is 0;

    public override Task<bool> ActivateAsync()
    {
        if (Dataset.Length == 0) Dataset = Shell.Settings.LastDataset ?? "";
        SeedFromProjectConfig();
        RefreshPreview();
        RefreshResumeState();
        return Task.FromResult(true);
    }

    /// <summary>
    /// Loads whatever the workspace loralei.yaml already sets into the form.
    /// <para>
    /// This is what makes writing the file safe. The form's empty fields mean "inherit from
    /// the preset", and the writer removes a key for a field the user cleared - so without
    /// seeding first, the very first save would delete settings the user had put in that
    /// file by hand.
    /// </para>
    /// </summary>
    private void SeedFromProjectConfig()
    {
        var doc = ReadProjectConfig();
        string? Text(string key) => doc.Scalar(key) is { } v && v != "null" ? v : null;

        OutputRoot ??= Text("output_root");
        Alpha ??= doc.Double("alpha");
        LoraDropout ??= doc.Double("lora_dropout");
        TargetRegex ??= Text("target_regex");
        ExcludeRegex ??= Text("exclude_regex");
        LrScheduler ??= Text("lr_scheduler");
        WarmupSteps ??= doc.Int("warmup_steps");
        GradientAccumulationSteps ??= doc.Int("gradient_accumulation_steps");
        MaxGradNorm ??= doc.Double("max_grad_norm");
        TimestepSampling ??= Text("timestep_sampling");
        FlowShift ??= doc.Double("flow_shift");
        WeightingScheme ??= Text("weighting_scheme");
        Dtype ??= Text("dtype");
        GradientCheckpointing ??= doc.Bool("gradient_checkpointing");
        SwapPinMemory ??= doc.Bool("swap_pin_memory");
        SwapPrefetch ??= doc.Int("swap_prefetch");
        Bucketing ??= doc.Bool("bucketing");
        BucketStep ??= doc.Int("bucket_step");
        CaptionExtension ??= Text("caption_extension");
        CaptionDropout ??= doc.Double("caption_dropout");
        Repeats ??= doc.Int("repeats");
        NumWorkers ??= doc.Int("num_workers");
        LogEvery ??= doc.Int("log_every");
        SampleEvery ??= doc.Int("sample_every");
        SampleSteps ??= doc.Int("sample_steps");
        SampleCfg ??= doc.Double("sample_cfg");
        SampleGpu ??= doc.Int("sample_gpu");
        if (SamplePrompts.Length == 0) SamplePrompts = string.Join('\n', doc.Sequence("sample_prompts"));
    }

    partial void OnStepChanged(int value) => OnPropertyChanged(nameof(Progress));
    partial void OnMaxStepsChanged(int value) => OnPropertyChanged(nameof(Progress));
    partial void OnSaveEveryChanged(int? value) => OnPropertyChanged(nameof(CheckpointWarning));
    partial void OnNameChanged(string value) => RefreshResumeState();

    private TrainRequest Build()
    {
        _request.Name = Name.Trim();
        _request.Dataset = Dataset.Trim();
        _request.Model = Model;
        _request.Profile = Profile;
        _request.Rank = Rank;
        _request.Steps = Steps;
        _request.Resolution = Resolution;
        _request.LearningRate = LearningRate;
        _request.BatchSize = BatchSize;
        _request.Optimizer = Optimizer;
        _request.Seed = Seed;
        _request.TriggerWord = string.IsNullOrWhiteSpace(TriggerWord) ? null : TriggerWord;
        _request.Gpu = Gpu;
        _request.SaveEvery = SaveEvery;
        _request.BlocksToSwap = BlocksToSwap;
        _request.Fp8Base = Fp8Base;
        _request.VaeDevice = VaeDevice;
        _request.Dtype = Dtype;
        _request.GradientCheckpointing = GradientCheckpointing;
        _request.SwapPinMemory = SwapPinMemory;
        _request.SwapPrefetch = SwapPrefetch;
        _request.Alpha = Alpha;
        _request.LoraDropout = LoraDropout;
        _request.TargetRegex = TargetRegex;
        _request.ExcludeRegex = ExcludeRegex;
        _request.LrScheduler = LrScheduler;
        _request.WarmupSteps = WarmupSteps;
        _request.GradientAccumulationSteps = GradientAccumulationSteps;
        _request.MaxGradNorm = MaxGradNorm;
        _request.TimestepSampling = TimestepSampling;
        _request.FlowShift = FlowShift;
        _request.WeightingScheme = WeightingScheme;
        _request.Bucketing = Bucketing;
        _request.BucketStep = BucketStep;
        _request.CaptionExtension = CaptionExtension;
        _request.CaptionDropout = CaptionDropout;
        _request.Repeats = Repeats;
        _request.NumWorkers = NumWorkers;
        _request.LogEvery = LogEvery;
        _request.SampleEvery = SampleEvery;
        _request.SampleSteps = SampleSteps;
        _request.SampleCfg = SampleCfg;
        _request.SampleGpu = SampleGpu;
        _request.SamplePrompts = SamplePrompts
            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToList();
        _request.OutputRoot = OutputRoot is { Length: > 0 } root && root != "runs" ? root : null;
        return _request;
    }

    /// <summary>
    /// Shows both halves of what will happen: the argv, and the loralei.yaml that carries the
    /// settings with no flag behind them. Neither is hidden from the user.
    /// </summary>
    [RelayCommand]
    private void RefreshPreview()
    {
        var request = Build();
        CommandPreview = Shell.Client.Preview(CommandBuilder.Train(request));

        var doc = ReadProjectConfig();
        request.ApplyYamlOnly(doc);
        YamlPreview = doc.ToString();
    }

    private FlatYaml ReadProjectConfig()
    {
        try
        {
            return FlatYaml.Parse(File.Exists(Shell.Paths.ProjectConfig)
                ? File.ReadAllText(Shell.Paths.ProjectConfig)
                : "");
        }
        catch (IOException) { return FlatYaml.Parse(""); }
    }

    /// <summary>Writes the flagless settings where loralei will look for them: ./loralei.yaml.</summary>
    private bool WriteProjectConfig(TrainRequest request)
    {
        try
        {
            var doc = ReadProjectConfig();
            request.ApplyYamlOnly(doc);
            var text = doc.ToString();
            File.WriteAllText(Shell.Paths.ProjectConfig, text);
            YamlPreview = text;
            Shell.Log.AppendNote($"wrote {Shell.Paths.ProjectConfig}");
            return true;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Error = $"Could not write {Shell.Paths.ProjectConfig}: {e.Message}";
            return false;
        }
    }

    private bool Prepare(out TrainRequest request)
    {
        Clear();
        request = Build();
        var errors = request.Validate();
        if (errors.Count > 0)
        {
            Error = string.Join('\n', errors);
            return false;
        }

        Shell.Settings.LastDataset = Dataset;
        Shell.Settings.Save();
        return WriteProjectConfig(request);
    }

    [RelayCommand]
    private async Task DryRun()
    {
        if (!Prepare(out var request)) return;

        await Shell.RunExclusiveAsync("train --dry-run", async ct =>
        {
            var sink = Shell.Sink(Shell.Client.Preview(CommandBuilder.Train(request, dryRun: true)));
            var result = await Shell.Client.TrainDryRunAsync(request, sink, ct);

            Estimate = result.Estimate;
            CacheStatus = result.CacheStatus;
            OverBudget = result.OverBudget;

            if (!result.Ok) Report(result.Process, "");
            else Status = OverBudget
                ? "Resolved, but the estimate is over this card's VRAM - raise blocks to swap or enable fp8."
                : "Resolved. Nothing was trained.";
        });
    }

    [RelayCommand]
    private Task Start() => Launch(resume: false);

    [RelayCommand]
    private Task Resume() => Launch(resume: true);

    private async Task Launch(bool resume)
    {
        if (!Prepare(out var request)) return;
        request.Resume = resume;

        BeginLiveView(request, resume);

        await Shell.RunExclusiveAsync(resume ? "train --resume" : "train", async ct =>
        {
            var sink = Shell.Sink(Shell.Client.Preview(CommandBuilder.Train(request)));
            var result = await Shell.Client.TrainAsync(request, sink, ct);

            EndLiveView();

            if (result.Cancelled)
            {
                Status = CanResume
                    ? "Cancelled. Resume from the newest checkpoint to continue."
                    : "Cancelled. No checkpoint was written, so this run cannot be resumed.";
                RunState = "failed";
            }
            else if (Report(result, $"Done. {Step} steps."))
            {
                var manifest = Shell.Runs.ReadManifest(request.Name, Shell.OutputRoot);
                FinalAdapter = manifest?.Final;
                RunState = manifest?.State ?? "done";
            }
            RefreshResumeState();
        });
    }

    private void BeginLiveView(TrainRequest request, bool resume)
    {
        Step = 0;
        MaxSteps = Steps ?? 0;
        Loss = 0;
        VramGib = 0;
        Eta = "";
        Elapsed = "";
        FinalAdapter = null;
        RunState = "running";
        _startedAt = DateTimeOffset.Now;

        if (!resume) LossSeries.Clear();
        Checkpoints.Clear();
        SampleImages.Clear();

        _tailer?.Dispose();
        _tailer = new TrainEventTailer(Shell.Runs.EventLogPath(request.Name, Shell.OutputRoot));

        _poll = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(500) };
        _poll.Tick += (_, _) => Pump(request.Name);
        _poll.Start();
    }

    private void EndLiveView()
    {
        _poll?.Stop();
        _poll = null;
        if (_tailer is not null) { Pump(Name); _tailer.Dispose(); _tailer = null; }
    }

    /// <summary>
    /// Drains the live state once. Driven by the poll timer while a run is going; exposed so
    /// tests can advance it without a dispatcher loop to tick that timer.
    /// </summary>
    internal void PollLiveState() => Pump(Name.Trim());

    /// <summary>
    /// Pulls whatever the trainer has appended since the last tick. The JSONL log carries
    /// per-step loss and VRAM; the manifest only moves at checkpoint boundaries, so it is
    /// used for the authoritative state rather than for progress.
    /// </summary>
    private void Pump(string runName)
    {
        if (_tailer is null) return;

        foreach (var ev in _tailer.Poll())
        {
            switch (ev.Event)
            {
                case "start":
                    if (ev.MaxSteps is { } max) MaxSteps = max;
                    break;
                case "save":
                    if (ev.Path is { } path) Checkpoints.Add(Path.GetFileName(path));
                    break;
                case "done":
                    if (ev.Step is { } last) Step = last;
                    FinalAdapter = ev.Final;
                    RunState = "done";
                    break;
                default:
                    if (!ev.IsMetric) break;
                    Step = ev.Step!.Value;
                    if (ev.Loss is { } loss) { Loss = loss; LossSeries.Add(loss); }
                    if (ev.VramGib is { } vram) VramGib = vram;
                    break;
            }
        }

        var elapsed = DateTimeOffset.Now - _startedAt;
        Elapsed = Format(elapsed);
        Eta = Step > 0 && MaxSteps > Step
            ? Format(elapsed * ((double)(MaxSteps - Step) / Step))
            : "";

        RefreshSamples(runName);
    }

    private void RefreshSamples(string runName)
    {
        if (runName.Length == 0) return;
        var images = Shell.Runs.SampleImages(runName, Shell.OutputRoot)
            .Select(f => f.FullName)
            .ToList();
        if (images.Count == SampleImages.Count) return;

        SampleImages.Clear();
        foreach (var image in images) SampleImages.Add(image);
    }

    private void RefreshResumeState() =>
        CanResume = Name.Length > 0 && Shell.Runs.CanResume(Name.Trim(), Shell.OutputRoot);

    private static string Format(TimeSpan span) => span.TotalHours >= 1
        ? $"{(int)span.TotalHours}h {span.Minutes:00}m"
        : $"{span.Minutes:00}m {span.Seconds:00}s";

    /// <summary>Repopulates the whole form from a run that already happened.</summary>
    public void LoadFrom(TrainRequest request)
    {
        Name = request.Name;
        Dataset = request.Dataset;
        Model = request.Model ?? Loralei.Core.Models.ModelRegistry.DefaultKey;
        Rank = request.Rank;
        Steps = request.Steps;
        Resolution = request.Resolution;
        LearningRate = request.LearningRate;
        BatchSize = request.BatchSize;
        Optimizer = request.Optimizer;
        Seed = request.Seed;
        TriggerWord = request.TriggerWord;
        Gpu = request.Gpu;
        SaveEvery = request.SaveEvery;
        BlocksToSwap = request.BlocksToSwap;
        Fp8Base = request.Fp8Base;
        VaeDevice = request.VaeDevice;
        Dtype = request.Dtype;
        GradientCheckpointing = request.GradientCheckpointing;
        SwapPinMemory = request.SwapPinMemory;
        SwapPrefetch = request.SwapPrefetch;
        Alpha = request.Alpha;
        LoraDropout = request.LoraDropout;
        TargetRegex = request.TargetRegex;
        ExcludeRegex = request.ExcludeRegex;
        LrScheduler = request.LrScheduler;
        WarmupSteps = request.WarmupSteps;
        GradientAccumulationSteps = request.GradientAccumulationSteps;
        MaxGradNorm = request.MaxGradNorm;
        TimestepSampling = request.TimestepSampling;
        FlowShift = request.FlowShift;
        WeightingScheme = request.WeightingScheme;
        Bucketing = request.Bucketing;
        BucketStep = request.BucketStep;
        CaptionExtension = request.CaptionExtension;
        CaptionDropout = request.CaptionDropout;
        Repeats = request.Repeats;
        NumWorkers = request.NumWorkers;
        LogEvery = request.LogEvery;
        SampleEvery = request.SampleEvery;
        SampleSteps = request.SampleSteps;
        SampleCfg = request.SampleCfg;
        SampleGpu = request.SampleGpu;
        SamplePrompts = string.Join('\n', request.SamplePrompts);
        OutputRoot = request.OutputRoot;
        RefreshPreview();
        RefreshResumeState();
    }
}
