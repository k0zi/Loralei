using System.Collections.ObjectModel;
using System.ComponentModel;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Loralei.DatasetMaker.Captioning;
using Loralei.DatasetMaker.Config;
using Loralei.DatasetMaker.Processing;
using Loralei.Core;
using Material.Icons;

namespace Loralei.Gui.ViewModels;

/// <summary>
/// The front end of the <c>Loralei.DatasetMaker</c> project: writes a <c>.txt</c> caption beside every
/// image in a folder, by asking a vision model behind an OpenAI-compatible endpoint.
/// <para>
/// Loralei.DatasetMaker was a command-line tool. Its arguments and its <c>appsettings.json</c> are the
/// fields of this page. It runs inside the GUI, and its progress is shown on the page: the
/// shared output pane only gets the start, the failures and the result.
/// </para>
/// <para>
/// This is the one page that talks to a network endpoint itself. It takes the command lock
/// all the same: the endpoint is usually a local model on the same cards a training run wants.
/// </para>
/// </summary>
public sealed partial class CaptionViewModel : PageViewModel
{
    public override string Title => "Caption";
    public override MaterialIconKind Icon => MaterialIconKind.ImageText;
    public override string Blurb =>
        "Write a .txt caption beside every image in a folder, using a vision model behind an OpenAI-compatible endpoint.";

    /// <summary>Kept beside our own settings, so <c>LORALEI_GUI_SETTINGS</c> moves both.</summary>
    public static string SettingsPath =>
        Path.Combine(Path.GetDirectoryName(GuiSettings.DefaultPath) ?? "", "dataset-maker.json");

    private readonly ConfigStore _store = new(SettingsPath);
    private bool _dirty;
    private DispatcherTimer? _clock;
    private DateTimeOffset _startedAt;

    public string[] ImageDetailOptions { get; } = ["high", "low", "auto"];

    /// <summary>Images that failed in the last run, with the reason.</summary>
    public ObservableCollection<string> Failures { get; } = [];

    /// <summary>Images with a request out right now; as many as there are parallel requests.</summary>
    public ObservableCollection<string> InFlight { get; } = [];

    /// <summary>What a preview found to caption.</summary>
    public ObservableCollection<string> Pending { get; } = [];

    [ObservableProperty] private string _dataset = "";

    /// <summary>Not remembered: replacing captions someone may have edited by hand is opted into per run.</summary>
    [ObservableProperty] private bool _overwrite;

    // The endpoint.
    [ObservableProperty] private string _baseUrl = "";
    [ObservableProperty] private string _apiKey = "";
    [ObservableProperty] private string _model = "";
    [ObservableProperty] private int _maxTokens;
    [ObservableProperty] private double _temperature;
    [ObservableProperty] private int _timeoutSeconds;
    [ObservableProperty] private string _imageDetail = "high";

    // The captions.
    [ObservableProperty] private string _systemPrompt = "";
    [ObservableProperty] private string _extraContext = "";
    [ObservableProperty] private string _triggerWord = "";
    [ObservableProperty] private int _maxParallel;
    [ObservableProperty] private int _maxRetries;

    // The run.
    [ObservableProperty] [NotifyPropertyChangedFor(nameof(HasProgress), nameof(Fraction))] private int _total;
    [ObservableProperty] [NotifyPropertyChangedFor(nameof(Fraction))] private int _done;
    [ObservableProperty] private int _succeeded;
    [ObservableProperty] private int _skipped;
    [ObservableProperty] private int _failed;
    [ObservableProperty] [NotifyPropertyChangedFor(nameof(HasLastCaption))] private string? _lastCaption;
    [ObservableProperty] private string? _lastImage;
    [ObservableProperty] private Bitmap? _lastThumbnail;
    [ObservableProperty] private string _phase = "";
    [ObservableProperty] private string _elapsed = "";
    [ObservableProperty] private string _eta = "";
    [ObservableProperty] private bool _isRunning;

    public bool HasProgress => Total > 0;
    public double Fraction => Total > 0 ? Math.Clamp((double)Done / Total, 0, 1) : 0;
    public bool HasLastCaption => !string.IsNullOrEmpty(LastCaption);

    public CaptionViewModel(MainWindowViewModel shell) : base(shell)
    {
        var config = _store.Load();
        BaseUrl = config.OpenAI.BaseUrl;
        ApiKey = config.OpenAI.ApiKey;
        Model = config.OpenAI.Model;
        MaxTokens = config.OpenAI.MaxTokens;
        Temperature = config.OpenAI.Temperature;
        TimeoutSeconds = config.OpenAI.TimeoutSeconds;
        ImageDetail = ImageDetailOptions.Contains(config.OpenAI.ImageDetail) ? config.OpenAI.ImageDetail : "high";
        SystemPrompt = config.Caption.SystemPrompt;
        ExtraContext = config.Caption.ExtraSystemPrompt;
        TriggerWord = config.Caption.TriggerWord;
        MaxParallel = config.Caption.MaxParallel;
        MaxRetries = config.Caption.MaxRetries;
        _dirty = false;
    }

    public override Task<bool> ActivateAsync()
    {
        if (Dataset.Length == 0) Dataset = Shell.Settings.LastDataset ?? "";
        return Task.FromResult(true);
    }

    /// <summary>Everything on the page that is remembered, in the shape Loralei.DatasetMaker's config file always had.</summary>
    public AppConfig CurrentConfig => new()
    {
        OpenAI = new OpenAiOptions
        {
            BaseUrl = BaseUrl.Trim(),
            ApiKey = ApiKey.Trim(),
            Model = Model.Trim(),
            MaxTokens = MaxTokens,
            Temperature = Temperature,
            TimeoutSeconds = TimeoutSeconds,
            ImageDetail = ImageDetail,
        },
        Caption = new CaptionOptions
        {
            SystemPrompt = string.IsNullOrWhiteSpace(SystemPrompt) ? CaptionOptions.DefaultSystemPrompt : SystemPrompt,
            ExtraSystemPrompt = ExtraContext,
            TriggerWord = TriggerWord.Trim(),
            MaxParallel = MaxParallel,
            MaxRetries = MaxRetries,
        },
    };

    private static readonly HashSet<string> Remembered =
    [
        nameof(BaseUrl), nameof(ApiKey), nameof(Model), nameof(MaxTokens), nameof(Temperature),
        nameof(TimeoutSeconds), nameof(ImageDetail), nameof(SystemPrompt), nameof(ExtraContext),
        nameof(TriggerWord), nameof(MaxParallel), nameof(MaxRetries),
    ];

    protected override void OnPropertyChanged(PropertyChangedEventArgs e)
    {
        base.OnPropertyChanged(e);
        if (e.PropertyName is { } name && Remembered.Contains(name)) _dirty = true;
    }

    /// <summary>Writes the fields out, if any changed. Untouched, the page leaves no file behind.</summary>
    public override void Persist()
    {
        if (!_dirty) return;
        _store.Save(CurrentConfig);
        _dirty = false;
    }

    [RelayCommand]
    private void ResetSystemPrompt() => SystemPrompt = CaptionOptions.DefaultSystemPrompt;

    /// <summary>The old <c>--dry-run</c>: lists what would be captioned and sends nothing.</summary>
    [RelayCommand]
    private Task Preview() => RunAsync(dryRun: true);

    [RelayCommand]
    private Task Caption() => RunAsync(dryRun: false);

    private async Task RunAsync(bool dryRun)
    {
        Clear();
        if (!Directory.Exists(Dataset))
        {
            Error = "Pick a dataset folder first.";
            return;
        }
        if (Problem() is { } problem)
        {
            Error = problem;
            return;
        }

        Shell.Settings.LastDataset = Dataset;
        Shell.Settings.Save();
        Persist();

        var config = CurrentConfig;
        var endpoint = config.OpenAI with { ApiKey = ConfigStore.ResolveApiKey(config.OpenAI.ApiKey) };
        var settings = new ProcessorSettings(
            SystemPrompt: PromptBuilder.BuildSystemPrompt(config.Caption.SystemPrompt, ExtraContext),
            TriggerWord: config.Caption.TriggerWord,
            Overwrite: Overwrite,
            MaxParallel: MaxParallel,
            MaxRetries: MaxRetries,
            DryRun: dryRun);
        var directory = Dataset;

        await Shell.RunExclusiveAsync(dryRun ? "caption preview" : "caption", async ct =>
        {
            ResetProgress();
            Shell.Log.AppendCommand(dryRun
                ? $"caption preview: {directory}"
                : $"caption: {directory}  endpoint: {endpoint.BaseUrl}  model: {endpoint.Model}");
            Phase = dryRun ? "Scanning" : "Captioning";
            StartClock();

            try
            {
                using var client = new OpenAiCaptionClient(endpoint);
                // Events come from the worker threads; the page and the log belong to this one.
                // The thumbnail is decoded out there too, so a large image never stalls the page.
                var processor = new DatasetProcessor(client, settings, e =>
                {
                    var thumbnail = e.Status == ItemStatus.Ok ? Thumbnail(e.Image) : null;
                    Dispatcher.UIThread.Post(() => OnEvent(e, directory, thumbnail));
                });

                var summary = await Task.Run(() => processor.RunAsync(directory, ct), ct);
                await DrainEventsAsync();

                Total = summary.Total;
                Phase = dryRun ? "Preview" : "Done";
                Status = Describe(summary, dryRun, directory);
                Shell.Log.AppendNote(Status);
            }
            catch (OperationCanceledException)
            {
                await DrainEventsAsync();
                Phase = "Cancelled";
                Status = $"Cancelled after {Succeeded} caption(s). The ones already written are kept.";
                Shell.Log.AppendNote(Status);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                Phase = "";
                Error = $"Could not read the folder: {e.Message}";
                Status = null;
            }
            finally
            {
                StopClock();
                InFlight.Clear();
            }
        });
    }

    /// <summary>What stops a run before it starts, or null. A bad value here would otherwise fail once per image.</summary>
    private string? Problem()
    {
        if (!Uri.TryCreate(BaseUrl.Trim(), UriKind.Absolute, out var uri) ||
            (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps))
            return "The endpoint must be an http(s) URL, e.g. http://127.0.0.1:8080/v1.";
        if (string.IsNullOrWhiteSpace(Model)) return "Name the model to ask.";
        if (MaxTokens < 1) return "Max tokens must be at least 1.";
        if (TimeoutSeconds < 1) return "The timeout must be at least 1 second.";
        if (MaxRetries < 0) return "Retries cannot be negative.";
        return null;
    }

    private void ResetProgress()
    {
        Failures.Clear();
        InFlight.Clear();
        Pending.Clear();
        Total = Done = Succeeded = Skipped = Failed = 0;
        LastCaption = LastImage = null;
        LastThumbnail = null;
        Elapsed = Eta = "";
    }

    private void OnEvent(ProcessingEvent e, string directory, Bitmap? thumbnail)
    {
        var name = Path.GetRelativePath(directory, e.Image);
        Total = e.Total;

        switch (e.Status)
        {
            case ItemStatus.Started:
                InFlight.Add(name);
                return; // not finished; it reports again
            case ItemStatus.Retrying:
                Shell.Log.AppendNote(e.ToString());
                return;
            case ItemStatus.Todo:
                Pending.Add(name);
                break;
            case ItemStatus.Ok:
                Succeeded++;
                LastImage = name;
                LastCaption = e.Caption;
                LastThumbnail = thumbnail;
                break;
            case ItemStatus.Skipped:
                Skipped++;
                break;
            case ItemStatus.Failed:
                Failed++;
                Failures.Add($"{e.Image}: {e.Detail}");
                Shell.Log.AppendNote(e.ToString());
                break;
        }
        InFlight.Remove(name);
        Done++;
        Tick();
    }

    /// <summary>A small copy for the page, or null for a format Avalonia cannot decode.</summary>
    private static Bitmap? Thumbnail(string image)
    {
        try
        {
            using var stream = File.OpenRead(image);
            return Bitmap.DecodeToHeight(stream, 160);
        }
        catch (Exception) { return null; }
    }

    private void StartClock()
    {
        _startedAt = DateTimeOffset.Now;
        IsRunning = true;
        _clock = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        _clock.Tick += (_, _) => Tick();
        _clock.Start();
    }

    private void StopClock()
    {
        _clock?.Stop();
        _clock = null;
        IsRunning = false;
        Eta = "";
    }

    /// <summary>
    /// The estimate counts only images that went to the endpoint: a skipped one costs nothing,
    /// and a folder that is mostly captioned already would otherwise promise to finish at once.
    /// </summary>
    private void Tick()
    {
        if (!IsRunning) return;
        var elapsed = DateTimeOffset.Now - _startedAt;
        Elapsed = Format(elapsed);

        var sent = Succeeded + Failed;
        var left = Total - Done;
        Eta = sent > 0 && left > 0 ? Format(elapsed * ((double)left / sent)) : "";
    }

    private static string Format(TimeSpan span) => span.TotalHours >= 1
        ? $"{(int)span.TotalHours}h {span.Minutes:00}m"
        : $"{span.Minutes:00}m {span.Seconds:00}s";

    /// <summary>
    /// Waits for the events already posted. They queue at the same priority as this, so by
    /// the time it runs they have all been applied and the counters are final.
    /// </summary>
    private static async Task DrainEventsAsync() =>
        await Dispatcher.UIThread.InvokeAsync(() => { }, DispatcherPriority.Default);

    private static string Describe(ProcessingSummary summary, bool dryRun, string directory)
    {
        if (summary.Total == 0) return $"No images found in {directory}.";
        if (dryRun)
            return $"{summary.Total - summary.Skipped} of {summary.Total} image(s) would be captioned; " +
                   $"{summary.Skipped} skipped. Nothing was sent.";
        return $"Done. Captioned {summary.Succeeded}, skipped {summary.Skipped}, failed {summary.Failed}.";
    }
}
