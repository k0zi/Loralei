using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Loralei.Core.Dto;

using Material.Icons;

namespace Loralei.Gui.ViewModels;

public sealed partial class DoctorViewModel(MainWindowViewModel shell) : PageViewModel(shell)
{
    public override string Title => "Doctor";
    public override MaterialIconKind Icon => MaterialIconKind.Stethoscope;
    public override string Blurb => "Validate the ROCm stack before committing a run to it.";

    public ObservableCollection<DoctorCheck> Checks { get; } = [];
    public ObservableCollection<KeyValuePair<string, string>> Environment { get; } = [];

    [ObservableProperty] private bool _thorough;
    [ObservableProperty] private bool _allOk;
    [ObservableProperty] private string _version = "";
    [ObservableProperty] private string? _fixSnippet;

    // Deliberately not run on activation: the quick checks still allocate VRAM for a bf16
    // matmul and an 8-bit Adam step, which could push an already-running training job -
    // possibly started from a terminal - over the edge of a 16 GiB card.
    public override Task<bool> ActivateAsync() => Task.FromResult(true);

    [RelayCommand]
    private async Task Run()
    {
        Clear();
        await Shell.RunExclusiveAsync("doctor", async ct =>
        {
            Shell.Log.AppendCommand(Shell.Client.Preview(
                Loralei.Core.CommandBuilder.Doctor(json: true).Concat(Thorough ? ["--thorough"] : []).ToList()));

            var (report, result, raw) = await Shell.Client.DoctorAsync(Thorough, ct);
            if (report is null)
            {
                // The table cannot be built, so the raw payload is all there is to go on.
                Shell.Log.AppendNote(raw);
                Report(result, "");
                Error ??= "doctor did not return parseable JSON. See the log below.";
                return;
            }

            Checks.Clear();
            foreach (var check in report.Checks) Checks.Add(check);

            Environment.Clear();
            foreach (var pair in report.EnvApplied) Environment.Add(pair);

            AllOk = report.Ok;
            Version = report.LoraleiVersion;
            var failed = report.Checks.Count(c => c.Status == "fail");
            var warned = report.Checks.Count(c => c.Status == "warn");
            Shell.Log.AppendNote(
                $"{report.Checks.Count} checks: {report.Checks.Count(c => c.Status == "ok")} ok, " +
                $"{warned} warn, {failed} fail (exit {result.ExitCode})");
            Status = failed > 0
                ? $"{failed} check(s) failed, {warned} warning(s)."
                : warned > 0 ? $"All checks passed, with {warned} warning(s)." : "All checks passed.";
        });
    }

    /// <summary>
    /// Sets up the worker's Python environment, then runs the checks against it. Only ever on
    /// a click: it is the one thing on this page that goes online, and it downloads gigabytes.
    /// </summary>
    [RelayCommand]
    private async Task Install()
    {
        Clear();
        var installed = false;
        await Shell.RunExclusiveAsync("doctor --install", async ct =>
        {
            var result = await Shell.Client.DoctorInstallAsync(
                Shell.Sink(Shell.Client.Preview(Loralei.Core.CommandBuilder.DoctorInstall())), ct);
            installed = Report(result, "Worker installed.");
        });
        if (installed) await Run();
    }

    [RelayCommand]
    private async Task ShowFix()
    {
        await Shell.RunExclusiveAsync("doctor --fix", async ct =>
        {
            FixSnippet = await Shell.Client.DoctorFixSnippetAsync(ct);
        });
    }
}
