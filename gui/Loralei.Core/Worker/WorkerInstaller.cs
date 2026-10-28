using System.Diagnostics;
using System.Text;
using Loralei.Core.Config;

namespace Loralei.Core.Worker;

/// <summary>One command of the install, and what it is for.</summary>
public sealed record InstallStep(string Title, IReadOnlyList<string> Command);

/// <summary>Where the worker gets installed, and the commands that do it.</summary>
/// <param name="Source">The checkout: the folder holding <c>pyproject.toml</c>.</param>
/// <param name="Venv">The environment the worker is installed into.</param>
public sealed record InstallPlan(string Source, string Venv, IReadOnlyList<InstallStep> Steps)
{
    public string Python => Path.Combine(Venv, "bin", "python");
}

/// <summary>
/// <c>loralei doctor --install</c> - sets up the Python worker's environment: a <c>.venv</c>
/// in the checkout, torch from the ROCm wheel index, then the worker itself. These are the
/// commands the README lists, so that a first run is one command (or one button) instead of four.
/// <para>
/// This is the one place besides <c>models pull</c> that goes online: it downloads several
/// gigabytes of wheels from download.pytorch.org and PyPI. It sends nothing of the user's.
/// </para>
/// </summary>
public static class WorkerInstaller
{
    /// <summary>The worker requires 3.12 exactly (<c>requires-python</c> in pyproject.toml).</summary>
    public const string PythonVersion = "3.12";

    /// <summary>Pinned to match the ROCm release the trainer was measured on.</summary>
    public const string TorchIndex = "https://download.pytorch.org/whl/rocm7.1";

    public static readonly IReadOnlyList<string> TorchPackages = ["torch==2.13.0+rocm7.1", "torchvision==0.28.0+rocm7.1"];

    /// <summary>
    /// The checkout the worker is installed from: the nearest folder with the worker's
    /// <c>pyproject.toml</c>, walking up from the working directory and then from this binary -
    /// the same walk <see cref="PythonWorker.FindPython"/> does, so the <c>.venv</c> made
    /// there is the one it finds.
    /// </summary>
    public static string? FindSource(IEnumerable<string>? starts = null)
    {
        foreach (var start in starts ?? [Directory.GetCurrentDirectory(), AppContext.BaseDirectory])
        for (var dir = new DirectoryInfo(start); dir is not null; dir = dir.Parent)
            if (File.Exists(Path.Combine(dir.FullName, "pyproject.toml")) &&
                Directory.Exists(Path.Combine(dir.FullName, "src", "loralei")))
                return dir.FullName;
        return null;
    }

    /// <param name="uv">Path to <c>uv</c>, or null to fall back to <c>python3.12 -m venv</c> and pip.</param>
    public static InstallPlan Plan(string source, string? uv)
    {
        var venv = Path.Combine(source, ".venv");
        var python = Path.Combine(venv, "bin", "python");
        // bnb: the default optimizer is 8-bit Adam.
        var worker = source + "[bnb]";

        List<string> Pip(params string[] args) => uv is not null
            ? [uv, "pip", "install", "--python", python, .. args]
            : [python, "-m", "pip", "install", .. args];

        var steps = new List<InstallStep>();
        if (!File.Exists(python))
            steps.Add(new($"create {venv}", uv is not null
                ? [uv, "venv", "--python", PythonVersion, venv]
                : ["python" + PythonVersion, "-m", "venv", venv]));
        steps.Add(new("install torch for ROCm", Pip(["--index-url", TorchIndex, .. TorchPackages])));
        steps.Add(new("install the loralei worker", Pip("-e", worker)));
        return new InstallPlan(source, venv, steps);
    }

    /// <summary>Works out the plan for this machine, or throws saying what is in the way.</summary>
    public static InstallPlan Plan()
    {
        var source = FindSource() ?? throw new ConfigException(
            "cannot find the loralei checkout (the folder with pyproject.toml and src/loralei); " +
            "run this from inside it");
        var uv = RocmEnvironment.Which("uv");
        if (uv is null && RocmEnvironment.Which("python" + PythonVersion) is null &&
            !File.Exists(Path.Combine(source, ".venv", "bin", "python")))
            throw new ConfigException(
                $"neither uv nor python{PythonVersion} is on PATH; install uv first: " +
                "curl -LsSf https://astral.sh/uv/install.sh | sh");
        return Plan(source, uv);
    }

    /// <summary>Runs the plan, stopping at the first step that fails.</summary>
    /// <param name="onStep">Called before each step.</param>
    /// <param name="onText">The tools' own output, as it arrives.</param>
    /// <returns>0, or the exit code of the step that failed.</returns>
    public static async Task<int> RunAsync(InstallPlan plan, Action<InstallStep>? onStep = null,
        Action<string>? onText = null, CancellationToken ct = default)
    {
        foreach (var step in plan.Steps)
        {
            ct.ThrowIfCancellationRequested();
            onStep?.Invoke(step);
            var code = await RunStepAsync(step, onText, ct).ConfigureAwait(false);
            if (code != 0) return code;
        }
        return 0;
    }

    private static async Task<int> RunStepAsync(InstallStep step, Action<string>? onText, CancellationToken ct)
    {
        var info = new ProcessStartInfo(step.Command[0])
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = true,
            UseShellExecute = false,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        foreach (var arg in step.Command.Skip(1)) info.ArgumentList.Add(arg);

        using var process = new Process { StartInfo = info };
        try
        {
            process.Start();
        }
        catch (System.ComponentModel.Win32Exception e)
        {
            onText?.Invoke($"could not start {step.Command[0]}: {e.Message}\n");
            return 127;
        }
        process.StandardInput.Close();

        var gate = new object();
        async Task PumpAsync(StreamReader reader)
        {
            var buffer = new char[4096];
            while (true)
            {
                var read = await reader.ReadAsync(buffer).ConfigureAwait(false);
                if (read == 0) break;
                lock (gate) onText?.Invoke(new string(buffer, 0, read));
            }
        }

        var pumps = Task.WhenAll(PumpAsync(process.StandardOutput), PumpAsync(process.StandardError));
        try
        {
            await process.WaitForExitAsync(ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            // Nothing to save: an interrupted install is simply run again.
            try { if (!process.HasExited) process.Kill(entireProcessTree: true); }
            catch (InvalidOperationException) { /* already reaped */ }
            await pumps.ConfigureAwait(false);
            throw;
        }

        await pumps.ConfigureAwait(false);
        process.WaitForExit();
        return process.ExitCode;
    }
}
