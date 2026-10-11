using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Loralei.Core.Worker;

public sealed record WorkerResult(int ExitCode, bool Cancelled, string? Error)
{
    public bool Ok => ExitCode == 0 && !Cancelled;
}

/// <summary>
/// Runs one job in the Python worker - the part of loralei that needs torch.
/// <para>
/// The contract is deliberately narrow. The job goes in as one JSON document on stdin, fully
/// resolved: every path absolute, every setting decided. Progress comes back as one JSON
/// object per line on stdout. stderr is whatever torch, diffusers and tqdm feel like printing
/// and is passed through untouched. The worker decides nothing that could be decided here.
/// </para>
/// </summary>
public sealed class PythonWorker(string python, IReadOnlyDictionary<string, string> environment)
{
    /// <summary>How long an interrupted job gets to stamp its manifest and release the GPU.</summary>
    public static readonly TimeSpan GracefulStopTimeout = TimeSpan.FromSeconds(25);

    public string Python { get; } = python;

    /// <summary>
    /// The interpreter to run the worker with: <c>$LORALEI_PYTHON</c>, else the nearest
    /// <c>.venv</c> walking up from the working directory and then from this binary, else
    /// <c>python3</c> on PATH.
    /// </summary>
    public static string FindPython()
    {
        if (Environment.GetEnvironmentVariable("LORALEI_PYTHON") is { Length: > 0 } configured) return configured;

        foreach (var start in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        {
            for (var dir = new DirectoryInfo(start); dir is not null; dir = dir.Parent)
            {
                var candidate = Path.Combine(dir.FullName, ".venv", "bin", "python");
                if (File.Exists(candidate)) return candidate;
            }
        }
        return RocmEnvironment.Which("python3") ?? "python3";
    }

    /// <param name="onEvent">Each JSON object the worker reports.</param>
    /// <param name="onText">Raw stderr chunks, and any stdout line that was not JSON.</param>
    public async Task<WorkerResult> RunAsync(JsonObject job, Action<JsonElement>? onEvent = null,
        Action<string>? onText = null, CancellationToken ct = default)
    {
        var info = new ProcessStartInfo(Python)
        {
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        info.ArgumentList.Add("-m");
        info.ArgumentList.Add("loralei");
        info.ArgumentList.Add("-");
        info.Environment.Clear();
        foreach (var (key, value) in environment) info.Environment[key] = value;

        using var process = new Process { StartInfo = info };
        try
        {
            process.Start();
        }
        catch (System.ComponentModel.Win32Exception e)
        {
            return new WorkerResult(127, false,
                $"could not start the Python worker ({Python}): {e.Message}. " +
                "Set LORALEI_PYTHON to the interpreter of the environment loralei's worker is installed in.");
        }

        string? error = null;
        var gate = new object();

        async Task PumpEventsAsync()
        {
            while (await process.StandardOutput.ReadLineAsync(CancellationToken.None).ConfigureAwait(false) is { } line)
            {
                if (line.Length == 0) continue;
                JsonDocument? doc = null;
                if (line[0] == '{')
                {
                    try { doc = JsonDocument.Parse(line); }
                    catch (JsonException) { /* a library printed something brace-shaped */ }
                }

                lock (gate)
                {
                    if (doc is null) { onText?.Invoke(line + "\n"); continue; }
                    using (doc)
                    {
                        var root = doc.RootElement;
                        if (root.TryGetProperty("event", out var kind) && kind.GetString() == "error")
                            error = root.TryGetProperty("message", out var message) ? message.GetString() : "worker failed";
                        onEvent?.Invoke(root);
                    }
                }
            }
        }

        async Task PumpTextAsync()
        {
            var buffer = new char[4096];
            while (true)
            {
                var read = await process.StandardError.ReadAsync(buffer).ConfigureAwait(false);
                if (read == 0) break;
                lock (gate) onText?.Invoke(new string(buffer, 0, read));
            }
        }

        var pumps = Task.WhenAll(PumpEventsAsync(), PumpTextAsync());

        try
        {
            await process.StandardInput.WriteAsync(job.ToJsonString()).ConfigureAwait(false);
            process.StandardInput.Close();
        }
        catch (IOException) { /* the worker died before reading its job; the exit code says why */ }

        var cancelled = false;
        try
        {
            await process.WaitForExitAsync(ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            cancelled = true;
            await StopAsync(process).ConfigureAwait(false);
        }

        await pumps.ConfigureAwait(false);
        process.WaitForExit();
        return new WorkerResult(process.ExitCode, cancelled, error);
    }

    /// <summary>
    /// SIGINT first, so the training loop can stamp the manifest and release the block
    /// swapper, then SIGKILL if it will not go. Only the worker is signalled: its sampler
    /// children run in their own session, and the trainer waits for them itself.
    /// </summary>
    private static async Task StopAsync(Process process)
    {
        if (process.HasExited) return;
        if (!Native.Signal(process.Id, Native.SIGINT, group: false))
        {
            TryKill(process);
            return;
        }

        try
        {
            using var grace = new CancellationTokenSource(GracefulStopTimeout);
            await process.WaitForExitAsync(grace.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            TryKill(process);
        }
    }

    private static void TryKill(Process process)
    {
        try { if (!process.HasExited) process.Kill(entireProcessTree: true); }
        catch (InvalidOperationException) { /* already reaped */ }
    }
}
