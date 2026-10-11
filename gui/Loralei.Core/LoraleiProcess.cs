using System.Diagnostics;
using System.Text;

namespace Loralei.Core;

public sealed record LoraleiResult(int ExitCode, bool Cancelled, IReadOnlyList<string> StdErrTail)
{
    public bool Ok => ExitCode == 0 && !Cancelled;
}

/// <summary>Runs the loralei CLI as a child process and streams its output back line by line.</summary>
public sealed class LoraleiProcess(LoraleiPaths paths)
{
    /// <summary>How long to let a SIGINT'd run shut down before offering to force it.</summary>
    public static readonly TimeSpan GracefulStopTimeout = TimeSpan.FromSeconds(30);

    private const int StdErrTailLines = 40;

    /// <param name="forceColor">
    /// Asks the CLI for colour and live progress frames even though it is writing to a pipe.
    /// Must stay off for the <c>--json</c> commands, whose output has to stay plain.
    /// </param>
    public async Task<LoraleiResult> RunAsync(
        IReadOnlyList<string> args,
        Action<ConsoleLine>? onOutput = null,
        bool forceColor = false,
        CancellationToken ct = default)
    {
        var info = new ProcessStartInfo(paths.Executable)
        {
            WorkingDirectory = paths.Workspace,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = true,
            UseShellExecute = false,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        foreach (var arg in args) info.ArgumentList.Add(arg);

        if (forceColor)
        {
            info.Environment["FORCE_COLOR"] = "1";
        }
        else
        {
            // Belt and braces: through a pipe the CLI is already plain, but an inherited
            // FORCE_COLOR from our own environment would corrupt --json output.
            info.Environment.Remove("FORCE_COLOR");
            info.Environment["NO_COLOR"] = "1";
        }

        using var process = new Process { StartInfo = info };
        var stdErrTail = new Queue<string>();
        var gate = new object();

        void Publish(ConsoleLine line, bool isStdErr)
        {
            lock (gate)
            {
                if (isStdErr)
                {
                    stdErrTail.Enqueue(line.Text);
                    while (stdErrTail.Count > StdErrTailLines) stdErrTail.Dequeue();
                }
                onOutput?.Invoke(line);
            }
        }

        process.Start();
        // Nothing types at these prompts; close stdin so an unexpected confirmation is
        // answered "no" at once instead of hanging the GUI forever.
        process.StandardInput.Close();

        var pump = Task.WhenAll(
            PumpAsync(process.StandardOutput, l => Publish(l, false)),
            PumpAsync(process.StandardError, l => Publish(l, true)));

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

        await pump.ConfigureAwait(false);
        // WaitForExitAsync returns before the redirected streams are drained; this
        // overload blocks until they are, so no trailing output is lost.
        process.WaitForExit();

        return new LoraleiResult(process.ExitCode, cancelled, stdErrTail.ToArray());
    }

    /// <summary>
    /// SIGINT first - the CLI passes it on to its Python worker, so the training loop can
    /// stamp the manifest and release the block swapper - then SIGKILL if it will not go.
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

    private static async Task PumpAsync(StreamReader reader, Action<ConsoleLine> sink)
    {
        var parser = new ConsoleStreamParser();
        var buffer = new char[4096];
        while (true)
        {
            var read = await reader.ReadAsync(buffer).ConfigureAwait(false);
            if (read == 0) break;
            foreach (var line in parser.Feed(new string(buffer, 0, read))) sink(line);
        }
        if (parser.Flush() is { } last) sink(last);
    }
}
