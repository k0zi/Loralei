using System.Runtime.InteropServices;

namespace Loralei.Core;

/// <summary>
/// Signal delivery. .NET has no API for sending a signal other than SIGKILL
/// (<see cref="System.Diagnostics.Process.Kill()"/>), but loralei needs SIGINT: its training
/// loop catches <c>BaseException</c>, stamps <c>manifest.json</c> with <c>state="failed"</c>
/// and closes the block swapper. SIGKILL would skip all of that.
/// </summary>
internal static partial class Native
{
    public const int SIGINT = 2;
    public const int SIGTERM = 15;
    public const int SIGKILL = 9;

    [LibraryImport("libc", SetLastError = true)]
    private static partial int kill(int pid, int sig);

    /// <summary>Signals a process, or the whole group when <paramref name="group"/> is set.</summary>
    /// <returns>False if the process was already gone, or on a non-Unix host.</returns>
    public static bool Signal(int pid, int sig, bool group = true)
    {
        if (!OperatingSystem.IsLinux() && !OperatingSystem.IsMacOS()) return false;
        try
        {
            // A negative pid addresses the process group, so detached sampler children
            // spawned by the trainer are signalled too.
            return kill(group ? -pid : pid, sig) == 0;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
    }
}
