using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using Loralei.Core.Config;
using Loralei.Core.Dto;
using Loralei.Core.Models;
using Loralei.Core.Worker;

namespace Loralei.Core.Diagnostics;

/// <summary>
/// <c>loralei doctor</c> - validates the ROCm stack before a long run wastes hours.
/// <para>
/// Every check knows how to explain itself: on failure it says what was expected, what was
/// observed, and the single command or flag that fixes it. The checks that need torch run in
/// the Python worker; everything else - driver access, the allocator and MIOpen settings the
/// worker is started with, disk, host RAM - is answered here.
/// </para>
/// </summary>
public static partial class Doctor
{
    /// <summary>The checks the worker reports, in the order they belong in the table.</summary>
    private static readonly string[] WorkerAfterDevices = ["SDPA backend", "bf16 matmul", "8-bit Adam"];

    private static string F(double value, string format = "F1") => value.ToString(format, CultureInfo.InvariantCulture);

    private static DoctorCheck Check(string name, string status, string detail = "", string remedy = "") =>
        new() { Name = name, Status = status, Detail = detail, Remedy = remedy };

    public static async Task<DoctorReport> RunAsync(bool quick, EnvReport env, string python,
        Action<string>? onText = null, CancellationToken ct = default)
    {
        var checks = new List<DoctorCheck> { DriverAccess(), RocmUserspace() };

        var reported = new Dictionary<string, DoctorCheck>(StringComparer.Ordinal);
        var worker = new PythonWorker(python, env.Environment);
        var result = await worker.RunAsync(WorkerJobs.Doctor(quick, env.Agents), e =>
        {
            if (!e.TryGetProperty("event", out var kind) || kind.GetString() != "check") return;
            var check = e.Deserialize<DoctorCheck>();
            if (check is not null) reported[check.Name] = check;
        }, onText, ct).ConfigureAwait(false);

        if (!reported.TryGetValue("torch", out var torch))
            torch = Check("torch", "fail",
                result.Error ?? $"the Python worker exited with code {result.ExitCode} before reporting",
                "check that LORALEI_PYTHON (or the project .venv) points at the environment torch is installed in");
        checks.Add(torch);

        if (torch.Status != "fail")
        {
            // A check that takes the whole process down - a driver abort inside a matmul, say -
            // reports nothing. Name it, rather than leaving a hole in the table.
            var crashed = !result.Ok && !result.Cancelled;
            DoctorCheck Reported(string name)
            {
                if (reported.TryGetValue(name, out var check)) return check;
                if (!crashed) return Check(name, "skip", "not run");
                crashed = false;
                return Check(name, "fail",
                    $"the worker exited with code {result.ExitCode} during this check; its output is above",
                    "this is the ROCm stack failing, not loralei - a run on this device would die the same way");
            }

            checks.Add(Reported("devices"));
            checks.Add(Allocator(env));
            checks.Add(Miopen(env));
            checks.AddRange(WorkerAfterDevices.Select(Reported));
            checks.Add(DiskSpace());
            checks.Add(HostRam());
            checks.Add(RocmSmi());
        }

        return new DoctorReport
        {
            LoraleiVersion = LoraleiInfo.Version,
            Ok = checks.All(c => c.Status != "fail"),
            EnvApplied = env.Applied,
            Notes = env.Notes,
            Checks = checks,
        };
    }

    /// <summary>
    /// Can this user actually talk to the GPU driver? <c>/dev/dri</c> itself is a root-owned
    /// directory, so its mode says nothing - what matters is whether at least one
    /// <c>renderD*</c> node is open for read/write.
    /// </summary>
    public static DoctorCheck DriverAccess()
    {
        var problems = new List<string>();
        if (!File.Exists("/dev/kfd")) problems.Add("/dev/kfd missing (amdgpu kernel driver not loaded?)");
        else if (!ReadWritable("/dev/kfd")) problems.Add("/dev/kfd not read/writable");

        string[] nodes;
        try { nodes = Directory.Exists("/dev/dri") ? Directory.GetFiles("/dev/dri", "renderD*") : []; }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { nodes = []; }

        if (nodes.Length == 0) problems.Add("no /dev/dri/renderD* node found");
        else if (!nodes.Any(ReadWritable)) problems.Add($"none of {nodes.Length} render node(s) are read/writable");

        return problems.Count == 0
            ? Check("driver access", "ok", $"/dev/kfd and {nodes.Length} render node(s) accessible")
            : Check("driver access", "fail",
                string.Join("; ", problems) + " (is your user in the 'render' and 'video' groups?)",
                "sudo usermod -aG render,video $USER   # then log out and back in");
    }

    private static DoctorCheck RocmUserspace() =>
        RocmEnvironment.Capture("hipconfig", ["--version"], 30) is { } version
            ? Check("ROCm userspace", "ok", $"hipconfig {version.Trim()}")
            : Check("ROCm userspace", "warn", "hipconfig not found on PATH",
                "not fatal - torch bundles its own ROCm runtime - but diagnostics are limited");

    private static DoctorCheck Allocator(EnvReport env)
    {
        var conf = env.Environment.GetValueOrDefault("PYTORCH_HIP_ALLOC_CONF", "");
        return conf.Contains("expandable_segments:True", StringComparison.Ordinal)
            ? Check("allocator", "ok", $"PYTORCH_HIP_ALLOC_CONF={conf}")
            : Check("allocator", "warn", $"PYTORCH_HIP_ALLOC_CONF={(conf.Length > 0 ? conf : "<unset>")}",
                "expandable_segments:True materially reduces fragmentation with block swap");
    }

    private static DoctorCheck Miopen(EnvReport env) =>
        env.Environment.GetValueOrDefault("MIOPEN_USER_DB_PATH") is { Length: > 0 } db
            ? Check("MIOpen cache", "ok", $"persisted at {db}")
            : Check("MIOpen cache", "warn", "MIOPEN_USER_DB_PATH unset",
                "first run may spend several minutes tuning kernels and look hung");

    public static DoctorCheck DiskSpace(IReadOnlyList<string>? paths = null)
    {
        paths ??= ["runs", Path.Combine(PathText.Home, ".cache", "huggingface")];
        var worst = double.PositiveInfinity;
        var parts = new List<string>();
        foreach (var path in paths)
        {
            var free = Disk.FreeBytes(path) / (double)Disk.Gib;
            worst = Math.Min(worst, free);
            parts.Add($"{path}: {F(free)} GiB free");
        }
        var detail = string.Join("; ", parts);
        if (worst < 5) return Check("disk", "fail", detail, "free space with `loralei models gc` before training");
        if (worst < 20)
            return Check("disk", "warn", detail,
                "latent caches and checkpoints need room; `loralei models gc --dry-run` lists candidates");
        return Check("disk", "ok", detail);
    }

    private static DoctorCheck HostRam()
    {
        long total = 0, available = 0;
        try
        {
            foreach (var line in File.ReadLines("/proc/meminfo"))
            {
                var fields = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (fields.Length < 2 || !long.TryParse(fields[1], out var kib)) continue;
                if (fields[0] == "MemTotal:") total = kib * 1024;
                else if (fields[0] == "MemAvailable:") available = kib * 1024;
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }

        if (total == 0) return Check("host RAM", "skip", "/proc/meminfo is not readable");
        if (available == 0) available = total;
        return Check("host RAM", "ok",
            $"{F(available / (double)Disk.Gib, "F0")} GiB available of {F(total / (double)Disk.Gib, "F0")} GiB");
    }

    private static DoctorCheck RocmSmi()
    {
        if (RocmEnvironment.Which("rocm-smi") is null) return Check("rocm-smi", "skip", "not installed");
        var output = RocmEnvironment.Capture("rocm-smi", ["--showuse"], 60, requireSuccess: false);
        if (output is null) return Check("rocm-smi", "warn", "did not answer within 60 s");
        return output.Contains("low-power state", StringComparison.Ordinal)
            ? Check("rocm-smi", "ok", "GPUs idle in a low-power state (normal; they wake on first allocation)")
            : Check("rocm-smi", "ok", "reachable");
    }

    private const int ReadOk = 4, WriteOk = 2;

    private static bool ReadWritable(string path)
    {
        try { return access(path, ReadOk | WriteOk) == 0; }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException) { return false; }
    }

    [LibraryImport("libc", StringMarshalling = StringMarshalling.Utf8, SetLastError = true)]
    private static partial int access(string path, int mode);
}
