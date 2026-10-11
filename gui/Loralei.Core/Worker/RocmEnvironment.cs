using System.Diagnostics;
using Loralei.Core.Config;

namespace Loralei.Core.Worker;

/// <summary>What <see cref="RocmEnvironment.Bootstrap"/> actually did, for <c>doctor</c> to print.</summary>
public sealed class EnvReport
{
    /// <summary>The complete environment the worker should start with.</summary>
    public Dictionary<string, string> Environment { get; } = new(StringComparer.Ordinal);
    public Dictionary<string, string> Applied { get; } = new(StringComparer.Ordinal);
    public Dictionary<string, string> Preexisting { get; } = new(StringComparer.Ordinal);
    public List<string> Agents { get; set; } = [];
    public List<int> Selected { get; set; } = [];
    public List<string> Notes { get; } = [];
}

/// <summary>
/// The environment the Python worker is started with.
/// <para>
/// On gfx1200 (RDNA4) PyTorch's SDPA silently falls back to the MATH backend unless
/// <c>TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1</c> is set. That flag is read into a
/// <c>static const bool</c> on the first SDPA dispatch inside the C++ extension, so it has to
/// be in place before torch is imported. Measured on a RX 9060 XT, 4096-token attention:
/// 17.3 ms / 0.18 GiB with flash vs 387.2 ms / 12.14 GiB with math. Setting it on the child's
/// environment here, before the interpreter even starts, is the strongest form of "before".
/// </para>
/// </summary>
public static class RocmEnvironment
{
    /// <summary>
    /// gfx targets that are integrated GPUs (APUs). They enumerate alongside discrete cards
    /// and will OOM instantly against shared system memory if training lands there.
    /// </summary>
    public static readonly IReadOnlySet<string> ApuGfxTargets = new HashSet<string>
    {
        "gfx90c", "gfx1013", "gfx1033", "gfx1035", "gfx1036", "gfx1103",
        "gfx1150", "gfx1151", "gfx1152", "gfx1153",
    };

    /// <summary>
    /// Applied unless already present in the environment. Deliberately not set:
    /// <c>HSA_OVERRIDE_GFX_VERSION</c> - gfx1200 is natively supported by ROCm >= 7.0.2, and
    /// overriding it only breaks kernel selection.
    /// </summary>
    public static readonly IReadOnlyList<KeyValuePair<string, string>> BaseEnv =
    [
        new("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1"),
        // Block swap allocates and frees large pinned/device buffers every step.
        new("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True"),
        new("HSA_FORCE_FINE_GRAIN_PCIE", "1"),
        // Offline by default. Every model component is resolved to a local directory before a
        // library sees it, but diffusers/transformers treat any string that is not an existing
        // path as a Hub repo id and will go to huggingface.co for it - measured: a mistyped
        // --lora path produced ten HTTPS connections. With these set, such a lookup fails
        // locally instead. The worker never downloads; `models pull` does that from C#.
        new("HF_HUB_OFFLINE", "1"),
        new("TRANSFORMERS_OFFLINE", "1"),
        new("HF_DATASETS_OFFLINE", "1"),
        new("HF_HUB_DISABLE_TELEMETRY", "1"),
        new("DISABLE_TELEMETRY", "1"),
        new("DO_NOT_TRACK", "1"),
    ];

    public static string MiopenDb => Path.Combine(PathText.Home, ".cache", "loralei", "miopen");

    /// <summary>
    /// gfx target of every GPU agent, in HIP device-index order. <c>rocm_agent_enumerator</c>
    /// reports agents in ROCr order, which is the order HIP assigns device indices in - so
    /// position in this list == HIP device index.
    /// </summary>
    public static List<string> EnumerateGpuAgents()
    {
        if (Capture("rocm_agent_enumerator", ["-t", "GPU"], 30) is { } output)
        {
            var agents = output.Split('\n')
                .Select(l => l.Trim())
                // Older builds emit a bare "gfx000" placeholder for the CPU agent.
                .Where(l => l.StartsWith("gfx", StringComparison.Ordinal) && l != "gfx000")
                .ToList();
            if (agents.Count > 0) return agents;
        }
        return EnumerateViaRocminfo();
    }

    private static List<string> EnumerateViaRocminfo()
    {
        var agents = new List<string>();
        if (Capture("rocminfo", [], 60) is not { } output) return agents;

        string? deviceType = null;
        foreach (var line in output.Split('\n').Select(l => l.Trim()))
        {
            if (line.StartsWith("Device Type:", StringComparison.Ordinal))
                deviceType = line[(line.IndexOf(':') + 1)..].Trim();
            else if (line.StartsWith("Name:", StringComparison.Ordinal) && deviceType == "GPU")
            {
                var name = line[(line.IndexOf(':') + 1)..].Trim();
                if (!name.StartsWith("gfx", StringComparison.Ordinal)) continue;
                agents.Add(name);
                deviceType = null;
            }
        }
        return agents;
    }

    /// <summary>HIP indices of the devices we are willing to train on.</summary>
    public static List<int> SelectDevices(IReadOnlyList<string> agents, bool includeApu = false)
    {
        var discrete = Enumerable.Range(0, agents.Count).Where(i => !ApuGfxTargets.Contains(agents[i])).ToList();
        return discrete.Count > 0 && !includeApu ? discrete : Enumerable.Range(0, agents.Count).ToList();
    }

    /// <summary>Builds the worker's environment. Never clobbers a variable the user already set.</summary>
    /// <param name="current">The starting environment; the process's own when null.</param>
    /// <param name="agents">GPU agents, when already known; enumerated when null.</param>
    public static EnvReport Bootstrap(IReadOnlyDictionary<string, string>? current = null,
        IReadOnlyList<string>? agents = null)
    {
        var report = new EnvReport();
        var env = report.Environment;
        if (current is not null)
            foreach (var (key, value) in current) env[key] = value;
        else
            foreach (System.Collections.DictionaryEntry entry in System.Environment.GetEnvironmentVariables())
                env[(string)entry.Key] = (string?)entry.Value ?? "";

        foreach (var (key, value) in BaseEnv)
        {
            if (env.TryGetValue(key, out var existing)) report.Preexisting[key] = existing;
            else report.Applied[key] = env[key] = value;
        }

        report.Agents = agents?.ToList() ?? EnumerateGpuAgents();

        var visibility = new[] { "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES" }.Where(env.ContainsKey).ToList();
        if (visibility.Count > 0)
        {
            foreach (var key in visibility) report.Preexisting[key] = env[key];
            report.Notes.Add("device visibility was already set in the environment; leaving it alone");
        }
        else
        {
            var selected = SelectDevices(report.Agents);
            if (selected.Count > 0)
            {
                report.Applied["HIP_VISIBLE_DEVICES"] = env["HIP_VISIBLE_DEVICES"] = string.Join(',', selected);
                report.Selected = selected;
                var dropped = Enumerable.Range(0, report.Agents.Count)
                    .Where(i => !selected.Contains(i))
                    .Select(i => $"{i}:{report.Agents[i]}")
                    .ToList();
                if (dropped.Count > 0)
                    report.Notes.Add($"excluded integrated/unsuitable agents: {string.Join(", ", dropped)}");
            }
            else
                report.Notes.Add("could not enumerate any GPU agent (is ROCm installed?)");
        }

        // The first run otherwise spends 3-11 minutes tuning MIOpen kernels and looks hung;
        // persisting the tuned database makes that a one-off.
        if (!env.ContainsKey("MIOPEN_USER_DB_PATH"))
        {
            try
            {
                Directory.CreateDirectory(MiopenDb);
                report.Applied["MIOPEN_USER_DB_PATH"] = env["MIOPEN_USER_DB_PATH"] = MiopenDb;
                env.TryAdd("MIOPEN_CUSTOM_CACHE_DIR", MiopenDb);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
        }

        // The worker speaks JSON lines on stdout; block buffering would hold progress back.
        env["PYTHONUNBUFFERED"] = "1";
        return report;
    }

    /// <summary>A shell snippet with the variables <see cref="Bootstrap"/> would set.</summary>
    public static string FixSnippet(EnvReport report)
    {
        var lines = new List<string> { "# loralei: source this, or export these before running python", "" };
        lines.AddRange(BaseEnv.Select(p => $"export {p.Key}={p.Value}"));
        if (report.Environment.TryGetValue("HIP_VISIBLE_DEVICES", out var visible) && visible.Length > 0)
            lines.Add($"export HIP_VISIBLE_DEVICES={visible}");
        return string.Join('\n', lines) + "\n";
    }

    /// <summary>Runs a diagnostic tool from PATH; null when it is absent, fails or times out.</summary>
    internal static string? Capture(string tool, IReadOnlyList<string> args, int timeoutSeconds, bool requireSuccess = true)
    {
        var exe = Which(tool);
        if (exe is null) return null;
        try
        {
            var info = new ProcessStartInfo(exe)
            {
                RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false,
            };
            foreach (var arg in args) info.ArgumentList.Add(arg);
            using var process = Process.Start(info);
            if (process is null) return null;
            var stdout = process.StandardOutput.ReadToEndAsync();
            _ = process.StandardError.ReadToEndAsync();
            if (!process.WaitForExit(TimeSpan.FromSeconds(timeoutSeconds)))
            {
                try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
                return null;
            }
            return requireSuccess && process.ExitCode != 0 ? null : stdout.GetAwaiter().GetResult();
        }
        catch (Exception e) when (e is System.ComponentModel.Win32Exception or IOException or InvalidOperationException)
        {
            return null;
        }
    }

    public static string? Which(string tool)
    {
        foreach (var entry in (System.Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
        {
            if (entry.Length == 0) continue;
            var candidate = Path.Combine(entry, tool);
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }
}
