using System.Globalization;
using Loralei.Core.Config;

namespace Loralei.Core.Models;

public sealed class NotEnoughSpaceException(string message) : Exception(message);

/// <summary>Something large enough to be worth offering for deletion.</summary>
public sealed record CacheEntry(string Path, string Name, long Bytes)
{
    public double Gib => Bytes / (double)Disk.Gib;
}

/// <summary>
/// Free-space guards and cache accounting. A 40 GB model pull onto a filesystem with 36 GB
/// free fills the root partition to 100%, so every download goes through
/// <see cref="EnsureSpace"/> first.
/// </summary>
public static class Disk
{
    public const long Gib = 1024L * 1024 * 1024;

    /// <summary>Refuse to download unless free >= needed * this.</summary>
    public const double SafetyFactor = 1.15;

    public static string F1(double value) => value.ToString("F1", CultureInfo.InvariantCulture);

    /// <summary>Free bytes on the filesystem holding <paramref name="path"/>, which need not exist yet.</summary>
    public static long FreeBytes(string path)
    {
        var target = Path.GetFullPath(path);
        while (!Directory.Exists(target) && !File.Exists(target))
            target = Path.GetDirectoryName(target) ?? "/";
        return new DriveInfo(target).AvailableFreeSpace;
    }

    public static void EnsureSpace(string path, long neededBytes, string what = "download")
    {
        var free = FreeBytes(path);
        var required = (long)(neededBytes * SafetyFactor);
        if (free >= required) return;
        throw new NotEnoughSpaceException(
            $"{what} needs ~{F1(neededBytes / (double)Gib)} GiB (+15% headroom = " +
            $"{F1(required / (double)Gib)} GiB) on {path}, but only {F1(free / (double)Gib)} GiB is free.\n" +
            "Run `loralei models gc --dry-run` to see what can be reclaimed.");
    }

    /// <summary>
    /// Bytes actually occupied. Symlinks are not followed and not counted: the Hugging Face
    /// cache is a tree of links into <c>blobs/</c>, and following them would count every
    /// file once per snapshot.
    /// </summary>
    public static long DirSize(string path)
    {
        long total = 0;
        var pending = new Stack<string>();
        pending.Push(path);
        while (pending.Count > 0)
        {
            IEnumerable<FileSystemInfo> entries;
            try { entries = new DirectoryInfo(pending.Pop()).EnumerateFileSystemInfos(); }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException) { continue; }

            foreach (var entry in entries)
            {
                try
                {
                    if (entry.LinkTarget is not null) continue;
                    if (entry is DirectoryInfo) pending.Push(entry.FullName);
                    else if (entry is FileInfo file) total += file.Length;
                }
                catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
            }
        }
        return total;
    }

    /// <summary>The default Hugging Face cache, honouring <c>HF_HOME</c> and <c>HUGGINGFACE_HUB_CACHE</c>.</summary>
    public static string HfHubDir()
    {
        if (Environment.GetEnvironmentVariable("HF_HOME") is { Length: > 0 } home)
            return Path.Combine(home, "hub");
        if (Environment.GetEnvironmentVariable("HUGGINGFACE_HUB_CACHE") is { Length: > 0 } hub)
            return hub;
        return Path.Combine(PathText.Home, ".cache", "huggingface", "hub");
    }

    /// <summary>Everything big enough to be worth offering for deletion, largest first.</summary>
    public static List<CacheEntry> Reclaimable(long minBytes = Gib)
    {
        var candidates = new List<CacheEntry>();

        var hub = HfHubDir();
        if (Directory.Exists(hub))
        {
            foreach (var entry in Directory.EnumerateDirectories(hub))
            {
                var name = Path.GetFileName(entry);
                if (!name.StartsWith("models--", StringComparison.Ordinal) &&
                    !name.StartsWith("datasets--", StringComparison.Ordinal)) continue;
                var size = DirSize(entry);
                if (size < minBytes) continue;
                var pretty = name[(name.IndexOf("--", StringComparison.Ordinal) + 2)..].Replace("--", "/");
                candidates.Add(new CacheEntry(entry, $"HF cache: {pretty}", size));
            }
        }

        foreach (var (name, path) in new[]
                 {
                     ("pip cache", Path.Combine(PathText.Home, ".cache", "pip")),
                     ("uv cache", Path.Combine(PathText.Home, ".cache", "uv")),
                     ("ROCm comgr cache", Path.Combine(PathText.Home, ".cache", "comgr")),
                 })
        {
            if (!Directory.Exists(path)) continue;
            var size = DirSize(path);
            if (size >= minBytes) candidates.Add(new CacheEntry(path, name, size));
        }

        return candidates.OrderByDescending(c => c.Bytes).ToList();
    }
}
