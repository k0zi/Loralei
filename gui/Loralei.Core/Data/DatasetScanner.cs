using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace Loralei.Core.Data;

/// <summary>One usable image and the caption that goes with it.</summary>
public sealed record DatasetItem(string Image, string? CaptionPath, string Caption, int Width, int Height,
    string ContentHash)
{
    public double Aspect => (double)Width / Height;

    /// <summary>Stable identifier used for cache filenames.</summary>
    public string Key => ContentHash[..16];

    public string ImageName => Path.GetFileName(Image);
}

public sealed class ScanResult(string root)
{
    public string Root { get; } = root;
    public List<DatasetItem> Items { get; } = [];
    public List<string> Problems { get; } = [];

    public string CaptionStats()
    {
        if (Items.Count == 0) return "no captions";
        var words = Items.Select(i => i.Caption.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length).ToList();
        var empty = Items.Count(i => i.Caption.Trim().Length == 0);
        var avg = words.Average().ToString("F1", CultureInfo.InvariantCulture);
        return $"{avg} words avg, {words.Min()}-{words.Max()} range" + (empty > 0 ? $", {empty} empty" : "");
    }

    /// <summary>Content hash of the whole dataset - recorded in the run manifest.</summary>
    public string Fingerprint()
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var item in Items.OrderBy(i => i.ImageName, StringComparer.Ordinal))
        {
            hash.AppendData(Encoding.UTF8.GetBytes(item.ContentHash));
            hash.AppendData(Encoding.UTF8.GetBytes(item.Caption));
        }
        return Convert.ToHexStringLower(hash.GetHashAndReset())[..32];
    }
}

/// <summary>
/// Turns a directory of <c>a.jpg</c> / <c>a.txt</c> pairs into a validated item list.
/// <para>
/// Everything that can go wrong with a hand-assembled dataset - an image with no caption, a
/// caption with no image, an unreadable file, the same photo twice under two names - is
/// reported here rather than surfacing as a confusing failure 2000 steps into training.
/// </para>
/// </summary>
public static class DatasetScanner
{
    public static readonly IReadOnlySet<string> ImageSuffixes =
        new HashSet<string> { ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif", ".jxl" };

    private static readonly UTF8Encoding StrictUtf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    /// <param name="progress">Called with (images examined, images found) as the scan hashes each file.</param>
    public static ScanResult Scan(string root, string captionExtension = ".txt", bool recursive = true,
        string? triggerWord = null, Action<int, int>? progress = null)
    {
        var result = new ScanResult(root);
        if (!Directory.Exists(root))
        {
            result.Problems.Add($"{root} is not a directory");
            return result;
        }

        var files = Files(root, recursive);
        files.Sort(ComparePaths);
        var images = files.Where(p => ImageSuffixes.Contains(Suffix(p).ToLowerInvariant())).ToList();
        var captions = files.Where(p => Suffix(p) == captionExtension).ToList();

        var usedCaptions = new HashSet<string>(StringComparer.Ordinal);
        var byHash = new Dictionary<string, string>(StringComparer.Ordinal);
        var examined = 0;

        foreach (var image in images)
        {
            progress?.Invoke(examined++, images.Count);

            string? captionPath = WithSuffix(image, captionExtension);
            var caption = "";
            if (File.Exists(captionPath))
            {
                usedCaptions.Add(captionPath);
                try
                {
                    caption = StrictUtf8.GetString(File.ReadAllBytes(captionPath)).Trim();
                }
                catch (Exception e) when (e is IOException or UnauthorizedAccessException or DecoderFallbackException)
                {
                    result.Problems.Add($"{Path.GetFileName(captionPath)}: unreadable ({e.Message})");
                    continue;
                }
                if (caption.Length == 0) result.Problems.Add($"{Path.GetFileName(captionPath)}: empty caption");
            }
            else
            {
                captionPath = null;
                result.Problems.Add($"{Path.GetFileName(image)}: no {captionExtension} caption");
            }

            int width, height;
            string digest;
            try
            {
                (width, height) = ImageHeader.Read(image);
                if (width < 64 || height < 64)
                {
                    result.Problems.Add($"{Path.GetFileName(image)}: too small ({width}x{height})");
                    continue;
                }
                digest = HashFile(image);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException or InvalidDataException)
            {
                result.Problems.Add($"{Path.GetFileName(image)}: cannot read image ({e.Message})");
                continue;
            }

            if (byHash.TryGetValue(digest, out var original))
            {
                result.Problems.Add($"{Path.GetFileName(image)}: duplicate of {Path.GetFileName(original)}");
                continue;
            }
            byHash[digest] = image;

            if (!string.IsNullOrEmpty(triggerWord) &&
                !caption.Contains(triggerWord, StringComparison.OrdinalIgnoreCase))
                caption = caption.Length > 0 ? $"{triggerWord}, {caption}" : triggerWord;

            result.Items.Add(new DatasetItem(image, captionPath, caption, width, height, digest));
        }
        progress?.Invoke(images.Count, images.Count);

        foreach (var orphan in captions.Where(c => !usedCaptions.Contains(c)))
            result.Problems.Add($"{Path.GetFileName(orphan)}: caption with no matching image");

        if (result.Items.Count > 1)
        {
            var first = result.Items[0];
            if (first.Width != first.Height &&
                result.Items.All(i => i.Width == first.Width && i.Height == first.Height))
                result.Problems.Add(
                    $"note: every image is {first.Width}x{first.Height}; bucketing will produce a single bucket");
        }
        return result;
    }

    /// <summary>Regular files under the root. Linked directories are not descended into; linked files count.</summary>
    private static List<string> Files(string root, bool recursive)
    {
        var found = new List<string>();
        var pending = new Stack<string>();
        pending.Push(root);
        while (pending.Count > 0)
        {
            IEnumerable<FileSystemInfo> entries;
            try { entries = new DirectoryInfo(pending.Pop()).EnumerateFileSystemInfos(); }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException) { continue; }

            foreach (var entry in entries)
            {
                if (entry is DirectoryInfo)
                {
                    if (recursive && entry.LinkTarget is null) pending.Push(entry.FullName);
                }
                else if (Config.PathText.IsFile(entry)) found.Add(Path.Combine(root, Path.GetRelativePath(root, entry.FullName)));
            }
        }
        return found;
    }

    /// <summary>
    /// Component-wise ordinal order - <c>a/z.png</c> before <c>a.png</c> is not what a plain
    /// string compare gives. The order decides which of two duplicates is kept, so it is pinned.
    /// </summary>
    private static int ComparePaths(string left, string right)
    {
        var a = left.Split(Path.DirectorySeparatorChar);
        var b = right.Split(Path.DirectorySeparatorChar);
        for (var i = 0; i < Math.Min(a.Length, b.Length); i++)
        {
            var order = string.CompareOrdinal(a[i], b[i]);
            if (order != 0) return order;
        }
        return a.Length.CompareTo(b.Length);
    }

    /// <summary>The last extension, where a leading dot alone (<c>.hidden</c>) is a name, not a suffix.</summary>
    private static string Suffix(string path)
    {
        var name = Path.GetFileName(path);
        var dot = name.LastIndexOf('.');
        return dot <= 0 ? "" : name[dot..];
    }

    private static string WithSuffix(string path, string suffix) =>
        path[..^Suffix(path).Length] + suffix;

    private static string HashFile(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}
