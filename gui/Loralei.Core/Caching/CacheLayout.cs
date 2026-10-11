using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Loralei.Core.Config;
using Loralei.Core.Data;

namespace Loralei.Core.Caching;

/// <summary>
/// Where pre-computed latents and text embeddings live.
/// <para>
/// Caching both means the VAE and the text encoder are never loaded in the training process
/// at all - which is what makes a 6B (or 20B) transformer fit on a 16 GiB card. The cache is
/// keyed by everything that would change its contents, so a stale cache is impossible rather
/// than merely unlikely.
/// </para>
/// </summary>
public sealed partial record CacheLayout(string Root, string LatentSignature, string TextSignature)
{
    public const string CacheDirName = ".loralei_cache";

    public static string FallbackRoot => Path.Combine(PathText.Home, ".cache", "loralei", "datasets");

    public string Latents => Path.Combine(Root, "latents", LatentSignature);
    public string Text => Path.Combine(Root, "text", TextSignature);

    public string LatentPath(string key, Bucket bucket) =>
        Path.Combine(Latents, $"{key}_{bucket.Width}x{bucket.Height}.safetensors");

    public string TextPath(string caption) => Path.Combine(Text, $"{CaptionHash(caption)}.safetensors");

    public static string CaptionHash(string caption) =>
        Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(caption)))[..24];

    /// <summary>
    /// The signature hash. Parts are joined in their Python <c>str()</c> spelling - booleans as
    /// <c>True</c>/<c>False</c> - so caches written by the earlier all-Python loralei keep their keys.
    /// </summary>
    private static string Hash(params object[] parts)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var part in parts)
        {
            var text = part switch
            {
                bool b => b ? "True" : "False",
                IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
                _ => part.ToString() ?? "",
            };
            hash.AppendData(Encoding.UTF8.GetBytes(text));
            hash.AppendData([0]);
        }
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }

    /// <summary>Prefer a cache next to the data; fall back to <c>~/.cache</c> for read-only datasets.</summary>
    public static string CacheRoot(string dataset)
    {
        var candidate = Path.Combine(dataset, CacheDirName);
        try
        {
            Directory.CreateDirectory(candidate);
            var probe = Path.Combine(candidate, ".writable");
            File.WriteAllBytes(probe, []);
            File.Delete(probe);
            return candidate;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return Path.Combine(FallbackRoot, Hash(RealPath(dataset))[..16]);
        }
    }

    public static CacheLayout For(string dataset, string modelKey, int resolution, int bucketStep, bool bucketing,
        string vaePath, string textEncoderPath, int maxSequenceLength) =>
        new(
            CacheRoot(dataset),
            // Latents depend on the VAE and on the geometry the images are resized to.
            Hash(modelKey, resolution, bucketStep, bucketing, RealPath(vaePath))[..16],
            // Text embeddings depend only on the encoder and the truncation length - so
            // changing resolution does not throw them away.
            Hash(modelKey, RealPath(textEncoderPath), maxSequenceLength)[..16]);

    /// <summary>The canonical path with links resolved; an absolute spelling when it does not exist.</summary>
    public static string RealPath(string path)
    {
        if (OperatingSystem.IsLinux() || OperatingSystem.IsMacOS())
        {
            var resolved = realpath(path, IntPtr.Zero);
            if (resolved != IntPtr.Zero)
            {
                var text = Marshal.PtrToStringUTF8(resolved);
                free(resolved);
                if (text is not null) return text;
            }
        }
        return Path.GetFullPath(path);
    }

    [LibraryImport("libc", StringMarshalling = StringMarshalling.Utf8, SetLastError = true)]
    private static partial IntPtr realpath(string path, IntPtr buffer);

    [LibraryImport("libc")]
    private static partial void free(IntPtr pointer);
}
