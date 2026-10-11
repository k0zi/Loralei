using Loralei.Core.Caching;
using Loralei.Core.Data;
using Loralei.Tests;
using Xunit;

namespace Loralei.Core.Tests;

public class DatasetTests : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("loralei-ds").FullName;

    public DatasetTests()
    {
        var seed = 0;
        foreach (var (name, w, h) in new[] { ("a", 768, 768), ("b", 1024, 576), ("c", 512, 768) })
        {
            // A different seed per file keeps the duplicate detector quiet.
            Png.Write(Path.Combine(_root, $"{name}.png"), w, h, seed++);
            File.WriteAllText(Path.Combine(_root, $"{name}.txt"), $"a photo of {name}");
        }
    }

    public void Dispose() => Directory.Delete(_root, recursive: true);

    [Fact]
    public void ScanPairsImagesWithCaptions()
    {
        var result = DatasetScanner.Scan(_root);
        Assert.Equal(3, result.Items.Count);
        Assert.Empty(result.Problems);
        Assert.All(result.Items, item => Assert.NotEmpty(item.Caption));
        Assert.Equal((1024, 576), (result.Items[1].Width, result.Items[1].Height));
    }

    [Fact]
    public void ScanReportsOrphansAndMissingCaptions()
    {
        File.WriteAllText(Path.Combine(_root, "orphan.txt"), "no image here");
        File.Delete(Path.Combine(_root, "c.txt"));

        var result = DatasetScanner.Scan(_root);
        var problems = string.Join(' ', result.Problems);
        Assert.Contains("orphan.txt", problems);
        Assert.Contains("c.png", problems);
        // The image with no caption is still trainable, just flagged.
        Assert.Equal(3, result.Items.Count);
    }

    [Fact]
    public void ScanDetectsDuplicates()
    {
        File.Copy(Path.Combine(_root, "a.png"), Path.Combine(_root, "copy.png"));
        File.WriteAllText(Path.Combine(_root, "copy.txt"), "a photo of a");

        var result = DatasetScanner.Scan(_root);
        Assert.Contains(result.Problems, p => p.Contains("duplicate of a.png"));
        Assert.Equal(3, result.Items.Count);
    }

    [Fact]
    public void ScanSkipsWhatItCannotRead()
    {
        File.WriteAllText(Path.Combine(_root, "broken.jpg"), "this is not a jpeg");
        Png.Write(Path.Combine(_root, "tiny.png"), 32, 32, seed: 9);
        File.WriteAllBytes(Path.Combine(_root, "latin1.txt"), [0xE9, 0xE8]);
        Png.Write(Path.Combine(_root, "latin1.png"), 128, 128, seed: 10);

        var result = DatasetScanner.Scan(_root);
        Assert.Equal(3, result.Items.Count);
        Assert.Contains(result.Problems, p => p.StartsWith("broken.jpg: cannot read image"));
        Assert.Contains(result.Problems, p => p == "tiny.png: too small (32x32)");
        Assert.Contains(result.Problems, p => p.StartsWith("latin1.txt: unreadable"));
    }

    [Fact]
    public void ScanDescendsIntoSubdirectoriesInAPinnedOrder()
    {
        var nested = Directory.CreateDirectory(Path.Combine(_root, "a")).FullName;
        Png.Write(Path.Combine(nested, "z.png"), 256, 256, seed: 20);
        File.WriteAllText(Path.Combine(nested, "z.txt"), "nested");

        var names = DatasetScanner.Scan(_root).Items.Select(i => Path.GetRelativePath(_root, i.Image)).ToList();
        // Component-wise: the directory "a" sorts before the file "a.png".
        Assert.Equal([Path.Combine("a", "z.png"), "a.png", "b.png", "c.png"], names);
    }

    [Fact]
    public void TriggerWordIsPrependedOnlyWhereMissing()
    {
        File.WriteAllText(Path.Combine(_root, "a.txt"), "a photo in SKS style");
        var items = DatasetScanner.Scan(_root, triggerWord: "sks").Items;
        Assert.Equal("a photo in SKS style", items[0].Caption);
        Assert.Equal("sks, a photo of b", items[1].Caption);
    }

    [Fact]
    public void FingerprintChangesWithCaptions()
    {
        var before = DatasetScanner.Scan(_root).Fingerprint();
        File.WriteAllText(Path.Combine(_root, "a.txt"), "something else entirely");
        Assert.NotEqual(before, DatasetScanner.Scan(_root).Fingerprint());
    }

    [Fact]
    public void BucketsArePatchAlignedAndBounded()
    {
        var buckets = BucketPlan.Generate(1024, step: 64);
        Assert.Contains(new Bucket(1024, 1024), buckets);
        Assert.All(buckets, b =>
        {
            Assert.Equal(0, b.Width % BucketPlan.LatentPatchDivisor);
            Assert.Equal(0, b.Height % BucketPlan.LatentPatchDivisor);
            Assert.True((long)b.Width * b.Height <= 1024 * 1024 * 1.10);
        });
        Assert.Throws<ArgumentException>(() => BucketPlan.Generate(1024, step: 24));
    }

    [Fact]
    public void PlanAssignsTheClosestAspect()
    {
        var items = DatasetScanner.Scan(_root).Items;
        var plan = BucketPlan.Build(items, resolution: 512, step: 64);
        var all = BucketPlan.Generate(512, 64);
        foreach (var item in items)
        {
            var chosen = Math.Abs(plan.BucketFor(item).Aspect - item.Aspect);
            Assert.True(chosen <= all.Min(b => Math.Abs(b.Aspect - item.Aspect)) + 1e-9);
        }
        Assert.Equal(3, plan.ByBucket.Sum(b => b.Items.Count));
    }

    [Fact]
    public void AModelWithADeeperVaeGetsACoarserGridAndFewerTokens()
    {
        var items = DatasetScanner.Scan(_root).Items;
        var plan = BucketPlan.Build(items, resolution: 512, divisor: 32);
        Assert.Equal(32, plan.Divisor);
        Assert.All(plan.Buckets, b => Assert.True(b.Width % 32 == 0 && b.Height % 32 == 0));
        // The same pixels are a quarter of the tokens.
        Assert.Equal(256, plan.Tokens(new Bucket(512, 512)));
        Assert.Throws<ArgumentException>(() => BucketPlan.Generate(512, step: 48, divisor: 32));
    }

    [Fact]
    public void BucketingCanBeDisabled()
    {
        var plan = BucketPlan.Build(DatasetScanner.Scan(_root).Items, resolution: 512, enabled: false);
        Assert.Equal([new Bucket(512, 512)], plan.Buckets);
        Assert.Equal(1024, plan.Tokens(new Bucket(512, 512)));
    }

    [Fact]
    public void CacheKeysFollowWhatWouldChangeTheContents()
    {
        var vae = Directory.CreateDirectory(Path.Combine(_root, "m", "vae")).FullName;
        var te = Directory.CreateDirectory(Path.Combine(_root, "m", "text_encoder")).FullName;
        CacheLayout Layout(int resolution = 1024, string model = "krea-2-raw", int tokens = 512) =>
            CacheLayout.For(_root, model, resolution, 64, true, vae, te, tokens);

        var baseline = Layout();
        Assert.Equal(Path.Combine(_root, CacheLayout.CacheDirName), baseline.Root);
        Assert.Equal(baseline, Layout());

        // Resolution changes the latents but must not throw the text embeddings away.
        Assert.NotEqual(baseline.LatentSignature, Layout(resolution: 512).LatentSignature);
        Assert.Equal(baseline.TextSignature, Layout(resolution: 512).TextSignature);
        Assert.NotEqual(baseline.TextSignature, Layout(tokens: 1024).TextSignature);
        Assert.NotEqual(baseline.LatentSignature, Layout(model: "qwen-image").LatentSignature);

        Assert.EndsWith("abc_640x384.safetensors", baseline.LatentPath("abc", new Bucket(640, 384)));
        // sha256("")[:24] - the worker derives the same file name on its side.
        Assert.Equal("e3b0c44298fc1c149afbf4c8", CacheLayout.CaptionHash(""));
    }
}

public class ImageHeaderTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("loralei-img").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private (int, int) Read(string name, params byte[][] parts)
    {
        var path = Path.Combine(_dir, name);
        File.WriteAllBytes(path, parts.SelectMany(p => p).ToArray());
        return ImageHeader.Read(path);
    }

    private static byte[] Ascii(string text) => System.Text.Encoding.ASCII.GetBytes(text);

    [Fact]
    public void ReadsPng()
    {
        var path = Path.Combine(_dir, "x.png");
        Png.Write(path, 320, 200);
        Assert.Equal((320, 200), ImageHeader.Read(path));
    }

    [Fact]
    public void ReadsJpegPastItsMetadataSegments()
    {
        // SOI, an APP0 segment to skip, then a progressive frame header: 600 wide, 400 high.
        Assert.Equal((600, 400), Read("x.jpg",
            [0xFF, 0xD8],
            [0xFF, 0xE0, 0x00, 0x04, 0x4A, 0x46],
            [0xFF, 0xC2, 0x00, 0x11, 0x08, 0x01, 0x90, 0x02, 0x58, 0x03]));
    }

    [Fact]
    public void ReadsAllThreeWebPLayouts()
    {
        byte[] Riff(string chunk, params byte[] data) =>
            [.. Ascii("RIFF"), 0, 0, 0, 0, .. Ascii("WEBP"), .. Ascii(chunk), 0, 0, 0, 0, .. data];

        // Extended: 24-bit canvas size, stored minus one.
        Assert.Equal((800, 600), Read("x.webp", Riff("VP8X", 0, 0, 0, 0, 0x1F, 0x03, 0, 0x57, 0x02, 0)));
        // Lossy: frame tag, start code, then 14-bit width and height.
        Assert.Equal((640, 480), Read("l.webp", Riff("VP8 ", 0, 0, 0, 0x9D, 0x01, 0x2A, 0x80, 0x02, 0xE0, 0x01)));
        // Lossless: 14 bits of width-1 then 14 bits of height-1, packed.
        Assert.Equal((512, 768), Read("ll.webp", Riff("VP8L", 0x2F, 0xFF, 0xC1, 0xBF, 0x00, 0, 0, 0, 0, 0)));
    }

    [Fact]
    public void ReadsBmp()
    {
        var header = new byte[30];
        header[0] = (byte)'B'; header[1] = (byte)'M';
        header[14] = 40;
        BitConverter.TryWriteBytes(header.AsSpan(18), 640);
        BitConverter.TryWriteBytes(header.AsSpan(22), -480); // top-down rows are stored negative
        Assert.Equal((640, 480), Read("x.bmp", header));
    }

    [Fact]
    public void ReadsAvifFromItsImageSizeProperty()
    {
        byte[] Box(string type, params byte[][] body)
        {
            var payload = body.SelectMany(b => b).ToArray();
            var size = payload.Length + 8;
            return [(byte)(size >> 24), (byte)(size >> 16), (byte)(size >> 8), (byte)size, .. Ascii(type), .. payload];
        }
        byte[] Ispe(int w, int h) => Box("ispe", [0, 0, 0, 0],
            [(byte)(w >> 24), (byte)(w >> 16), (byte)(w >> 8), (byte)w],
            [(byte)(h >> 24), (byte)(h >> 16), (byte)(h >> 8), (byte)h]);

        // A thumbnail-sized property first: the image is the largest one, not the first.
        Assert.Equal((800, 600), Read("x.avif",
            Box("ftyp", Ascii("avif"), [0, 0, 0, 0]),
            Box("meta", [0, 0, 0, 0], Box("iprp", Box("ipco", Ispe(160, 120), Ispe(800, 600))))));
    }

    [Fact]
    public void RefusesWhatItDoesNotKnow()
    {
        Assert.Throws<InvalidDataException>(() => Read("x.jxl", [0xFF, 0x0A, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
        Assert.Throws<InvalidDataException>(() => Read("cut.jpg", [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04]));
        Assert.Throws<InvalidDataException>(() => Read("empty.png"));
    }
}
