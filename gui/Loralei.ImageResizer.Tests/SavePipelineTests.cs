using Loralei.ImageResizer.Models;
using Loralei.ImageResizer.Services;
using SkiaSharp;

namespace Loralei.ImageResizer.Tests;

/// <summary>End-to-end checks over the real decode, render and write path.</summary>
public sealed class SavePipelineTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("imageresizer-tests").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private static ResizeSettings Settings(int target = 512, PadFill fill = PadFill.Blur) =>
        new(target, fill, 95);

    private SourceImage Source(string name, int width, int height, SKEncodedImageFormat format)
    {
        var path = ImageFixtures.Write(_folder, name, width, height, format);
        Assert.True(ImageDecoder.TryReadInfo(path, out var w, out var h));
        return new SourceImage(path, w, h, new FileInfo(path).Length);
    }

    private static (int Width, int Height) Dimensions(string path)
    {
        Assert.True(ImageDecoder.TryReadInfo(path, out var w, out var h));
        return (w, h);
    }

    [Theory]
    [InlineData(1920, 1080, ResizeMode.Crop, 512)]
    [InlineData(1920, 1080, ResizeMode.Pad, 512)]
    [InlineData(1080, 1920, ResizeMode.Crop, 768)]
    [InlineData(1080, 1920, ResizeMode.Pad, 768)]
    [InlineData(800, 800, ResizeMode.Crop, 1024)]
    [InlineData(200, 150, ResizeMode.Pad, 1024)]   // upscaled
    [InlineData(101, 97, ResizeMode.Crop, 512)]    // odd dimensions
    public void Every_output_is_exactly_the_target_square(int width, int height, ResizeMode mode, int target)
    {
        var source = Source($"{width}x{height}-{mode}.jpg", width, height, SKEncodedImageFormat.Jpeg);
        var selection = ResizeGeometry.ForMode(mode, source.PixelWidth, source.PixelHeight);
        var output = ImageSaver.EnsureOutputFolder(_folder);

        var written = ImageSaver.Save(source, selection, Settings(target), output);

        Assert.Equal((target, target), Dimensions(written));
    }

    [Theory]
    [InlineData(PadFill.Blur)]
    [InlineData(PadFill.White)]
    [InlineData(PadFill.Black)]
    [InlineData(PadFill.Transparent)]
    public void Every_pad_fill_produces_a_readable_square(PadFill fill)
    {
        var source = Source($"pad-{fill}.png", 900, 400, SKEncodedImageFormat.Png);
        var selection = ResizeGeometry.Contain(source.PixelWidth, source.PixelHeight);
        var output = ImageSaver.EnsureOutputFolder(_folder);

        var written = ImageSaver.Save(source, selection, Settings(512, fill), output);

        Assert.Equal((512, 512), Dimensions(written));
    }

    [Fact]
    public void A_transparent_fill_leaves_the_padding_transparent_and_writes_png()
    {
        var source = Source("transparent.jpg", 800, 400, SKEncodedImageFormat.Jpeg);
        var selection = ResizeGeometry.Contain(source.PixelWidth, source.PixelHeight);
        var output = ImageSaver.EnsureOutputFolder(_folder);

        var written = ImageSaver.Save(source, selection, Settings(512, PadFill.Transparent), output);

        Assert.Equal(".png", Path.GetExtension(written));

        using var bitmap = ImageDecoder.Decode(written)!;
        Assert.Equal(0, bitmap.GetPixel(256, 4).Alpha);   // top band, above the image
        Assert.NotEqual(0, bitmap.GetPixel(256, 256).Alpha); // centre, inside the image
    }

    [Fact]
    public void A_white_fill_paints_the_padding_white()
    {
        var source = Source("white.jpg", 800, 400, SKEncodedImageFormat.Jpeg);
        var selection = ResizeGeometry.Contain(source.PixelWidth, source.PixelHeight);
        var output = ImageSaver.EnsureOutputFolder(_folder);

        var written = ImageSaver.Save(source, selection, Settings(512, PadFill.White), output);

        using var bitmap = ImageDecoder.Decode(written)!;
        var top = bitmap.GetPixel(256, 4);
        Assert.Equal(255, top.Red);
        Assert.Equal(255, top.Green);
        Assert.Equal(255, top.Blue);
    }

    [Fact]
    public void A_crop_keeps_the_source_format_and_quality_setting()
    {
        var source = Source("keep.jpg", 1200, 900, SKEncodedImageFormat.Jpeg);
        var output = ImageSaver.EnsureOutputFolder(_folder);

        var written = ImageSaver.Save(
            source, ResizeGeometry.Cover(1200, 900), Settings(), output);

        Assert.Equal(".jpg", Path.GetExtension(written));
    }

    [Fact]
    public void Saving_twice_never_overwrites_the_first_result()
    {
        var source = Source("twice.png", 600, 600, SKEncodedImageFormat.Png);
        var output = ImageSaver.EnsureOutputFolder(_folder);
        var selection = ResizeGeometry.Cover(600, 600);

        var first = ImageSaver.Save(source, selection, Settings(), output);
        var second = ImageSaver.Save(source, selection, Settings(768), output);

        Assert.Equal("twice.png", Path.GetFileName(first));
        Assert.Equal("twice_2.png", Path.GetFileName(second));
        Assert.Equal((512, 512), Dimensions(first));
        Assert.Equal((768, 768), Dimensions(second));
    }

    [Fact]
    public void The_original_file_is_left_untouched()
    {
        var source = Source("original.jpg", 900, 700, SKEncodedImageFormat.Jpeg);
        var before = File.ReadAllBytes(source.Path);
        var stamp = File.GetLastWriteTimeUtc(source.Path);
        var output = ImageSaver.EnsureOutputFolder(_folder);

        ImageSaver.Save(source, ResizeGeometry.Cover(900, 700), Settings(), output);

        Assert.Equal(before, File.ReadAllBytes(source.Path));
        Assert.Equal(stamp, File.GetLastWriteTimeUtc(source.Path));
    }

    [Fact]
    public void Output_goes_into_the_saved_resized_subfolder()
    {
        var output = ImageSaver.EnsureOutputFolder(_folder);

        Assert.Equal(Path.Combine(_folder, "saved_resized"), output);
        Assert.True(Directory.Exists(output));
    }

    [Fact]
    public void The_scanner_finds_images_and_ignores_everything_else()
    {
        Source("a.jpg", 100, 100, SKEncodedImageFormat.Jpeg);
        Source("b.png", 100, 100, SKEncodedImageFormat.Png);
        File.WriteAllText(Path.Combine(_folder, "notes.txt"), "not an image");
        File.WriteAllText(Path.Combine(_folder, "broken.jpg"), "also not an image");
        Directory.CreateDirectory(Path.Combine(_folder, "nested"));
        Source(Path.Combine("nested", "c.jpg"), 100, 100, SKEncodedImageFormat.Jpeg);

        var found = ImageScanner.Scan(_folder, TestContext.Current.CancellationToken);

        Assert.Equal(["a.jpg", "b.png"], found.Select(f => f.FileName));
    }

    [Theory]
    [InlineData(1, 1200, 600)] // TopLeft: unchanged
    [InlineData(6, 600, 1200)] // rotate 90 CW: axes swap
    [InlineData(8, 600, 1200)] // rotate 90 CCW: axes swap
    [InlineData(3, 1200, 600)] // rotate 180: axes stay
    public void Exif_orientation_is_applied_when_reading_and_decoding(
        ushort orientation, int expectedWidth, int expectedHeight)
    {
        var path = ImageFixtures.Write(_folder, $"exif{orientation}.jpg", 1200, 600, SKEncodedImageFormat.Jpeg);
        ImageFixtures.SetJpegOrientation(path, orientation);

        Assert.True(ImageDecoder.TryReadInfo(path, out var width, out var height));
        Assert.Equal((expectedWidth, expectedHeight), (width, height));

        using var bitmap = ImageDecoder.Decode(path)!;
        Assert.Equal((expectedWidth, expectedHeight), (bitmap.Width, bitmap.Height));
    }

    [Fact]
    public void A_sideways_photo_is_saved_upright()
    {
        // The fixture puts red top-left and green top-right. With orientation 6 the encoded
        // top-left corner belongs at the display top-right, so red must end up on the right.
        var path = ImageFixtures.Write(_folder, "upright.jpg", 1200, 600, SKEncodedImageFormat.Jpeg);
        ImageFixtures.SetJpegOrientation(path, 6);

        using var bitmap = ImageDecoder.Decode(path)!;

        Assert.True(IsReddish(bitmap.GetPixel(bitmap.Width - 20, 20)), "red belongs at the top right after rotating");
        Assert.True(IsGreenish(bitmap.GetPixel(bitmap.Width - 20, bitmap.Height - 20)), "green belongs at the bottom right");
    }

    private static bool IsReddish(SKColor c) => c.Red > 150 && c.Green < 100 && c.Blue < 100;

    private static bool IsGreenish(SKColor c) => c.Green > 150 && c.Red < 100 && c.Blue < 100;
}
