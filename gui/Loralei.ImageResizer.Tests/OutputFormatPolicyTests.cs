using Loralei.ImageResizer.Models;
using Loralei.ImageResizer.Services;
using SkiaSharp;

namespace Loralei.ImageResizer.Tests;

public class OutputFormatPolicyTests
{
    [Theory]
    [InlineData(".jpg")]
    [InlineData(".JPG")]
    [InlineData(".jpeg")]
    [InlineData(".png")]
    [InlineData(".webp")]
    [InlineData(".bmp")]
    [InlineData(".gif")]
    public void Supported_extensions_are_recognised_whatever_the_case(string extension)
    {
        Assert.True(OutputFormatPolicy.IsSupported(extension));
    }

    [Theory]
    [InlineData(".tif")]
    [InlineData(".psd")]
    [InlineData(".txt")]
    [InlineData("")]
    public void Everything_else_is_ignored_by_the_scanner(string extension)
    {
        Assert.False(OutputFormatPolicy.IsSupported(extension));
    }

    [Theory]
    [InlineData(".jpg", SKEncodedImageFormat.Jpeg, ".jpg")]
    [InlineData(".JPEG", SKEncodedImageFormat.Jpeg, ".jpeg")]
    [InlineData(".png", SKEncodedImageFormat.Png, ".png")]
    [InlineData(".webp", SKEncodedImageFormat.Webp, ".webp")]
    [InlineData(".bmp", SKEncodedImageFormat.Png, ".png")] // Skia cannot encode BMP or GIF
    [InlineData(".gif", SKEncodedImageFormat.Png, ".png")]
    public void The_source_format_is_kept_where_Skia_can_write_it(
        string extension, SKEncodedImageFormat expectedFormat, string expectedExtension)
    {
        var format = OutputFormatPolicy.Choose(extension, PadFill.White, 90, needsFill: true);

        Assert.Equal(expectedFormat, format.Format);
        Assert.Equal(expectedExtension, format.Extension);
    }

    [Fact]
    public void Jpeg_carries_the_chosen_quality_through()
    {
        Assert.Equal(72, OutputFormatPolicy.Choose(".jpg", PadFill.Black, 72, needsFill: false).Quality);
    }

    [Theory]
    [InlineData(".jpg")]
    [InlineData(".bmp")]
    [InlineData(".gif")]
    public void A_transparent_fill_forces_png_because_those_formats_have_no_alpha(string extension)
    {
        var format = OutputFormatPolicy.Choose(extension, PadFill.Transparent, 90, needsFill: true);

        Assert.Equal(SKEncodedImageFormat.Png, format.Format);
        Assert.Equal(".png", format.Extension);
    }

    [Fact]
    public void Webp_keeps_its_own_format_for_a_transparent_fill_and_switches_to_lossless()
    {
        var format = OutputFormatPolicy.Choose(".webp", PadFill.Transparent, 80, needsFill: true);

        Assert.Equal(SKEncodedImageFormat.Webp, format.Format);
        Assert.Equal(100, format.Quality);
    }

    [Fact]
    public void A_transparent_fill_that_is_never_used_leaves_the_format_alone()
    {
        // A pure crop covers the whole canvas, so there is no transparency to preserve.
        var format = OutputFormatPolicy.Choose(".jpg", PadFill.Transparent, 90, needsFill: false);

        Assert.Equal(SKEncodedImageFormat.Jpeg, format.Format);
        Assert.Equal(".jpg", format.Extension);
    }
}
