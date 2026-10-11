using Loralei.ImageResizer.Models;
using Loralei.ImageResizer.Services;

namespace Loralei.ImageResizer.Tests;

public class ResizeGeometryTests
{
    [Theory]
    [InlineData(1920, 1080, 420, 0, 1080)]   // landscape: trim the sides
    [InlineData(1080, 1920, 0, 420, 1080)]   // portrait: trim top and bottom
    [InlineData(800, 800, 0, 0, 800)]        // already square
    [InlineData(1001, 1000, 0.5, 0, 1000)]   // odd difference stays sub-pixel rather than rounding
    public void Cover_takes_the_largest_centred_square_inside_the_image(
        int width, int height, double x, double y, double size)
    {
        var selection = ResizeGeometry.Cover(width, height);

        Assert.Equal(x, selection.X, 6);
        Assert.Equal(y, selection.Y, 6);
        Assert.Equal(size, selection.Size, 6);
        Assert.True(selection.IsInside(width, height));
    }

    [Theory]
    [InlineData(1920, 1080, 0, -420, 1920)]
    [InlineData(1080, 1920, -420, 0, 1920)]
    [InlineData(800, 800, 0, 0, 800)]
    public void Contain_takes_the_smallest_centred_square_around_the_image(
        int width, int height, double x, double y, double size)
    {
        var selection = ResizeGeometry.Contain(width, height);

        Assert.Equal(x, selection.X, 6);
        Assert.Equal(y, selection.Y, 6);
        Assert.Equal(size, selection.Size, 6);
    }

    [Fact]
    public void Contain_needs_a_fill_but_cover_does_not()
    {
        Assert.False(ResizeGeometry.NeedsFill(ResizeGeometry.Cover(1920, 1080), 1920, 1080));
        Assert.True(ResizeGeometry.NeedsFill(ResizeGeometry.Contain(1920, 1080), 1920, 1080));
    }

    [Fact]
    public void A_square_source_needs_no_fill_in_either_mode()
    {
        Assert.False(ResizeGeometry.NeedsFill(ResizeGeometry.Cover(800, 800), 800, 800));
        Assert.False(ResizeGeometry.NeedsFill(ResizeGeometry.Contain(800, 800), 800, 800));
    }

    [Theory]
    [InlineData(512)]
    [InlineData(768)]
    [InlineData(1024)]
    [InlineData(1280)]
    [InlineData(2048)]
    public void Cover_maps_the_short_edge_exactly_onto_the_canvas(int target)
    {
        var destination = ResizeGeometry.ImageDestination(ResizeGeometry.Cover(1920, 1080), 1920, 1080, target);

        // The cropped square must land flush against both edges of the canvas, with the long axis overflowing.
        Assert.Equal(0, destination.Y, 6);
        Assert.Equal(target, destination.Height, 6);
        Assert.True(destination.X < 0);

        var overflowLeft = -destination.X;
        var overflowRight = destination.X + destination.Width - target;
        Assert.Equal(overflowLeft, overflowRight, 4);
    }

    [Fact]
    public void Contain_fits_the_long_edge_and_centres_the_short_one()
    {
        var destination = ResizeGeometry.ImageDestination(ResizeGeometry.Contain(1920, 1080), 1920, 1080, 1024);

        Assert.Equal(0, destination.X, 6);
        Assert.Equal(1024, destination.Width, 6);
        Assert.Equal(576, destination.Height, 6);
        Assert.Equal(224, destination.Y, 6);
        // Equal bands above and below.
        Assert.Equal(destination.Y, 1024 - (destination.Y + destination.Height), 6);
    }

    [Fact]
    public void A_one_pixel_image_still_produces_a_valid_destination()
    {
        var selection = ResizeGeometry.Cover(1, 1);
        var destination = ResizeGeometry.ImageDestination(selection, 1, 1, 1024);

        Assert.Equal(1, selection.Size, 6);
        Assert.Equal(0, destination.X, 6);
        Assert.Equal(1024, destination.Width, 6);
        Assert.True(ResizeGeometry.IsUpscaling(selection, 1024));
    }

    [Theory]
    [InlineData(400, 300, 1024, true)]   // short edge 300 < 1024
    [InlineData(4000, 3000, 1024, false)]
    [InlineData(1024, 1024, 1024, false)] // exactly the target is not upscaling
    [InlineData(1023, 1023, 1024, true)]
    public void IsUpscaling_compares_the_selected_region_to_the_output(
        int width, int height, int target, bool expected)
    {
        Assert.Equal(expected, ResizeGeometry.IsUpscaling(ResizeGeometry.Cover(width, height), target));
    }

    [Fact]
    public void ResizeAbout_keeps_the_anchor_point_under_the_cursor()
    {
        var selection = new SquareSelection(100, 100, 400);
        var anchor = new { X = 200.0, Y = 150.0 };

        var zoomed = ResizeGeometry.ResizeAbout(selection, 200, anchor.X, anchor.Y, 1000, 1000);

        // The anchor sat 25% across and 12.5% down the old square; it must still sit there.
        Assert.Equal(0.25, (anchor.X - zoomed.X) / zoomed.Size, 6);
        Assert.Equal(0.125, (anchor.Y - zoomed.Y) / zoomed.Size, 6);
    }

    [Fact]
    public void ResizeAbout_will_not_shrink_below_the_minimum()
    {
        var zoomed = ResizeGeometry.ResizeAbout(new SquareSelection(0, 0, 100), 1, 50, 50, 1000, 1000);

        Assert.Equal(ResizeGeometry.MinSelectionSize, zoomed.Size, 6);
    }

    [Fact]
    public void Sanitize_keeps_a_square_dragged_off_the_image_partly_overlapping_it()
    {
        var dragged = new SquareSelection(-100_000, -100_000, 400);

        var sane = ResizeGeometry.Sanitize(dragged, 1000, 800);

        Assert.Equal(400, sane.Size, 6);
        Assert.True(sane.Right > 0, "the square must still overlap the image horizontally");
        Assert.True(sane.Bottom > 0, "the square must still overlap the image vertically");
    }

    [Fact]
    public void Sanitize_allows_a_manual_square_to_hang_off_the_edge()
    {
        var overhanging = new SquareSelection(-50, -50, 400);

        var sane = ResizeGeometry.Sanitize(overhanging, 1000, 800);

        Assert.Equal(overhanging, sane);
        Assert.True(ResizeGeometry.NeedsFill(sane, 1000, 800));
    }

    [Fact]
    public void Centered_keeps_the_size_and_recentres()
    {
        var centred = ResizeGeometry.Centered(new SquareSelection(0, 0, 400), 1000, 800);

        Assert.Equal(300, centred.X, 6);
        Assert.Equal(200, centred.Y, 6);
        Assert.Equal(400, centred.Size, 6);
    }

    [Theory]
    [InlineData(ResizeMode.Crop, 1080)]
    [InlineData(ResizeMode.Pad, 1920)]
    [InlineData(ResizeMode.Manual, 1080)] // manual has no automatic answer, so it starts from the crop
    public void ForMode_picks_the_matching_automatic_square(ResizeMode mode, double expectedSize)
    {
        Assert.Equal(expectedSize, ResizeGeometry.ForMode(mode, 1920, 1080).Size, 6);
    }
}
