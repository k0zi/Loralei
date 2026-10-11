using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using Loralei.ImageResizer.Controls;
using Loralei.ImageResizer.Models;

namespace Loralei.ImageResizer.Tests;

public class CropOverlayTests
{
    private static (Window Window, CropOverlay Overlay) Open(
        SquareSelection selection, int width = 1000, int height = 1000, PadFill fill = PadFill.Blur)
    {
        var overlay = new CropOverlay
        {
            SourceWidth = width,
            SourceHeight = height,
            Selection = selection,
            PadFill = fill,
        };

        var window = new Window { Content = overlay, Width = 600, Height = 600 };
        window.Show();
        return (window, overlay);
    }

    [AvaloniaTheory]
    [InlineData(PadFill.Blur)]
    [InlineData(PadFill.White)]
    [InlineData(PadFill.Black)]
    [InlineData(PadFill.Transparent)]
    public void It_renders_every_pad_fill_without_an_image(PadFill fill)
    {
        var (window, _) = Open(new SquareSelection(-100, 200, 600), fill: fill);

        using var frame = window.CaptureRenderedFrame();

        Assert.NotNull(frame);
    }

    [AvaloniaFact]
    public void Dragging_inside_the_square_moves_it_without_resizing()
    {
        var (window, overlay) = Open(new SquareSelection(200, 200, 400));
        var start = overlay.SelectionViewRect().Center;

        window.MouseDown(start, MouseButton.Left);
        window.MouseMove(start + new Vector(60, 30));
        window.MouseUp(start + new Vector(60, 30), MouseButton.Left);

        var moved = overlay.Selection;
        Assert.Equal(400, moved.Size, 6);
        Assert.True(moved.X > 200, "dragging right must move the square right");
        Assert.True(moved.Y > 200, "dragging down must move the square down");
        // The drag was twice as far horizontally as vertically, and source space is uniform.
        Assert.Equal(2.0, (moved.X - 200) / (moved.Y - 200), 3);
    }

    [AvaloniaFact]
    public void Dragging_back_to_where_it_started_restores_the_square()
    {
        var (window, overlay) = Open(new SquareSelection(200, 200, 400));
        var start = overlay.SelectionViewRect().Center;

        window.MouseDown(start, MouseButton.Left);
        window.MouseMove(start + new Vector(80, -40));
        window.MouseMove(start);
        window.MouseUp(start, MouseButton.Left);

        Assert.Equal(200, overlay.Selection.X, 3);
        Assert.Equal(200, overlay.Selection.Y, 3);
    }

    [AvaloniaFact]
    public void Dragging_the_top_left_handle_grows_the_square_and_pins_the_opposite_corner()
    {
        var (window, overlay) = Open(new SquareSelection(200, 200, 400));
        var before = overlay.Selection;
        var handle = overlay.SelectionViewRect().TopLeft;

        window.MouseDown(handle, MouseButton.Left);
        window.MouseMove(handle + new Vector(-40, -40));
        window.MouseUp(handle + new Vector(-40, -40), MouseButton.Left);

        var after = overlay.Selection;
        Assert.True(after.Size > before.Size, "dragging the corner outwards must enlarge the square");
        Assert.Equal(before.Right, after.Right, 3);
        Assert.Equal(before.Bottom, after.Bottom, 3);
    }

    [AvaloniaFact]
    public void Dragging_outside_the_square_leaves_it_alone()
    {
        var (window, overlay) = Open(new SquareSelection(400, 400, 100));
        var before = overlay.Selection;

        window.MouseDown(new Point(5, 5), MouseButton.Left);
        window.MouseMove(new Point(120, 120));
        window.MouseUp(new Point(120, 120), MouseButton.Left);

        Assert.Equal(before, overlay.Selection);
    }

    [AvaloniaFact]
    public void The_square_stays_square_when_a_corner_is_dragged_along_one_axis_only()
    {
        var (window, overlay) = Open(new SquareSelection(200, 200, 400));
        var handle = overlay.SelectionViewRect().BottomRight;

        window.MouseDown(handle, MouseButton.Left);
        window.MouseMove(handle + new Vector(70, 0));
        window.MouseUp(handle + new Vector(70, 0), MouseButton.Left);

        var after = overlay.Selection;
        Assert.Equal(200, after.X, 3);
        Assert.Equal(200, after.Y, 3);
        Assert.True(after.Size > 400);
    }
}
