using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Input;
using Avalonia.Media;
using Loralei.ImageResizer.Models;
using Loralei.ImageResizer.Services;

namespace Loralei.ImageResizer.Controls;

/// <summary>
/// The manual crop surface: shows the image with a square selection that can be dragged, resized
/// from its corners and zoomed with the wheel. The square is allowed to hang off the edge of the
/// image, and the part that does is painted in the current pad fill so the preview matches the
/// file that will be written.
/// </summary>
public class CropOverlay : Control
{
    /// <summary>Grab radius for the corner handles, in control pixels.</summary>
    private const double HandleHitRadius = 14;

    private const double HandleSize = 11;

    /// <summary>Empty space kept around the image so there is somewhere to drag the square to.</summary>
    private const double ViewportMargin = 0.18;

    // These stay fixed rather than following the application theme, on purpose. A neutral dark
    // surround is what photo editors use so the image's own colours are judged against a constant;
    // a light canvas in light mode would change how every crop looks. Only the window chrome and
    // the button bar around this control follow the theme.
    private static readonly IBrush Backdrop = new SolidColorBrush(Color.FromRgb(0x24, 0x26, 0x2b));
    private static readonly IBrush Dim = new SolidColorBrush(Colors.Black, 0.55);
    private static readonly IBrush HandleFill = Brushes.White;
    private static readonly IBrush CheckerLight = new SolidColorBrush(Color.FromRgb(0xcc, 0xcc, 0xcc));
    private static readonly IBrush CheckerDark = new SolidColorBrush(Color.FromRgb(0x99, 0x99, 0x99));
    private static readonly IPen OutlinePen = new Pen(Brushes.White, 2);
    private static readonly IPen OutlineShadowPen = new Pen(new SolidColorBrush(Colors.Black, 0.6), 4);
    private static readonly IPen ThirdsPen = new Pen(new SolidColorBrush(Colors.White, 0.35), 1);
    private static readonly IPen HandleOutlinePen = new Pen(new SolidColorBrush(Colors.Black, 0.7), 1);

    public static readonly StyledProperty<IImage?> ImageProperty =
        AvaloniaProperty.Register<CropOverlay, IImage?>(nameof(Image));

    public static readonly StyledProperty<int> SourceWidthProperty =
        AvaloniaProperty.Register<CropOverlay, int>(nameof(SourceWidth));

    public static readonly StyledProperty<int> SourceHeightProperty =
        AvaloniaProperty.Register<CropOverlay, int>(nameof(SourceHeight));

    public static readonly StyledProperty<SquareSelection> SelectionProperty =
        AvaloniaProperty.Register<CropOverlay, SquareSelection>(
            nameof(Selection), defaultBindingMode: BindingMode.TwoWay);

    public static readonly StyledProperty<PadFill> PadFillProperty =
        AvaloniaProperty.Register<CropOverlay, PadFill>(nameof(PadFill));

    static CropOverlay()
    {
        AffectsRender<CropOverlay>(
            ImageProperty, SourceWidthProperty, SourceHeightProperty, SelectionProperty, PadFillProperty);
    }

    /// <summary>Which corner is being dragged, or -1 while moving the whole square.</summary>
    private int _dragCorner = -1;

    private bool _dragging;
    private Point _dragOriginSource;
    private SquareSelection _dragOriginSelection;

    /// <summary>The region of source space currently on screen. Grows so a square dragged outwards stays visible.</summary>
    private Rect _viewport;
    private PixelSize _viewportFor;

    public IImage? Image
    {
        get => GetValue(ImageProperty);
        set => SetValue(ImageProperty, value);
    }

    public int SourceWidth
    {
        get => GetValue(SourceWidthProperty);
        set => SetValue(SourceWidthProperty, value);
    }

    public int SourceHeight
    {
        get => GetValue(SourceHeightProperty);
        set => SetValue(SourceHeightProperty, value);
    }

    public SquareSelection Selection
    {
        get => GetValue(SelectionProperty);
        set => SetValue(SelectionProperty, value);
    }

    public PadFill PadFill
    {
        get => GetValue(PadFillProperty);
        set => SetValue(PadFillProperty, value);
    }

    public override void Render(DrawingContext context)
    {
        var bounds = new Rect(Bounds.Size);
        context.FillRectangle(Backdrop, bounds);

        if (SourceWidth <= 0 || SourceHeight <= 0 || bounds.Width <= 0 || bounds.Height <= 0)
            return;

        var view = CreateView(bounds);
        var imageRect = view.ToView(new Rect(0, 0, SourceWidth, SourceHeight));
        var selectionRect = view.ToView(SelectionRect());

        DrawPadFill(context, selectionRect, imageRect);

        if (Image is { } image)
            context.DrawImage(image, imageRect);

        DrawDimming(context, bounds, selectionRect);

        context.DrawRectangle(null, OutlineShadowPen, selectionRect);
        context.DrawRectangle(null, OutlinePen, selectionRect);
        DrawThirds(context, selectionRect);
        DrawHandles(context, selectionRect);
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        if (SourceWidth <= 0 || !e.GetCurrentPoint(this).Properties.IsLeftButtonPressed)
            return;

        var view = CreateView(new Rect(Bounds.Size));
        var position = e.GetPosition(this);
        var selectionRect = view.ToView(SelectionRect());

        _dragCorner = HitCorner(selectionRect, position);
        if (_dragCorner < 0 && !selectionRect.Contains(position))
            return;

        _dragging = true;
        _dragOriginSource = view.ToSource(position);
        _dragOriginSelection = Selection;
        e.Pointer.Capture(this);
        e.Handled = true;
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        if (!_dragging)
        {
            UpdateCursor(e.GetPosition(this));
            return;
        }

        var view = CreateView(new Rect(Bounds.Size));
        var current = view.ToSource(e.GetPosition(this));

        var next = _dragCorner < 0
            ? _dragOriginSelection.MovedTo(
                _dragOriginSelection.X + (current.X - _dragOriginSource.X),
                _dragOriginSelection.Y + (current.Y - _dragOriginSource.Y))
            : ResizeFromCorner(_dragOriginSelection, _dragCorner, current);

        Commit(next);
        e.Handled = true;
    }

    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        base.OnPointerReleased(e);
        if (!_dragging)
            return;

        _dragging = false;
        _dragCorner = -1;
        e.Pointer.Capture(null);
        e.Handled = true;
    }

    protected override void OnPointerWheelChanged(PointerWheelEventArgs e)
    {
        base.OnPointerWheelChanged(e);
        if (SourceWidth <= 0)
            return;

        var view = CreateView(new Rect(Bounds.Size));
        var anchor = view.ToSource(e.GetPosition(this));
        var factor = Math.Pow(0.9, e.Delta.Y);

        Commit(ResizeGeometry.ResizeAbout(
            Selection, Selection.Size * factor, anchor.X, anchor.Y, SourceWidth, SourceHeight));
        e.Handled = true;
    }

    /// <summary>Resizes from <paramref name="corner"/> with the opposite corner pinned, keeping the square square.</summary>
    private SquareSelection ResizeFromCorner(SquareSelection selection, int corner, Point pointer)
    {
        // Corners: 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left.
        var anchorX = corner is 0 or 3 ? selection.Right : selection.X;
        var anchorY = corner is 0 or 1 ? selection.Bottom : selection.Y;

        var size = ResizeGeometry.ClampSize(
            Math.Max(Math.Abs(pointer.X - anchorX), Math.Abs(pointer.Y - anchorY)), SourceWidth, SourceHeight);

        var x = corner is 0 or 3 ? anchorX - size : anchorX;
        var y = corner is 0 or 1 ? anchorY - size : anchorY;
        return new SquareSelection(x, y, size);
    }

    private void Commit(SquareSelection selection) =>
        SetCurrentValue(SelectionProperty, ResizeGeometry.Sanitize(selection, SourceWidth, SourceHeight));

    /// <summary>The selection in control pixels. Exposed so the tests can aim at the corner handles.</summary>
    internal Rect SelectionViewRect() => CreateView(new Rect(Bounds.Size)).ToView(SelectionRect());

    private Rect SelectionRect()
    {
        var selection = Selection;
        return selection.Size <= 0
            ? new Rect(0, 0, SourceWidth, SourceHeight)
            : new Rect(selection.X, selection.Y, selection.Size, selection.Size);
    }

    /// <summary>
    /// Maps source pixels to control pixels. The viewport starts as the image plus a margin and only
    /// ever grows, so dragging the square outwards reveals more space instead of jumping the zoom about.
    /// </summary>
    private ViewTransform CreateView(Rect bounds)
    {
        var size = new PixelSize(SourceWidth, SourceHeight);
        if (_viewportFor != size)
        {
            _viewportFor = size;
            _viewport = BaseViewport();
        }

        _viewport = _viewport.Union(SelectionRect().Inflate(Math.Max(SourceWidth, SourceHeight) * 0.04));

        var scale = Math.Min(bounds.Width / _viewport.Width, bounds.Height / _viewport.Height);
        var offsetX = (bounds.Width - (_viewport.Width * scale)) / 2 - (_viewport.X * scale);
        var offsetY = (bounds.Height - (_viewport.Height * scale)) / 2 - (_viewport.Y * scale);
        return new ViewTransform(scale, offsetX, offsetY);
    }

    private Rect BaseViewport()
    {
        var marginX = SourceWidth * ViewportMargin;
        var marginY = SourceHeight * ViewportMargin;
        return new Rect(-marginX, -marginY, SourceWidth + (2 * marginX), SourceHeight + (2 * marginY));
    }

    private void DrawPadFill(DrawingContext context, Rect selectionRect, Rect imageRect)
    {
        // Only the part of the selection the image does not cover is ever visible, but painting the
        // whole square is simpler and the image is drawn over it immediately afterwards.
        switch (PadFill)
        {
            case PadFill.White:
                context.FillRectangle(Brushes.White, selectionRect);
                break;
            case PadFill.Black:
                context.FillRectangle(Brushes.Black, selectionRect);
                break;
            case PadFill.Transparent:
                DrawChecker(context, selectionRect);
                break;
            case PadFill.Blur:
                if (Image is { } image)
                {
                    // A cheap stand-in for the real Gaussian blur: the image stretched to cover and knocked back.
                    using (context.PushOpacity(0.45))
                        context.DrawImage(image, CoverRect(selectionRect, imageRect));
                    context.FillRectangle(new SolidColorBrush(Colors.Black, 0.25), selectionRect);
                }
                else
                {
                    context.FillRectangle(Dim, selectionRect);
                }

                break;
        }
    }

    /// <summary>Scales the image rect up so it covers <paramref name="target"/> entirely, keeping it centred.</summary>
    private static Rect CoverRect(Rect target, Rect image)
    {
        if (image.Width <= 0 || image.Height <= 0)
            return target;

        var scale = Math.Max(target.Width / image.Width, target.Height / image.Height);
        var width = image.Width * scale;
        var height = image.Height * scale;
        return new Rect(target.Center.X - (width / 2), target.Center.Y - (height / 2), width, height);
    }

    private static void DrawChecker(DrawingContext context, Rect rect)
    {
        const double cell = 12;
        using var clip = context.PushClip(rect);
        context.FillRectangle(CheckerLight, rect);

        for (var y = 0; y * cell < rect.Height; y++)
        {
            for (var x = y % 2; x * cell < rect.Width; x += 2)
            {
                context.FillRectangle(
                    CheckerDark,
                    new Rect(rect.X + (x * cell), rect.Y + (y * cell), cell, cell));
            }
        }
    }

    private static void DrawDimming(DrawingContext context, Rect bounds, Rect selection)
    {
        var clipped = selection.Intersect(bounds);
        context.FillRectangle(Dim, new Rect(bounds.X, bounds.Y, bounds.Width, Math.Max(0, clipped.Y - bounds.Y)));
        context.FillRectangle(Dim, new Rect(bounds.X, clipped.Bottom, bounds.Width, Math.Max(0, bounds.Bottom - clipped.Bottom)));
        context.FillRectangle(Dim, new Rect(bounds.X, clipped.Y, Math.Max(0, clipped.X - bounds.X), clipped.Height));
        context.FillRectangle(Dim, new Rect(clipped.Right, clipped.Y, Math.Max(0, bounds.Right - clipped.Right), clipped.Height));
    }

    private static void DrawThirds(DrawingContext context, Rect rect)
    {
        for (var i = 1; i <= 2; i++)
        {
            var x = rect.X + (rect.Width * i / 3);
            var y = rect.Y + (rect.Height * i / 3);
            context.DrawLine(ThirdsPen, new Point(x, rect.Y), new Point(x, rect.Bottom));
            context.DrawLine(ThirdsPen, new Point(rect.X, y), new Point(rect.Right, y));
        }
    }

    private static void DrawHandles(DrawingContext context, Rect rect)
    {
        foreach (var corner in Corners(rect))
        {
            var handle = new Rect(
                corner.X - (HandleSize / 2), corner.Y - (HandleSize / 2), HandleSize, HandleSize);
            context.DrawRectangle(HandleFill, HandleOutlinePen, handle);
        }
    }

    private static Point[] Corners(Rect rect) =>
    [
        rect.TopLeft,
        rect.TopRight,
        rect.BottomRight,
        rect.BottomLeft,
    ];

    private static int HitCorner(Rect rect, Point point)
    {
        var corners = Corners(rect);
        for (var i = 0; i < corners.Length; i++)
        {
            if (Math.Abs(corners[i].X - point.X) <= HandleHitRadius &&
                Math.Abs(corners[i].Y - point.Y) <= HandleHitRadius)
            {
                return i;
            }
        }

        return -1;
    }

    private void UpdateCursor(Point position)
    {
        if (SourceWidth <= 0)
            return;

        var selectionRect = CreateView(new Rect(Bounds.Size)).ToView(SelectionRect());
        var corner = HitCorner(selectionRect, position);

        Cursor = corner switch
        {
            0 or 2 => new Cursor(StandardCursorType.TopLeftCorner),
            1 or 3 => new Cursor(StandardCursorType.TopRightCorner),
            _ when selectionRect.Contains(position) => new Cursor(StandardCursorType.SizeAll),
            _ => Cursor.Default,
        };
    }

    private readonly record struct ViewTransform(double Scale, double OffsetX, double OffsetY)
    {
        public Rect ToView(Rect source) => new(
            (source.X * Scale) + OffsetX,
            (source.Y * Scale) + OffsetY,
            source.Width * Scale,
            source.Height * Scale);

        public Point ToSource(Point view) => new(
            (view.X - OffsetX) / Scale,
            (view.Y - OffsetY) / Scale);
    }
}
