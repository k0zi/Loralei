using System.Collections.Specialized;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;

namespace Loralei.Gui.Controls;

/// <summary>
/// The training loss over time, drawn directly. A chart library would be a third dependency
/// for one line on one page, and the data is already a plain list of doubles.
/// <para>
/// Flow-matching loss is noisy step to step - the raw series is drawn faintly and an
/// exponential moving average on top carries the trend, which is the part worth reading.
/// </para>
/// </summary>
public sealed class LossChart : Control
{
    /// <summary>Above this, the series is decimated so a long run stays cheap to redraw.</summary>
    private const int MaxPoints = 600;

    private const double Smoothing = 0.1;

    public static readonly StyledProperty<System.Collections.IEnumerable?> SeriesProperty =
        AvaloniaProperty.Register<LossChart, System.Collections.IEnumerable?>(nameof(Series));

    public System.Collections.IEnumerable? Series
    {
        get => GetValue(SeriesProperty);
        set => SetValue(SeriesProperty, value);
    }

    private INotifyCollectionChanged? _observed;

    static LossChart()
    {
        AffectsRender<LossChart>(SeriesProperty);
        SeriesProperty.Changed.AddClassHandler<LossChart>((chart, e) => chart.Rebind(e.NewValue));
    }

    protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
    {
        base.OnAttachedToVisualTree(e);
        Rebind(Series);
        // The chart paints itself, so nothing else would repaint it on a Light/Dark switch.
        ActualThemeVariantChanged += OnThemeChanged;
    }

    private void OnThemeChanged(object? sender, EventArgs e) => InvalidateVisual();

    private void Rebind(object? source)
    {
        if (_observed is not null) _observed.CollectionChanged -= OnSeriesChanged;
        _observed = source as INotifyCollectionChanged;
        if (_observed is not null) _observed.CollectionChanged += OnSeriesChanged;
        InvalidateVisual();
    }

    private void OnSeriesChanged(object? sender, NotifyCollectionChangedEventArgs e) => InvalidateVisual();

    protected override void OnDetachedFromVisualTree(VisualTreeAttachmentEventArgs e)
    {
        // The series belongs to the view model, which outlives this control.
        if (_observed is not null) _observed.CollectionChanged -= OnSeriesChanged;
        _observed = null;
        ActualThemeVariantChanged -= OnThemeChanged;
        base.OnDetachedFromVisualTree(e);
    }


    /// <summary>
    /// Resolves one of the chart brushes for the variant this control is actually in.
    /// <para>
    /// The control's own lookup is tried first, then the application's: our palette lives in
    /// a <c>Styles.Resources</c> ThemeDictionary reached through <c>Application.Styles</c>,
    /// which the control-level search does not always walk. Falling back to a hardcoded dark
    /// value would silently paint a dark chart on a light page, so the application lookup
    /// matters more than it looks.
    /// </para>
    /// </summary>
    internal IBrush Brush(string key, string fallback)
    {
        var variant = ActualThemeVariant;
        if (TryGetResource(key, variant, out var found) && found is IBrush local) return local;
        if (Application.Current?.TryGetResource(key, variant, out var shared) == true &&
            shared is IBrush fromApp) return fromApp;
        return new SolidColorBrush(Color.Parse(fallback));
    }

    public override void Render(DrawingContext context)
    {
        var width = Bounds.Width;
        var height = Bounds.Height;
        if (width <= 4 || height <= 4) return;

        context.FillRectangle(Brush("ChartBackgroundBrush", "#15171C"), new Rect(0, 0, width, height));

        var points = Collect();
        if (points.Count < 2)
        {
            var hint = new FormattedText("waiting for the first steps...",
                System.Globalization.CultureInfo.InvariantCulture, FlowDirection.LeftToRight,
                Typeface.Default, 12, Brush("MutedBrush", "#8A8F98"));
            context.DrawText(hint, new Point(10, height / 2 - 8));
            return;
        }

        var min = points.Min();
        var max = points.Max();
        // A flat series would divide by zero; give it a band to sit in the middle of.
        if (max - min < 1e-9) { min -= 0.5; max += 0.5; }
        var pad = (max - min) * 0.08;
        min -= pad;
        max += pad;

        DrawGrid(context, width, height, min, max,
            Brush("ChartGridBrush", "#242830"), Brush("MutedBrush", "#8A8F98"));
        DrawSeries(context, points, width, height, min, max,
            new Pen(Brush("ChartSeriesBrush", "#3D6FA8"), 1));
        DrawSeries(context, Smooth(points), width, height, min, max,
            new Pen(Brush("ChartTrendBrush", "#6FB4F5"), 2));
    }

    private List<double> Collect()
    {
        var all = new List<double>();
        if (Series is not null)
            foreach (var value in Series)
                if (value is double d && !double.IsNaN(d) && !double.IsInfinity(d)) all.Add(d);

        if (all.Count <= MaxPoints) return all;

        // Keep the first and last of each bucket so spikes are not smoothed away by the
        // decimation itself.
        var stride = (double)all.Count / MaxPoints;
        var reduced = new List<double>(MaxPoints);
        for (var i = 0; i < MaxPoints; i++) reduced.Add(all[(int)(i * stride)]);
        reduced[^1] = all[^1];
        return reduced;
    }

    private static List<double> Smooth(IReadOnlyList<double> points)
    {
        var smoothed = new List<double>(points.Count);
        var average = points[0];
        foreach (var value in points)
        {
            average += Smoothing * (value - average);
            smoothed.Add(average);
        }
        return smoothed;
    }

    private static void DrawGrid(DrawingContext context, double width, double height,
        double min, double max, IBrush grid, IBrush label)
    {
        var pen = new Pen(grid, 1);

        for (var i = 0; i <= 4; i++)
        {
            var y = height * i / 4;
            context.DrawLine(pen, new Point(0, y), new Point(width, y));

            var value = max - (max - min) * i / 4;
            var text = new FormattedText(value.ToString("F3",
                    System.Globalization.CultureInfo.InvariantCulture),
                System.Globalization.CultureInfo.InvariantCulture, FlowDirection.LeftToRight,
                Typeface.Default, 10, label);
            context.DrawText(text, new Point(4, Math.Min(y + 2, height - 14)));
        }
    }

    private static void DrawSeries(DrawingContext context, IReadOnlyList<double> points,
        double width, double height, double min, double max, Pen pen)
    {
        if (points.Count < 2) return;

        const double left = 46;
        var plotWidth = Math.Max(1, width - left - 8);
        var geometry = new StreamGeometry();
        using (var sink = geometry.Open())
        {
            for (var i = 0; i < points.Count; i++)
            {
                var x = left + plotWidth * i / (points.Count - 1);
                var y = height - (points[i] - min) / (max - min) * height;
                var point = new Point(x, Math.Clamp(y, 0, height));
                if (i == 0) sink.BeginFigure(point, false);
                else sink.LineTo(point);
            }
            sink.EndFigure(false);
        }
        context.DrawGeometry(null, pen, geometry);
    }
}
