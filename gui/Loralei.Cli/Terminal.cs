using System.Globalization;
using System.Text;

namespace Loralei.Cli;

/// <summary>
/// Everything the CLI prints goes through here, so one rule decides colour and live redraws.
/// <para>
/// Colour and in-place progress are on for a terminal, and for a pipe when
/// <c>FORCE_COLOR</c> is set - which is how the GUI asks for live frames. A frame is written
/// after a bare carriage return, so a reader that distinguishes CR from LF can tell a redraw
/// from a line that is meant to stay. <c>NO_COLOR</c> turns both off.
/// </para>
/// </summary>
public static class Terminal
{
    private static readonly object Gate = new();
    private static bool _liveShown;

    public static bool Rich { get; } = Decide();

    private static bool Decide()
    {
        if (Environment.GetEnvironmentVariable("NO_COLOR") is { Length: > 0 }) return false;
        if (Environment.GetEnvironmentVariable("FORCE_COLOR") is { Length: > 0 } force) return force != "0";
        return !Console.IsOutputRedirected;
    }

    static Terminal()
    {
        Console.OutputEncoding = new UTF8Encoding(false);
    }

    private static string Wrap(string text, string code) => Rich ? $"\x1b[{code}m{text}\x1b[0m" : text;

    public static string Bold(string text) => Wrap(text, "1");
    public static string Dim(string text) => Wrap(text, "2");
    public static string Green(string text) => Wrap(text, "32");
    public static string Yellow(string text) => Wrap(text, "33");
    public static string Red(string text) => Wrap(text, "31");

    public static string F(double value, string format = "F1") => value.ToString(format, CultureInfo.InvariantCulture);

    public static void Line(string text = "")
    {
        lock (Gate)
        {
            ClearLive();
            Console.Out.WriteLine(text);
            Console.Out.Flush();
        }
    }

    /// <summary>Worker chatter - tqdm bars, library warnings - passed through on stderr as it arrives.</summary>
    public static void Passthrough(string chunk)
    {
        lock (Gate)
        {
            ClearLive();
            Console.Error.Write(chunk);
            Console.Error.Flush();
        }
    }

    /// <summary>Draws a status line that the next call replaces. A no-op when output is plain.</summary>
    public static void Live(string text)
    {
        if (!Rich) return;
        lock (Gate)
        {
            Console.Out.Write("\r\x1b[2K" + text);
            Console.Out.Flush();
            _liveShown = true;
        }
    }

    private static void ClearLive()
    {
        if (!_liveShown) return;
        Console.Out.Write("\r\x1b[2K");
        _liveShown = false;
    }

    public static void EndLive()
    {
        lock (Gate)
        {
            ClearLive();
            Console.Out.Flush();
        }
    }

    public static string Bar(long done, long total, int width = 28)
    {
        var fraction = total > 0 ? Math.Clamp((double)done / total, 0, 1) : 0;
        var filled = (int)Math.Round(fraction * width);
        return new string('━', filled) + Dim(new string('─', width - filled));
    }

    /// <summary>Prints rows under a header, each column as wide as its widest cell.</summary>
    /// <param name="rightAlign">Indexes of the numeric columns.</param>
    public static void Table(IReadOnlyList<string> headers, IReadOnlyList<IReadOnlyList<string>> rows,
        params int[] rightAlign)
    {
        // Cells may carry colour codes, which take no room on screen.
        static int Visible(string cell) => Loralei.Core.AnsiStripper.Strip(cell).Length;

        var widths = headers.Select((h, c) => rows.Select(r => Visible(r[c])).Append(h.Length).Max()).ToArray();

        string Render(IReadOnlyList<string> cells, Func<string, string>? style = null)
        {
            var parts = cells.Select((cell, c) =>
            {
                var gap = new string(' ', widths[c] - Visible(cell));
                var styled = style is null ? cell : style(cell);
                return rightAlign.Contains(c) ? gap + styled : styled + gap;
            });
            return string.Join("  ", parts).TrimEnd();
        }

        Line(Render(headers, Bold));
        foreach (var row in rows) Line(Render(row));
    }

    /// <summary>Asks a yes/no question. Anything but an explicit yes - including a closed stdin - is no.</summary>
    public static bool Confirm(string question)
    {
        lock (Gate)
        {
            ClearLive();
            Console.Out.Write($"{question} [y/N]: ");
            Console.Out.Flush();
        }
        var answer = Console.In.ReadLine();
        if (answer is null) Line();
        return answer?.Trim().ToLowerInvariant() is "y" or "yes";
    }
}
