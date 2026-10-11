using System.Text;

namespace Loralei.Core;

/// <summary>A chunk of child output: one display line, and whether it replaces the previous one.</summary>
/// <param name="Text">ANSI-stripped, trailing whitespace removed.</param>
/// <param name="Transient">
/// True when the line was terminated by a bare CR - i.e. the CLI redrawing a progress bar in
/// place. The log view should overwrite the previous transient line instead of appending,
/// or a 2000-step run becomes 2000 near-identical entries.
/// </param>
public readonly record struct ConsoleLine(string Text, bool Transient);

/// <summary>
/// Splits a raw character stream into display lines, distinguishing CR (redraw) from LF
/// (commit). <see cref="StreamReader.ReadLine"/> cannot be used for this: it treats CR,
/// LF and CRLF as the same terminator, which is exactly the distinction that matters here.
/// </summary>
public sealed class ConsoleStreamParser
{
    private readonly StringBuilder _pending = new();
    private bool _sawCarriageReturn;

    public IEnumerable<ConsoleLine> Feed(string chunk)
    {
        var lines = new List<ConsoleLine>();
        foreach (var ch in chunk)
        {
            if (_sawCarriageReturn)
            {
                _sawCarriageReturn = false;
                // CRLF is a normal newline; a CR followed by anything else was a redraw.
                if (ch == '\n')
                {
                    Emit(lines, transient: false);
                    continue;
                }
                Emit(lines, transient: true);
            }

            switch (ch)
            {
                case '\r':
                    _sawCarriageReturn = true;
                    break;
                case '\n':
                    Emit(lines, transient: false);
                    break;
                default:
                    _pending.Append(ch);
                    break;
            }
        }
        return lines;
    }

    /// <summary>Whatever is buffered when the stream ends, if it is not blank.</summary>
    public ConsoleLine? Flush()
    {
        var lines = new List<ConsoleLine>();
        if (_sawCarriageReturn)
        {
            _sawCarriageReturn = false;
            Emit(lines, transient: true);
        }
        Emit(lines, transient: false);
        return lines.Count > 0 ? lines[^1] : null;
    }

    private void Emit(List<ConsoleLine> sink, bool transient)
    {
        var text = AnsiStripper.Strip(_pending.ToString()).TrimEnd();
        _pending.Clear();
        // The CLI erases a line before redrawing it; the resulting empty frame is noise.
        if (text.Length > 0) sink.Add(new ConsoleLine(text, transient));
    }
}
