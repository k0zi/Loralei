using Xunit;

namespace Loralei.Core.Tests;

public class AnsiStripperTests
{
    private const string Esc = "";

    [Fact]
    public void StripsSgrAndEraseSequences()
    {
        var frame = $"{Esc}[2Kstep 1 loss=0.21 {Esc}[38;2;249;38;114mbar{Esc}[0m {Esc}[35m 67%{Esc}[0m";
        Assert.Equal("step 1 loss=0.21 bar  67%", AnsiStripper.Strip(frame));
    }

    [Fact]
    public void StripsCursorVisibilityToggles()
    {
        Assert.Equal("done", AnsiStripper.Strip($"{Esc}[?25ldone{Esc}[?25h"));
    }

    [Fact]
    public void StripsOscHyperlinks()
    {
        Assert.Equal("avaloniaui", AnsiStripper.Strip($"{Esc}]8;;https://nuget.org/pavaloniaui{Esc}]8;;"));
    }

    [Fact]
    public void LeavesPlainTextAlone()
    {
        Assert.Equal("5 image/caption pairs", AnsiStripper.Strip("5 image/caption pairs"));
    }
}

public class ConsoleStreamParserTests
{
    private const string Esc = "";

    private static List<ConsoleLine> All(params string[] chunks)
    {
        var parser = new ConsoleStreamParser();
        var lines = chunks.SelectMany(parser.Feed).ToList();
        if (parser.Flush() is { } last) lines.Add(last);
        return lines;
    }

    [Fact]
    public void CarriageReturnMarksARedraw()
    {
        var lines = All("train 1/10\rtrain 2/10\rtrain 3/10\n");
        Assert.Equal(3, lines.Count);
        Assert.True(lines[0].Transient);
        Assert.True(lines[1].Transient);
        Assert.False(lines[2].Transient);
        Assert.Equal("train 3/10", lines[2].Text);
    }

    [Fact]
    public void CrLfIsANormalNewlineNotARedraw()
    {
        var lines = All("first\r\nsecond\r\n");
        Assert.All(lines, l => Assert.False(l.Transient));
        Assert.Equal(["first", "second"], lines.Select(l => l.Text));
    }

    [Fact]
    public void SplitsAcrossChunkBoundaries()
    {
        // The CR and the LF of a single CRLF arrive in different reads.
        var lines = All("hel", "lo\r", "\nworld\n");
        Assert.Equal(["hello", "world"], lines.Select(l => l.Text));
        Assert.All(lines, l => Assert.False(l.Transient));
    }

    [Fact]
    public void DropsBlankFramesLeftByEraseLine()
    {
        var lines = All($"\r{Esc}[2K\r{Esc}[2Kwork 50%\n");
        Assert.Single(lines);
        Assert.Equal("work 50%", lines[0].Text);
    }

    [Fact]
    public void FlushEmitsAnUnterminatedTail()
    {
        var lines = All("no newline at the end");
        Assert.Single(lines);
        Assert.Equal("no newline at the end", lines[0].Text);
    }
}
