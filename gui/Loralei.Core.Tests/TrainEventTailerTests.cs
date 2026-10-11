using Xunit;

namespace Loralei.Core.Tests;

public class TrainEventTailerTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("loralei-tailer").FullName;
    private string Path => System.IO.Path.Combine(_dir, "train.jsonl");

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    /// <summary>Verbatim lines from a real 12-step run of the patched CLI.</summary>
    private const string Start =
        """{"event": "start", "name": "guitest", "model": "qwen-image", "max_steps": 12, "log_every": 10, "images": 5, "resume": false, "t": 1789914633.819}""";
    private const string Step1 =
        """{"step": 1, "epoch": 0, "loss": 2.569479, "vram_gib": 11.822, "t": 1789914691.516}""";
    private const string Save =
        """{"event": "save", "step": 6, "path": "runs/guitest/checkpoints/guitest-000006.safetensors", "t": 1789914695.441}""";
    private const string Done =
        """{"event": "done", "step": 12, "loss": 2.134308, "vram_gib": 11.887, "seconds": 28.0, "final": "runs/guitest/guitest.safetensors", "t": 1789914701.271}""";

    private void Append(params string[] lines) =>
        File.AppendAllText(Path, string.Join('\n', lines) + "\n");

    [Fact]
    public void ReturnsNothingUntilTheTrainerCreatesTheFile()
    {
        using var tailer = new TrainEventTailer(Path);
        Assert.Empty(tailer.Poll());
    }

    [Fact]
    public void ReadsOnlyWhatIsNewOnEachPoll()
    {
        using var tailer = new TrainEventTailer(Path);
        Append(Start, Step1);

        var first = tailer.Poll();
        Assert.Equal(2, first.Count);
        Assert.Equal("start", first[0].Event);
        Assert.Equal(12, first[0].MaxSteps);
        Assert.True(first[1].IsMetric);
        Assert.Equal(1, first[1].Step);
        Assert.Equal(2.569479, first[1].Loss);
        Assert.Equal(11.822, first[1].VramGib);

        Assert.Empty(tailer.Poll());

        Append(Save, Done);
        var second = tailer.Poll();
        Assert.Equal(["save", "done"], second.Select(e => e.Event));
        Assert.Equal("runs/guitest/guitest.safetensors", second[1].Final);
        Assert.Equal(28.0, second[1].Seconds);
    }

    [Fact]
    public void HoldsBackALineThatIsStillBeingWritten()
    {
        using var tailer = new TrainEventTailer(Path);
        File.AppendAllText(Path, Start + "\n" + """{"step": 2, "los""");

        var first = tailer.Poll();
        Assert.Single(first);
        Assert.Equal("start", first[0].Event);

        // The trainer finishes the line; only now should it be reported.
        File.AppendAllText(Path, "s\": 0.5, \"t\": 1.0}\n");
        var second = tailer.Poll();
        Assert.Single(second);
        Assert.Equal(2, second[0].Step);
        Assert.Equal(0.5, second[0].Loss);
    }

    [Fact]
    public void StartsOverWhenARerunTruncatesTheFile()
    {
        using var tailer = new TrainEventTailer(Path);
        Append(Start, Step1, Save, Done);
        Assert.Equal(4, tailer.Poll().Count);

        // A fresh (non-resume) run of the same name opens the log with "w".
        File.WriteAllText(Path, Start + "\n");
        var afterRerun = tailer.Poll();
        Assert.Single(afterRerun);
        Assert.Equal("start", afterRerun[0].Event);
    }

    [Fact]
    public void SkipsACorruptLineWithoutLosingTheRest()
    {
        using var tailer = new TrainEventTailer(Path);
        Append(Start, "{not json at all", Step1);

        var events = tailer.Poll();
        Assert.Equal(2, events.Count);
        Assert.Equal("start", events[0].Event);
        Assert.Equal(1, events[1].Step);
    }

    [Fact]
    public void MetricLinesAreDistinguishableFromLifecycleLines()
    {
        using var tailer = new TrainEventTailer(Path);
        Append(Start, Step1, Save, Done);

        var events = tailer.Poll();
        Assert.Equal([Step1Index()], events.Where(e => e.IsMetric).Select(e => e.Step));

        static int Step1Index() => 1;
    }
}
