using Xunit;

namespace Loralei.Core.Tests;

/// <summary>
/// Runs the real CLI - the built <c>loralei</c> beside these tests, and through it the Python
/// worker. Everything the GUI knows about models, datasets and run health is read from these
/// commands, so this is where the two halves are checked against each other.
/// <para>Skipped automatically when no loralei is installed.</para>
/// </summary>
public class LiveCliTests
{
    private static readonly LoraleiPaths Paths = LoraleiPaths.Discover(null, null);
    private static bool Available => Paths.Problem() is null && File.Exists(Paths.Executable);

    private static LoraleiClient Client() => new(Paths);
    private static CancellationToken Timeout(int seconds = 180) =>
        new CancellationTokenSource(TimeSpan.FromSeconds(seconds)).Token;

    [Fact]
    public void DiscoversTheCliAndTheWorkspace()
    {
        Assert.SkipUnless(Available, "loralei is not installed");
        Assert.True(File.Exists(Paths.ProjectConfig), $"no loralei.yaml in {Paths.Workspace}");
        Assert.EndsWith("loralei", Paths.Executable);
    }

    [Fact]
    public void ASavedPathToTheOldPythonScriptIsNotMistakenForTheCli()
    {
        var script = Path.Combine(Path.GetTempPath(), $"loralei-{Guid.NewGuid():N}");
        File.WriteAllText(script, "#!/usr/bin/python3\nfrom loralei.__main__ import main\n");
        try
        {
            var paths = LoraleiPaths.Discover(script, null);
            Assert.NotEqual(script, paths.Executable);
            // A real binary named explicitly is still taken as given.
            Assert.Equal(Paths.Executable, LoraleiPaths.Discover(Paths.Executable, null).Executable);
        }
        finally { File.Delete(script); }
    }

    [Fact]
    public async Task DoctorJsonParsesIntoTheReportShape()
    {
        Assert.SkipUnless(Available, "loralei is not installed");
        using var cli = Loralei.Tests.CliLock.Acquire();

        var (report, result, _) = await Client().DoctorAsync(ct: Timeout());

        Assert.NotNull(report);
        Assert.NotEmpty(report.LoraleiVersion);
        Assert.NotEmpty(report.Checks);
        Assert.Contains(report.Checks, c => c.Name == "driver access");
        Assert.All(report.Checks, c =>
            Assert.Contains(c.Status, new[] { "ok", "warn", "fail", "skip" }));
        // doctor exits non-zero only when a check actually failed.
        Assert.Equal(report.Checks.Any(c => c.Status == "fail") ? 1 : 0, result.ExitCode);
    }

    [Fact]
    public async Task ModelsListingParsesEveryRegisteredModel()
    {
        Assert.SkipUnless(Available, "loralei is not installed");
        using var cli = Loralei.Tests.CliLock.Acquire();

        var listing = await Client().ModelsLsAsync(ct: Timeout());

        Assert.True(listing.Ok, listing.RawText);
        Assert.Equal(LoraleiVocabulary.Models, listing.Models.Select(m => m.Key));
        Assert.All(listing.Models, m => Assert.NotEmpty(m.Status));
        Assert.All(listing.Models, m => Assert.NotEmpty(m.Description));
        Assert.Contains(" free on ", listing.FreeSpace);
    }

    [Fact]
    public async Task DataCheckParsesPairsAndBuckets()
    {
        Assert.SkipUnless(Available, "loralei is not installed");
        using var cli = Loralei.Tests.CliLock.Acquire();

        var dir = Directory.CreateTempSubdirectory("loralei-datacheck");
        try
        {
            Loralei.Tests.Png.Write(Path.Combine(dir.FullName, "a.png"), 64, 64);
            Loralei.Tests.Png.Write(Path.Combine(dir.FullName, "b.png"), 96, 64);
            File.WriteAllText(Path.Combine(dir.FullName, "a.txt"), "a circle");
            File.WriteAllText(Path.Combine(dir.FullName, "b.txt"), "a wide circle");

            var check = await Client().DataCheckAsync(dir.FullName, resolution: 256, ct: Timeout());

            Assert.True(check.Ok, check.RawText);
            Assert.Equal(2, check.Pairs);
            Assert.NotEmpty(check.Buckets);
            Assert.All(check.Buckets, b =>
            {
                Assert.Contains(" x ", b.Bucket);
                Assert.True(int.TryParse(b.Images, out _), $"images cell was '{b.Images}'");
                Assert.True(int.TryParse(b.LatentTokens, out _), $"tokens cell was '{b.LatentTokens}'");
            });
            Assert.Equal(2, check.Buckets.Sum(b => int.Parse(b.Images)));
            Assert.StartsWith("captions:", check.Captions);
        }
        finally { dir.Delete(recursive: true); }
    }

    [Fact]
    public async Task CancellingSendsSigintAndTheChildStops()
    {
        Assert.SkipUnless(Available, "loralei is not installed");
        using var cli = Loralei.Tests.CliLock.Acquire();

        // --thorough runs long benchmarks, which gives us something to interrupt.
        using var cts = new CancellationTokenSource();
        var process = new LoraleiProcess(Paths);
        var started = new TaskCompletionSource();

        var run = process.RunAsync(["doctor", "--thorough"],
            _ => started.TrySetResult(), forceColor: false, ct: cts.Token);

        await Task.WhenAny(started.Task, Task.Delay(TimeSpan.FromSeconds(60)));
        await cts.CancelAsync();

        var result = await run.WaitAsync(TimeSpan.FromSeconds(60));
        Assert.True(result.Cancelled);
    }

    [Fact]
    public void RunStoreReadsRunsWrittenByTheCli()
    {
        Assert.SkipUnless(Available, "loralei is not installed");
        Assert.SkipUnless(Directory.Exists(Paths.OutputRoot()), "no runs yet");

        var store = new RunStore(Paths);
        foreach (var run in store.List())
        {
            Assert.NotEmpty(run.Name);
            Assert.Contains(run.State, new[] { "running", "done", "failed", "unknown" });
            if (run.State == "unknown") continue;

            var manifest = store.ReadManifest(run.Name);
            Assert.NotNull(manifest);
            Assert.Equal(run.Step, manifest.Step);
            Assert.NotEmpty(manifest.AllFields());
        }
    }
}
