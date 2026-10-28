using Loralei.Core.Worker;
using Xunit;

namespace Loralei.Core.Tests;

/// <summary>
/// The install is planned as plain argv and run step by step. Nothing here downloads: the
/// plan is checked as data, and the runner against a stand-in tool.
/// </summary>
public class WorkerInstallerTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("loralei-install").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private string Checkout()
    {
        File.WriteAllText(Path.Combine(_dir, "pyproject.toml"), "[project]\nname = \"loralei\"\n");
        Directory.CreateDirectory(Path.Combine(_dir, "src", "loralei"));
        return _dir;
    }

    private string Script(string body)
    {
        var path = Path.Combine(_dir, $"tool-{Guid.NewGuid():N}");
        File.WriteAllText(path, "#!/bin/sh\n" + body + "\n");
        if (!OperatingSystem.IsWindows())
            File.SetUnixFileMode(path, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        return path;
    }

    [Fact]
    public void TheCheckoutIsFoundFromAnywhereInsideIt()
    {
        var nested = Directory.CreateDirectory(Path.Combine(Checkout(), "gui", "bin")).FullName;
        Assert.Equal(_dir, WorkerInstaller.FindSource([nested]));
        Assert.Null(WorkerInstaller.FindSource([Path.GetTempPath()]));
    }

    [Fact]
    public void WithUvThePlanIsTheReadmesThreeCommands()
    {
        var plan = WorkerInstaller.Plan(Checkout(), "/bin/uv");
        var python = Path.Combine(_dir, ".venv", "bin", "python");

        Assert.Equal(python, plan.Python);
        Assert.Equal(3, plan.Steps.Count);
        Assert.Equal(["/bin/uv", "venv", "--python", "3.12", Path.Combine(_dir, ".venv")], plan.Steps[0].Command);
        Assert.Equal(
            ["/bin/uv", "pip", "install", "--python", python, "--index-url", WorkerInstaller.TorchIndex,
                "torch==2.13.0+rocm7.1", "torchvision==0.28.0+rocm7.1"],
            plan.Steps[1].Command);
        Assert.Equal(["/bin/uv", "pip", "install", "--python", python, "-e", _dir + "[bnb]"], plan.Steps[2].Command);
    }

    [Fact]
    public void WithoutUvItFallsBackToVenvAndPip()
    {
        var plan = WorkerInstaller.Plan(Checkout(), uv: null);
        var python = Path.Combine(_dir, ".venv", "bin", "python");

        Assert.Equal(["python3.12", "-m", "venv", Path.Combine(_dir, ".venv")], plan.Steps[0].Command);
        Assert.All(plan.Steps.Skip(1), s => Assert.Equal([python, "-m", "pip", "install"], s.Command.Take(4)));
    }

    [Fact]
    public void AnExistingEnvironmentIsReusedNotRecreated()
    {
        Directory.CreateDirectory(Path.Combine(Checkout(), ".venv", "bin"));
        File.WriteAllText(Path.Combine(_dir, ".venv", "bin", "python"), "");

        var plan = WorkerInstaller.Plan(_dir, "/bin/uv");
        Assert.Equal(2, plan.Steps.Count);
        Assert.DoesNotContain(plan.Steps, s => s.Command.Contains("venv"));
    }

    [Fact]
    public async Task StepsRunInOrderAndTheFirstFailureStopsTheRest()
    {
        var log = Path.Combine(_dir, "ran");
        var tool = Script($"echo \"$1\" >> '{log}'; echo \"out $1\"; echo \"err $1\" >&2; [ \"$1\" != two ] || exit 3");
        var plan = new InstallPlan(_dir, _dir,
            [new("first", [tool, "one"]), new("second", [tool, "two"]), new("third", [tool, "three"])]);

        var titles = new List<string>();
        var text = new List<string>();
        var code = await WorkerInstaller.RunAsync(plan, s => titles.Add(s.Title), text.Add,
            TestContext.Current.CancellationToken);

        Assert.Equal(3, code);
        Assert.Equal(["first", "second"], titles);
        Assert.Equal(["one", "two"], File.ReadAllLines(log));
        var printed = string.Concat(text);
        Assert.Contains("out one", printed);
        Assert.Contains("err two", printed);
    }

    [Fact]
    public async Task AToolThatIsNotThereIsAFailedStepNotACrash()
    {
        var plan = new InstallPlan(_dir, _dir, [new("missing", [Path.Combine(_dir, "no-such-tool")])]);
        var text = new List<string>();
        Assert.Equal(127, await WorkerInstaller.RunAsync(plan, onText: text.Add,
            ct: TestContext.Current.CancellationToken));
        Assert.Contains("could not start", string.Concat(text));
    }
}
