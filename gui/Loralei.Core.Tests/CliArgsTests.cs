using Loralei.Cli;
using Xunit;

namespace Loralei.Core.Tests;

public class CliArgsTests
{
    private static readonly OptionSet Options = new OptionSet()
        .Value("--name", "-n").Value("--prompt", "-p").Value("--lr").Value("--seed")
        .Flag("--fp8-base", "--no-fp8-base").Flag("--dry-run");

    [Fact]
    public void ValuesFlagsAndPositionalsAreSeparated()
    {
        var args = Options.Parse(["./data", "-n", "style", "--lr=5e-5", "--dry-run", "-p", "a cat", "-p", "a dog"]);
        Assert.Equal(["./data"], args.Positional);
        Assert.Equal("style", args.Get("--name"));
        Assert.Equal(5e-5, args.Double("--lr"));
        Assert.True(args.On("--dry-run"));
        Assert.Equal(["a cat", "a dog"], args.All("--prompt"));
    }

    [Fact]
    public void ASwitchIsTriStateSoAConfigLayerCanStillDecide()
    {
        Assert.Null(Options.Parse([]).Flag("--fp8-base"));
        Assert.True(Options.Parse(["--fp8-base"]).Flag("--fp8-base"));
        Assert.False(Options.Parse(["--no-fp8-base"]).Flag("--fp8-base"));
    }

    [Fact]
    public void NegativeNumbersAreValuesNotOptions()
    {
        Assert.Equal(-1, Options.Parse(["--seed", "-1"]).Int("--seed"));
    }

    [Theory]
    [InlineData("--rank", "unknown option --rank")]
    [InlineData("--name", "--name needs a value")]
    [InlineData("--dry-run=yes", "--dry-run does not take a value")]
    public void MistakesAreNamed(string token, string message)
    {
        Assert.Equal(message, Assert.Throws<UsageException>(() => Options.Parse([token])).Message);
    }

    [Fact]
    public void BadNumbersAndMissingArgumentsAreUsageErrors()
    {
        Assert.Throws<UsageException>(() => Options.Parse(["--seed", "soon"]).Int("--seed"));
        Assert.Throws<UsageException>(() => Options.Parse([]).Required(0, "the dataset directory"));
        Assert.Throws<UsageException>(() => Options.Parse(["a", "b"]).NoMoreThan(1));
        Assert.True(Options.Parse(["--help"]).Help);
    }
}
