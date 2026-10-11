using Loralei.DatasetMaker.Config;

namespace Loralei.DatasetMaker.Tests;

public sealed class ConfigStoreTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("datasetmaker-config").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private ConfigStore Store() => new(Path.Combine(_folder, "nested", "dataset-maker.json"));

    [Fact]
    public void Everything_survives_a_round_trip()
    {
        var store = Store();
        var original = new AppConfig
        {
            OpenAI = new OpenAiOptions
            {
                BaseUrl = "http://localhost:1234/v1", ApiKey = "sk-test", Model = "some-vlm",
                MaxTokens = 250, Temperature = 0.7, TimeoutSeconds = 45, ImageDetail = "low",
            },
            Caption = new CaptionOptions
            {
                SystemPrompt = "Describe it.", ExtraSystemPrompt = "It is a cat.", TriggerWord = "ohwx",
                MaxParallel = 4, MaxRetries = 1,
            },
        };

        store.Save(original);

        Assert.Equal(original, store.Load());
    }

    [Fact]
    public void With_no_file_of_its_own_it_starts_from_the_shipped_defaults()
    {
        // The defaults file travels with the library into this test's output folder.
        Assert.True(File.Exists(Path.Combine(AppContext.BaseDirectory, ConfigStore.DefaultsFileName)));

        var config = Store().Load();

        Assert.StartsWith("http", config.OpenAI.BaseUrl);
        Assert.False(string.IsNullOrWhiteSpace(config.Caption.SystemPrompt));
    }

    [Theory]
    [InlineData("this is not json")]
    [InlineData("{ \"OpenAI\": ")]
    public void A_corrupt_file_is_ignored_instead_of_throwing(string content)
    {
        var store = Store();
        Directory.CreateDirectory(Path.GetDirectoryName(store.Path)!);
        File.WriteAllText(store.Path, content);

        Assert.False(string.IsNullOrWhiteSpace(store.Load().OpenAI.Model));
    }

    [Fact]
    public void A_blank_system_prompt_becomes_the_built_in_one()
    {
        var store = Store();
        store.Save(new AppConfig { Caption = new CaptionOptions { SystemPrompt = "  " } });

        Assert.Equal(CaptionOptions.DefaultSystemPrompt, store.Load().Caption.SystemPrompt);
    }

    [Fact]
    public void The_file_is_private_to_its_owner_because_it_can_hold_a_key()
    {
        if (OperatingSystem.IsWindows()) return;
        var store = Store();

        store.Save(new AppConfig());

        Assert.Equal(UnixFileMode.UserRead | UnixFileMode.UserWrite, File.GetUnixFileMode(store.Path));
    }

    [Fact]
    public void A_configured_key_wins_over_the_environment() =>
        Assert.Equal("sk-mine", ConfigStore.ResolveApiKey("sk-mine"));
}
