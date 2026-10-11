using Loralei.ImageResizer.Models;
using Loralei.ImageResizer.Services;

namespace Loralei.ImageResizer.Tests;

public sealed class AppSettingsStoreTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("imageresizer-settings").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private AppSettingsStore Store(string name = "settings.json") =>
        new(Path.Combine(_folder, name));

    [Fact]
    public void A_missing_file_gives_the_defaults()
    {
        var settings = Store().Load();

        Assert.Equal(1024, settings.TargetSize);
        Assert.Equal(ResizeMode.Crop, settings.DefaultMode);
        Assert.Equal(PadFill.Blur, settings.PadFill);
        Assert.Equal(95, settings.JpegQuality);
        Assert.Null(settings.LastFolder);
    }

    [Fact]
    public void Everything_survives_a_round_trip()
    {
        var store = Store();
        var original = new AppSettings
        {
            TargetSize = 2048,
            DefaultMode = ResizeMode.Pad,
            PadFill = PadFill.Transparent,
            JpegQuality = 72,
            LastFolder = "/tmp/pictures",
        };

        store.Save(original);

        Assert.Equal(original, store.Load());
    }

    [Fact]
    public void The_file_is_readable_json_with_named_enums()
    {
        var store = Store();
        store.Save(new AppSettings { DefaultMode = ResizeMode.Pad, PadFill = PadFill.White });

        var text = File.ReadAllText(store.Path);

        Assert.Contains("\"DefaultMode\": \"Pad\"", text);
        Assert.Contains("\"PadFill\": \"White\"", text);
    }

    [Theory]
    [InlineData("this is not json at all")]
    [InlineData("{ \"Theme\": ")]
    [InlineData("")]
    public void A_corrupt_file_falls_back_to_the_defaults_instead_of_throwing(string content)
    {
        var store = Store();
        File.WriteAllText(store.Path, content);

        Assert.Equal(new AppSettings(), store.Load());
    }

    [Fact]
    public void Out_of_range_values_are_brought_back_into_range()
    {
        var store = Store();
        File.WriteAllText(store.Path, """
            {
              "DefaultMode": "Pad",
              "TargetSize": 999,
              "JpegQuality": 5000
            }
            """);

        var settings = store.Load();

        Assert.Equal(ResizeMode.Pad, settings.DefaultMode); // the valid parts are kept
        Assert.Equal(1024, settings.TargetSize);            // 999 is not an offered size
        Assert.Equal(100, settings.JpegQuality);
    }

    [Fact]
    public void Theme_keys_left_by_an_older_version_are_ignored()
    {
        // The theme used to live in this file; it is KD.Avalonia.Rice's business now.
        var store = Store();
        File.WriteAllText(store.Path, """
            {
              "Theme": "Dark",
              "Accent": "Green",
              "TargetSize": 2048
            }
            """);

        Assert.Equal(2048, store.Load().TargetSize);
    }

    [Fact]
    public void Saving_creates_the_folder_it_needs()
    {
        var store = new AppSettingsStore(Path.Combine(_folder, "nested", "deeper", "settings.json"));

        store.Save(new AppSettings { TargetSize = 512 });

        Assert.True(File.Exists(store.Path));
        Assert.Equal(512, store.Load().TargetSize);
    }

    [Fact]
    public void The_default_path_sits_under_the_users_config_directory()
    {
        var path = AppSettingsStore.DefaultPath();

        Assert.EndsWith(Path.Combine("ImageResizer", "settings.json"), path);
        Assert.StartsWith(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), path);
    }
}
