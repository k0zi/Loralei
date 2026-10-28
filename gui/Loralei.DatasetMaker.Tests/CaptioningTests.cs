using System.Text.Json;
using Loralei.DatasetMaker.Captioning;
using Loralei.DatasetMaker.Config;
using Loralei.DatasetMaker.Processing;
using Loralei.Tests;

namespace Loralei.DatasetMaker.Tests;

/// <summary>The whole pipeline - scan, request, post-process, write - against a loopback endpoint.</summary>
public sealed class CaptioningTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("datasetmaker-run").FullName;
    private readonly StubEndpoint _endpoint = new();
    private readonly List<ProcessingEvent> _events = [];

    public void Dispose()
    {
        _endpoint.Dispose();
        Directory.Delete(_folder, recursive: true);
    }

    private string Image(string name, int seed = 0)
    {
        var path = Path.Combine(_folder, name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        Png.Write(path, 8, 8, seed);
        return path;
    }

    private async Task<ProcessingSummary> Run(
        bool overwrite = false, bool dryRun = false, string? trigger = null, int retries = 0, string apiKey = "")
    {
        var options = new OpenAiOptions
        {
            BaseUrl = _endpoint.BaseUrl, ApiKey = apiKey, Model = "test-vlm", MaxTokens = 123, ImageDetail = "low",
        };
        using var client = new OpenAiCaptionClient(options);
        var settings = new ProcessorSettings("Describe it.", trigger, overwrite, MaxParallel: 2, retries, dryRun);
        var processor = new DatasetProcessor(client, settings, e =>
        {
            // Started only says a request is on its way; what these tests read is how each image ended.
            if (e.Status != ItemStatus.Started) lock (_events) _events.Add(e);
        });
        return await processor.RunAsync(_folder, TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task Every_image_gets_a_caption_file_beside_it()
    {
        var a = Image("a.png", 1);
        var b = Image(Path.Combine("nested", "b.png"), 2);
        File.WriteAllText(Path.Combine(_folder, "notes.md"), "not an image");

        var summary = await Run(trigger: "ohwx");

        Assert.Equal(new ProcessingSummary(Total: 2, Succeeded: 2, Skipped: 0, Failed: 0), summary);
        Assert.Equal("ohwx, A red cube on a table.", File.ReadAllText(Path.ChangeExtension(a, ".txt")));
        Assert.Equal("ohwx, A red cube on a table.", File.ReadAllText(Path.ChangeExtension(b, ".txt")));
        Assert.Empty(Directory.GetFiles(_folder, "*.tmp", SearchOption.AllDirectories));
        Assert.All(_events, e => Assert.Equal(ItemStatus.Ok, e.Status));
        Assert.Contains(_events, e => e.Caption == "ohwx, A red cube on a table.");
    }

    [Fact]
    public async Task The_request_carries_the_model_the_prompts_and_the_image()
    {
        Image("a.png");

        await Run(apiKey: "sk-test");

        using var body = JsonDocument.Parse(Assert.Single(_endpoint.Bodies));
        var root = body.RootElement;
        Assert.Equal("test-vlm", root.GetProperty("model").GetString());
        Assert.Equal(123, root.GetProperty("max_tokens").GetInt32());

        var messages = root.GetProperty("messages");
        Assert.Equal("Describe it.", messages[0].GetProperty("content").GetString());
        var image = messages[1].GetProperty("content")[1].GetProperty("image_url");
        Assert.StartsWith("data:image/png;base64,", image.GetProperty("url").GetString());
        Assert.Equal("low", image.GetProperty("detail").GetString());

        Assert.Equal("Bearer sk-test", Assert.Single(_endpoint.Authorizations));
    }

    [Fact]
    public async Task Without_a_key_no_authorization_is_sent()
    {
        Image("a.png");

        await Run();

        Assert.Null(Assert.Single(_endpoint.Authorizations));
    }

    [Fact]
    public async Task An_existing_caption_is_kept_unless_overwriting()
    {
        var image = Image("a.png");
        var caption = Path.ChangeExtension(image, ".txt");
        File.WriteAllText(caption, "written by hand");

        var kept = await Run();
        Assert.Equal(new ProcessingSummary(1, 0, 1, 0), kept);
        Assert.Equal("written by hand", File.ReadAllText(caption));
        Assert.Empty(_endpoint.Bodies);

        var replaced = await Run(overwrite: true);
        Assert.Equal(new ProcessingSummary(1, 1, 0, 0), replaced);
        Assert.Equal("A red cube on a table.", File.ReadAllText(caption));
    }

    [Fact]
    public async Task Two_images_that_would_share_a_caption_file_do_not_fight_over_it()
    {
        Image("same.jpg", 1);
        Image("same.png", 2);

        var summary = await Run();

        Assert.Equal(new ProcessingSummary(2, 1, 1, 0), summary);
        Assert.Single(_endpoint.Bodies);
    }

    [Fact]
    public async Task A_dry_run_lists_the_work_and_sends_nothing()
    {
        var todo = Image("todo.png", 1);
        var done = Image("done.png", 2);
        File.WriteAllText(Path.ChangeExtension(done, ".txt"), "already there");

        var summary = await Run(dryRun: true);

        Assert.Equal(new ProcessingSummary(2, 0, 1, 0), summary);
        Assert.Empty(_endpoint.Bodies);
        Assert.False(File.Exists(Path.ChangeExtension(todo, ".txt")));
        Assert.Equal(todo, Assert.Single(_events, e => e.Status == ItemStatus.Todo).Image);
    }

    [Fact]
    public async Task A_server_error_is_retried()
    {
        var image = Image("a.png");
        _endpoint.Respond = n => n == 1 ? (503, "busy") : (200, StubEndpoint.Completion("Second time lucky."));

        var summary = await Run(retries: 1);

        Assert.Equal(new ProcessingSummary(1, 1, 0, 0), summary);
        Assert.Equal("Second time lucky.", File.ReadAllText(Path.ChangeExtension(image, ".txt")));
        Assert.Contains(_events, e => e.Status == ItemStatus.Retrying);
    }

    [Fact]
    public async Task A_refused_request_fails_that_image_without_retrying_and_says_why()
    {
        var image = Image("a.png");
        _endpoint.Respond = _ => (400, "model does not support images");

        var summary = await Run(retries: 3);

        Assert.Equal(new ProcessingSummary(1, 0, 0, 1), summary);
        Assert.Single(_endpoint.Bodies);
        Assert.False(File.Exists(Path.ChangeExtension(image, ".txt")));
        var failure = Assert.Single(_events);
        Assert.Equal(ItemStatus.Failed, failure.Status);
        Assert.Contains("model does not support images", failure.Detail);
        Assert.StartsWith("[1/1] FAIL ", failure.ToString());
    }
}
