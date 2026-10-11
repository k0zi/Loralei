using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using Loralei.Core.Models;
using Xunit;

namespace Loralei.Core.Tests;

/// <summary>
/// <c>models pull</c> against a Hub that is a loopback listener: the real HTTP client, the real
/// file layout, and nothing leaving the machine.
/// </summary>
public class ModelDownloaderTests : IDisposable
{
    private readonly string _tmp = Directory.CreateTempSubdirectory("loralei-pull").FullName;
    private readonly HttpListener _listener = new();
    private readonly string _endpoint;
    private readonly HttpClient _http = new();

    /// <summary>Repo-relative path to content; every repo on this Hub serves the same files.</summary>
    private readonly Dictionary<string, byte[]> _files = new();
    private readonly ConcurrentQueue<string> _requests = new();
    private readonly ConcurrentQueue<string?> _ranges = new();

    /// <summary>
    /// The real model with toy sizes: the space guard works from the registry's byte counts,
    /// and a test must not need 24 GiB free in the temp directory to download 30 kB.
    /// </summary>
    private static readonly ModelSpec Qwen = ModelRegistry.QwenImage is var real
        ? real with { Components = real.Components.Select(c => c with { ApproxBytes = 4096 }).ToList() }
        : real;

    public ModelDownloaderTests()
    {
        var probe = new TcpListener(IPAddress.Loopback, 0);
        probe.Start();
        var port = ((IPEndPoint)probe.LocalEndpoint).Port;
        probe.Stop();

        _endpoint = $"http://127.0.0.1:{port}";
        _listener.Prefixes.Add(_endpoint + "/");
        _listener.Start();
        _ = Task.Run(ServeAsync);

        foreach (var component in Qwen.Components)
        {
            _files[$"{component.Name}/config.json"] = "{}"u8.ToArray();
            if (component.Weights)
                _files[$"{component.Name}/diffusion_pytorch_model.safetensors"] =
                    Encoding.ASCII.GetBytes(new string('w', 5000) + component.Name);
        }
        _files["README.md"] = "not part of any component"u8.ToArray();
    }

    public void Dispose()
    {
        _listener.Stop();
        _listener.Close();
        _http.Dispose();
        Directory.Delete(_tmp, recursive: true);
    }

    private async Task ServeAsync()
    {
        while (_listener.IsListening)
        {
            HttpListenerContext context;
            try { context = await _listener.GetContextAsync(); }
            catch (Exception e) when (e is HttpListenerException or ObjectDisposedException) { return; }

            var path = Uri.UnescapeDataString(context.Request.Url!.AbsolutePath);
            _requests.Enqueue(path);
            byte[] body;
            var status = 200;
            long? from = null;

            if (path.StartsWith("/api/models/", StringComparison.Ordinal))
            {
                body = JsonSerializer.SerializeToUtf8Bytes(new
                {
                    sha = "rev123",
                    siblings = _files.Select(f => new { rfilename = f.Key, size = f.Value.Length })
                        // A hostile listing must not be able to write outside the target.
                        .Append(new { rfilename = "transformer/../../escape.txt", size = 1 }),
                });
            }
            else
            {
                // /<org>/<repo>/resolve/<revision>/<path>
                var relative = string.Join('/', path.Split('/').Skip(5));
                if (!_files.TryGetValue(relative, out body!))
                {
                    status = 404;
                    body = [];
                }
                else if (context.Request.Headers["Range"] is { } range)
                {
                    _ranges.Enqueue($"{relative} {range}");
                    from = long.Parse(range["bytes=".Length..].TrimEnd('-'));
                    body = body[(int)from.Value..];
                    status = 206;
                }
            }

            context.Response.StatusCode = status;
            try
            {
                await context.Response.OutputStream.WriteAsync(body);
                context.Response.Close();
            }
            catch (Exception e) when (e is HttpListenerException or IOException or ObjectDisposedException) { }
        }
    }

    private ModelDownloader Downloader() => new(_http, _endpoint);

    [Fact]
    public async Task MissingComponentsLandAsABrowsableTreeUnderTheFirstRoot()
    {
        var root = Path.Combine(_tmp, "models");
        var sources = new ModelSources(ModelsDir: [root]);
        var progress = new ConcurrentQueue<DownloadProgress>();

        var resolved = await Downloader().DownloadMissingAsync(Qwen, sources, progress: progress.Enqueue,
            ct: TestContext.Current.CancellationToken);

        Assert.True(resolved.Complete);
        Assert.Equal(Path.Combine(root, "qwen-image", "transformer"), resolved.PathOf("transformer"));
        Assert.Equal(_files["vae/diffusion_pytorch_model.safetensors"],
            File.ReadAllBytes(Path.Combine(root, "qwen-image", "vae", "diffusion_pytorch_model.safetensors")));

        // Only what belongs to a component is fetched, and nothing escapes the target.
        Assert.False(File.Exists(Path.Combine(root, "qwen-image", "README.md")));
        Assert.Empty(Directory.GetFiles(_tmp, "escape.txt", SearchOption.AllDirectories));
        Assert.Empty(Directory.GetFiles(root, "*.incomplete", SearchOption.AllDirectories));

        var last = progress.MaxBy(p => p.FilesDone)!;
        Assert.Equal(last.FilesTotal, last.FilesDone);
        Assert.Equal(last.BytesTotal, progress.Max(p => p.BytesDone));

        // The transformer is preferred from the original repo, the rest from a mirror: two
        // repos, each listed exactly once however many components it supplies.
        Assert.Equal(
            ["/api/models/Qwen/Qwen-Image-2512/revision/main", "/api/models/unsloth/Qwen-Image-2512/revision/main"],
            _requests.Where(r => r.StartsWith("/api/models/", StringComparison.Ordinal)).Order());
        Assert.Contains("/Qwen/Qwen-Image-2512/resolve/rev123/transformer/config.json", _requests);
        Assert.Contains("/unsloth/Qwen-Image-2512/resolve/rev123/vae/config.json", _requests);
    }

    [Fact]
    public async Task AnInterruptedDownloadResumesFromItsPartialFile()
    {
        var root = Path.Combine(_tmp, "models");
        var weights = Path.Combine(root, "qwen-image", "transformer", "diffusion_pytorch_model.safetensors");
        Directory.CreateDirectory(Path.GetDirectoryName(weights)!);
        var full = _files["transformer/diffusion_pytorch_model.safetensors"];
        File.WriteAllBytes(weights + ".incomplete", full[..2000]);

        var resolved = await Downloader().DownloadMissingAsync(Qwen, new ModelSources(ModelsDir: [root]),
            ct: TestContext.Current.CancellationToken);

        Assert.True(resolved.Complete);
        Assert.Equal(full, File.ReadAllBytes(weights));
        Assert.Contains("transformer/diffusion_pytorch_model.safetensors bytes=2000-", _ranges);
    }

    [Fact]
    public async Task OnlyWhatIsMissingIsFetched()
    {
        var root = Path.Combine(_tmp, "models");
        var sources = new ModelSources(ModelsDir: [root]);
        await Downloader().DownloadMissingAsync(Qwen, sources, ct: TestContext.Current.CancellationToken);

        Directory.Delete(Path.Combine(root, "qwen-image", "vae"), recursive: true);
        _requests.Clear();
        var resolved = await Downloader().DownloadMissingAsync(Qwen, sources, ct: TestContext.Current.CancellationToken);

        Assert.True(resolved.Complete);
        Assert.All(_requests.Where(r => r.Contains("/resolve/")), r => Assert.Contains("/vae/", r));
    }

    [Fact]
    public async Task ADryRunChecksSpaceAndTouchesNothing()
    {
        var root = Path.Combine(_tmp, "models");
        var resolved = await Downloader().DownloadMissingAsync(Qwen, new ModelSources(ModelsDir: [root]),
            dryRun: true, ct: TestContext.Current.CancellationToken);

        Assert.False(resolved.Complete);
        Assert.Empty(_requests);
        Assert.False(Directory.Exists(root));
    }
}
