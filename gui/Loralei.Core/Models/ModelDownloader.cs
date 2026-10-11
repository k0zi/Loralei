using System.Net;
using System.Net.Http.Headers;
using System.Text.Json;
using Loralei.Core.Config;

namespace Loralei.Core.Models;

/// <param name="Path">Repo-relative, e.g. <c>transformer/config.json</c>.</param>
public sealed record RemoteFile(string Path, long Size);

public sealed record DownloadProgress(long BytesDone, long BytesTotal, int FilesDone, int FilesTotal, string Current);

/// <summary>
/// Fetches missing model components from the Hugging Face Hub. This is the one place in
/// loralei that opens a network connection, and it does so only for <c>models pull</c>.
/// <para>
/// Files land where <see cref="ModelResolver"/> will look for them: as a browsable
/// <c>&lt;models_dir&gt;/&lt;model&gt;/&lt;component&gt;/</c> tree when a root is configured, otherwise as a
/// snapshot in the default Hugging Face cache. A download that is interrupted leaves an
/// <c>.incomplete</c> file behind and resumes from it.
/// </para>
/// </summary>
/// <param name="endpoint">The Hub to talk to; the default honours <c>HF_ENDPOINT</c>, which points at a mirror.</param>
public sealed class ModelDownloader(HttpClient http, string? endpoint = null)
{
    private const int Parallelism = 4;
    private const string IncompleteSuffix = ".incomplete";

    public string Endpoint { get; } = (endpoint
        ?? (Environment.GetEnvironmentVariable("HF_ENDPOINT") is { Length: > 0 } custom ? custom : "https://huggingface.co"))
        .TrimEnd('/');

    public static HttpClient CreateClient()
    {
        var client = new HttpClient(new SocketsHttpHandler { AutomaticDecompression = DecompressionMethods.None })
        {
            Timeout = Timeout.InfiniteTimeSpan, // a 12 GiB shard is not done in 100 seconds
        };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("loralei/0.2");
        if (Token() is { } token)
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        return client;
    }

    /// <summary>A Hub token for gated repos: <c>HF_TOKEN</c>, else the file <c>hf auth login</c> writes.</summary>
    private static string? Token()
    {
        if (Environment.GetEnvironmentVariable("HF_TOKEN") is { Length: > 0 } token) return token.Trim();
        var home = Environment.GetEnvironmentVariable("HF_HOME") is { Length: > 0 } custom
            ? custom
            : Path.Combine(PathText.Home, ".cache", "huggingface");
        try
        {
            var file = Path.Combine(home, "token");
            return File.Exists(file) && File.ReadAllText(file).Trim() is { Length: > 0 } saved ? saved : null;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return null; }
    }

    /// <summary>The revision <c>main</c> points at, and every file in the repo with its size.</summary>
    public async Task<(string Revision, List<RemoteFile> Files)> ListAsync(string repo, CancellationToken ct = default)
    {
        using var response = await http.GetAsync($"{Endpoint}/api/models/{repo}/revision/main?blobs=true", ct)
            .ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
            throw new HttpRequestException(
                $"{repo}: the Hub answered {(int)response.StatusCode} {response.ReasonPhrase}" +
                (response.StatusCode is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden
                    ? " - a gated repo needs HF_TOKEN" : ""));

        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false));
        var root = doc.RootElement;
        var revision = root.GetProperty("sha").GetString() ?? "main";
        var files = new List<RemoteFile>();
        foreach (var sibling in root.GetProperty("siblings").EnumerateArray())
        {
            var path = sibling.GetProperty("rfilename").GetString();
            if (string.IsNullOrEmpty(path) || !IsSafe(path)) continue;
            var size = sibling.TryGetProperty("size", out var s) && s.ValueKind == JsonValueKind.Number ? s.GetInt64() : -1;
            files.Add(new RemoteFile(path, size));
        }
        return (revision, files);
    }

    /// <summary>A listing is remote input: a name that climbs out of the target directory is refused.</summary>
    private static bool IsSafe(string path) =>
        !Path.IsPathRooted(path) && !path.Split('/').Any(part => part is "" or "." or "..");

    /// <summary>Downloads every component the model is missing and resolves it again.</summary>
    public async Task<ResolvedModel> DownloadMissingAsync(ModelSpec model, ModelSources? sources,
        bool dryRun = false, Action<DownloadProgress>? progress = null, CancellationToken ct = default)
    {
        var resolved = ModelResolver.Resolve(model, sources);
        if (resolved.Complete) return resolved;

        var target = ModelResolver.DownloadTarget(model, sources);
        Disk.EnsureSpace(target ?? Disk.HfHubDir(), resolved.MissingBytes,
            $"{model.Key} download ({resolved.Missing.Count} component(s))");
        if (dryRun) return resolved;

        // Several components usually come from one repo; list each repo once.
        var plan = new List<(string Url, string Destination, RemoteFile File)>();
        foreach (var group in resolved.Missing.GroupBy(m => model.Component(m.Component).Repos[0]))
        {
            var repo = group.Key;
            var (revision, files) = await ListAsync(repo, ct).ConfigureAwait(false);
            // Land the files as <models_dir>/<model>/<component>/... rather than in the cache's
            // blob layout, so the directory stays browsable and portable.
            var destination = target ?? Path.Combine(
                Disk.HfHubDir(), "models--" + repo.Replace("/", "--"), "snapshots", revision);

            foreach (var component in group)
            {
                var prefix = component.Component + "/";
                var wanted = files.Where(f => f.Path.StartsWith(prefix, StringComparison.Ordinal)).ToList();
                if (wanted.Count == 0)
                    throw new InvalidOperationException($"{repo} has no {component.Component}/ folder");
                foreach (var file in wanted)
                {
                    var encoded = string.Join('/', file.Path.Split('/').Select(Uri.EscapeDataString));
                    plan.Add(($"{Endpoint}/{repo}/resolve/{revision}/{encoded}",
                        Path.Combine(destination, Path.Combine(file.Path.Split('/'))), file));
                }
            }
        }

        var total = plan.Sum(p => Math.Max(0, p.File.Size));
        long done = 0;
        var filesDone = 0;
        await Parallel.ForEachAsync(plan,
            new ParallelOptions { MaxDegreeOfParallelism = Parallelism, CancellationToken = ct },
            async (item, token) =>
            {
                await FetchAsync(item.Url, item.Destination, item.File.Size, bytes =>
                {
                    var now = Interlocked.Add(ref done, bytes);
                    progress?.Invoke(new DownloadProgress(now, total, Volatile.Read(ref filesDone), plan.Count, item.File.Path));
                }, token).ConfigureAwait(false);
                var finished = Interlocked.Increment(ref filesDone);
                progress?.Invoke(new DownloadProgress(Interlocked.Read(ref done), total, finished, plan.Count, item.File.Path));
            }).ConfigureAwait(false);

        return ModelResolver.Resolve(model, sources);
    }

    /// <param name="advance">Called with the number of bytes that just became durable.</param>
    private async Task FetchAsync(string url, string destination, long size, Action<long> advance,
        CancellationToken ct)
    {
        if (File.Exists(destination) && (size < 0 || new FileInfo(destination).Length == size))
        {
            advance(Math.Max(0, size));
            return;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        var partial = destination + IncompleteSuffix;
        var have = File.Exists(partial) ? new FileInfo(partial).Length : 0;
        if (size >= 0 && have > size) have = 0;

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (have > 0) request.Headers.Range = new RangeHeaderValue(have, null);

        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct)
            .ConfigureAwait(false);
        // 416 means the partial file already holds everything; anything else unexpected is an error.
        if (response.StatusCode != HttpStatusCode.RequestedRangeNotSatisfiable)
        {
            if (!response.IsSuccessStatusCode)
                throw new HttpRequestException($"{url}: {(int)response.StatusCode} {response.ReasonPhrase}");

            // A server that ignores Range sends the whole file again with a plain 200.
            var resuming = have > 0 && response.StatusCode == HttpStatusCode.PartialContent;
            if (!resuming) have = 0;
            if (have > 0) advance(have);

            await using var source = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
            await using var sink = new FileStream(partial, resuming ? FileMode.Append : FileMode.Create,
                FileAccess.Write, FileShare.None, 1 << 20);
            var buffer = new byte[1 << 20];
            int read;
            while ((read = await source.ReadAsync(buffer, ct).ConfigureAwait(false)) > 0)
            {
                await sink.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
                advance(read);
            }
        }
        else advance(have);

        var written = new FileInfo(partial).Length;
        if (size >= 0 && written != size)
            throw new IOException($"{Path.GetFileName(destination)}: got {written} of {size} bytes");
        File.Move(partial, destination, overwrite: true);
    }
}
