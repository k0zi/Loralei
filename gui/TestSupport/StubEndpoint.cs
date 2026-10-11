using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;

namespace Loralei.Tests;

/// <summary>
/// A chat-completions endpoint on a loopback port, standing in for the vision model. The
/// captioning tests go through the real HTTP client against it, so the request they check is
/// the one a real endpoint would get - and nothing leaves the machine.
/// </summary>
internal sealed class StubEndpoint : IDisposable
{
    private readonly HttpListener _listener = new();

    public string BaseUrl { get; }

    /// <summary>The JSON body of every request received, in arrival order.</summary>
    public ConcurrentQueue<string> Bodies { get; } = new();

    /// <summary>The Authorization header of every request, null where there was none.</summary>
    public ConcurrentQueue<string?> Authorizations { get; } = new();

    /// <summary>Given the 1-based request number, what to answer. Defaults to a fixed caption.</summary>
    public Func<int, (int Status, string Body)> Respond { get; set; } =
        _ => (200, Completion("A red cube on a table."));

    public StubEndpoint()
    {
        var probe = new TcpListener(IPAddress.Loopback, 0);
        probe.Start();
        var port = ((IPEndPoint)probe.LocalEndpoint).Port;
        probe.Stop();

        BaseUrl = $"http://127.0.0.1:{port}/v1";
        _listener.Prefixes.Add($"http://127.0.0.1:{port}/");
        _listener.Start();
        _ = Task.Run(ServeAsync);
    }

    public static string Completion(string content) => JsonSerializer.Serialize(new
    {
        choices = new[] { new { message = new { role = "assistant", content } } },
    });

    private async Task ServeAsync()
    {
        var count = 0;
        while (_listener.IsListening)
        {
            HttpListenerContext context;
            try { context = await _listener.GetContextAsync(); }
            catch (Exception e) when (e is HttpListenerException or ObjectDisposedException) { return; }

            using (var reader = new StreamReader(context.Request.InputStream, Encoding.UTF8))
                Bodies.Enqueue(await reader.ReadToEndAsync());
            Authorizations.Enqueue(context.Request.Headers["Authorization"]);

            var (status, body) = Respond(Interlocked.Increment(ref count));
            var bytes = Encoding.UTF8.GetBytes(body);
            context.Response.StatusCode = status;
            context.Response.ContentType = "application/json";
            try
            {
                await context.Response.OutputStream.WriteAsync(bytes);
                context.Response.Close();
            }
            catch (Exception e) when (e is HttpListenerException or IOException or ObjectDisposedException) { }
        }
    }

    public void Dispose()
    {
        try { _listener.Stop(); _listener.Close(); }
        catch (ObjectDisposedException) { }
    }
}
