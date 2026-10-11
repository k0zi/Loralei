using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Loralei.DatasetMaker.Config;

namespace Loralei.DatasetMaker.Captioning;

public sealed class CaptionRequestException(string message, bool isTransient) : Exception(message)
{
    public bool IsTransient { get; } = isTransient;
}

/// <summary>Minimal client for any OpenAI-compatible /chat/completions endpoint with vision support.</summary>
public sealed class OpenAiCaptionClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly OpenAiOptions _options;

    public OpenAiCaptionClient(OpenAiOptions options)
    {
        _options = options;
        _http = new HttpClient
        {
            BaseAddress = new Uri(options.BaseUrl.TrimEnd('/') + "/"),
            Timeout = TimeSpan.FromSeconds(options.TimeoutSeconds),
        };
        if (!string.IsNullOrWhiteSpace(options.ApiKey))
            _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", options.ApiKey);
    }

    public async Task<string> CaptionAsync(string imagePath, string systemPrompt, CancellationToken ct)
    {
        var bytes = await File.ReadAllBytesAsync(imagePath, ct);
        var dataUrl = $"data:{GetMimeType(imagePath)};base64,{Convert.ToBase64String(bytes)}";

        var body = new JsonObject
        {
            ["model"] = _options.Model,
            ["max_tokens"] = _options.MaxTokens,
            ["temperature"] = _options.Temperature,
            ["messages"] = new JsonArray
            {
                new JsonObject { ["role"] = "system", ["content"] = systemPrompt },
                new JsonObject
                {
                    ["role"] = "user",
                    ["content"] = new JsonArray
                    {
                        new JsonObject { ["type"] = "text", ["text"] = PromptBuilder.UserPrompt },
                        new JsonObject
                        {
                            ["type"] = "image_url",
                            ["image_url"] = new JsonObject { ["url"] = dataUrl, ["detail"] = _options.ImageDetail },
                        },
                    },
                },
            },
        };

        HttpResponseMessage response;
        try
        {
            using var requestContent = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json");
            response = await _http.PostAsync("chat/completions", requestContent, ct);
        }
        catch (HttpRequestException ex)
        {
            throw new CaptionRequestException($"Request failed: {ex.Message}", isTransient: true);
        }
        catch (TaskCanceledException) when (!ct.IsCancellationRequested)
        {
            throw new CaptionRequestException($"Request timed out after {_options.TimeoutSeconds}s", isTransient: true);
        }

        using (response)
        {
            var text = await response.Content.ReadAsStringAsync(ct);
            if (!response.IsSuccessStatusCode)
            {
                var transient = response.StatusCode is HttpStatusCode.TooManyRequests or HttpStatusCode.RequestTimeout
                                || (int)response.StatusCode >= 500;
                throw new CaptionRequestException(
                    $"HTTP {(int)response.StatusCode} {response.StatusCode}: {Truncate(text, 300)}", transient);
            }

            var content = ExtractContent(text);
            if (string.IsNullOrWhiteSpace(content))
                throw new CaptionRequestException($"Empty response from model: {Truncate(text, 300)}", isTransient: true);
            return content;
        }
    }

    private static string? ExtractContent(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var message = doc.RootElement.GetProperty("choices")[0].GetProperty("message");
            return message.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.String
                ? content.GetString()
                : null;
        }
        catch (Exception ex) when (ex is JsonException or KeyNotFoundException or IndexOutOfRangeException or InvalidOperationException)
        {
            throw new CaptionRequestException($"Unexpected response format: {Truncate(json, 300)}", isTransient: false);
        }
    }

    private static string GetMimeType(string path) => Path.GetExtension(path).ToLowerInvariant() switch
    {
        ".jpg" or ".jpeg" => "image/jpeg",
        ".png" => "image/png",
        ".webp" => "image/webp",
        ".gif" => "image/gif",
        ".bmp" => "image/bmp",
        _ => "application/octet-stream",
    };

    private static string Truncate(string s, int max) => s.Length <= max ? s : s[..max] + "...";

    public void Dispose() => _http.Dispose();
}
