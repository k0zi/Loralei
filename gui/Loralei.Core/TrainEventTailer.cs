using System.Text;
using System.Text.Json;
using Loralei.Core.Dto;

namespace Loralei.Core;

/// <summary>
/// Follows <c>runs/&lt;name&gt;/logs/train.jsonl</c> as the trainer appends to it.
/// <para>
/// The file is the only per-step source there is: <c>manifest.json</c> advances only at
/// <c>save_every</c> boundaries, and the progress bar exists only on a terminal.
/// </para>
/// </summary>
public sealed class TrainEventTailer(string path) : IDisposable
{
    private static readonly JsonSerializerOptions Json = new() { PropertyNameCaseInsensitive = true };

    private FileStream? _stream;
    private long _offset;
    private readonly StringBuilder _partial = new();

    public string Path { get; } = path;

    /// <summary>Events appended since the last call; empty until the trainer creates the file.</summary>
    public IReadOnlyList<TrainEvent> Poll()
    {
        try
        {
            if (_stream is null)
            {
                if (!File.Exists(Path)) return [];
                // Shared read/write: the trainer holds this file open for the whole run.
                _stream = new FileStream(Path, FileMode.Open, FileAccess.Read,
                    FileShare.ReadWrite | FileShare.Delete);
            }

            // A rerun of the same name truncates the file; start over rather than seek past the end.
            if (_stream.Length < _offset) { _offset = 0; _partial.Clear(); }
            if (_stream.Length == _offset) return [];

            _stream.Seek(_offset, SeekOrigin.Begin);
            var buffer = new byte[_stream.Length - _offset];
            var read = _stream.Read(buffer, 0, buffer.Length);
            _offset += read;
            _partial.Append(Encoding.UTF8.GetString(buffer, 0, read));
        }
        catch (IOException) { return []; }
        catch (UnauthorizedAccessException) { return []; }

        var text = _partial.ToString();
        var lastBreak = text.LastIndexOf('\n');
        if (lastBreak < 0) return [];

        // Keep the trailing fragment: a line may still be mid-write.
        _partial.Clear();
        _partial.Append(text[(lastBreak + 1)..]);

        var events = new List<TrainEvent>();
        foreach (var line in text[..lastBreak].Split('\n'))
        {
            var trimmed = line.Trim();
            if (trimmed.Length == 0) continue;
            try
            {
                if (JsonSerializer.Deserialize<TrainEvent>(trimmed, Json) is { } ev) events.Add(ev);
            }
            catch (JsonException)
            {
                // A torn line is not worth failing a live view over.
            }
        }
        return events;
    }

    public void Reset()
    {
        _stream?.Dispose();
        _stream = null;
        _offset = 0;
        _partial.Clear();
    }

    public void Dispose() => Reset();
}
