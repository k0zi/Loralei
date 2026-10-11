using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Loralei.Core.Lora;

/// <summary>One tensor as stored: its element type, shape and raw little-endian bytes, row-major.</summary>
public sealed record Tensor(string Dtype, long[] Shape, byte[] Data)
{
    /// <summary>Bytes per element, from the safetensors dtype name.</summary>
    public int ElementSize => Dtype switch
    {
        "BOOL" or "U8" or "I8" or "F8_E4M3" or "F8_E5M2" => 1,
        "U16" or "I16" or "F16" or "BF16" => 2,
        "U32" or "I32" or "F32" => 4,
        "U64" or "I64" or "F64" => 8,
        _ => throw new InvalidDataException($"unsupported safetensors dtype {Dtype}"),
    };
}

/// <summary>
/// The safetensors container: an 8-byte little-endian header length, a JSON header mapping
/// each tensor name to its dtype, shape and byte range, then the tensor bytes back to back.
/// Reading and re-writing it needs no tensor library at all, which is what lets the ComfyUI
/// export run without starting Python.
/// </summary>
public sealed class SafeTensorsFile
{
    private const string MetadataKey = "__metadata__";

    /// <summary>A header this large is a corrupt or hostile file, not a LoRA.</summary>
    private const long MaxHeaderBytes = 100 * 1024 * 1024;

    public Dictionary<string, string> Metadata { get; init; } = new(StringComparer.Ordinal);
    public Dictionary<string, Tensor> Tensors { get; init; } = new(StringComparer.Ordinal);

    /// <summary>Tensor names only, without reading any tensor data.</summary>
    public static IReadOnlyList<string> Keys(string path)
    {
        using var stream = File.OpenRead(path);
        return ReadHeader(stream).Header.Where(p => p.Key != MetadataKey).Select(p => p.Key).ToList();
    }

    public static SafeTensorsFile Load(string path)
    {
        using var stream = File.OpenRead(path);
        var (header, dataStart) = ReadHeader(stream);
        var file = new SafeTensorsFile();

        foreach (var (name, node) in header)
        {
            if (name == MetadataKey)
            {
                if (node is JsonObject metadata)
                    foreach (var (key, value) in metadata)
                        file.Metadata[key] = value?.GetValue<string>() ?? "";
                continue;
            }

            var entry = node!.AsObject();
            var offsets = entry["data_offsets"]!.AsArray();
            var begin = offsets[0]!.GetValue<long>();
            var end = offsets[1]!.GetValue<long>();
            if (begin < 0 || end < begin || dataStart + end > stream.Length)
                throw new InvalidDataException($"{name}: data range lies outside the file");

            var data = new byte[end - begin];
            stream.Position = dataStart + begin;
            stream.ReadExactly(data);
            var tensor = new Tensor(
                entry["dtype"]!.GetValue<string>(),
                entry["shape"]!.AsArray().Select(n => n!.GetValue<long>()).ToArray(),
                data);
            if (tensor.Shape.Aggregate(1L, (a, b) => a * b) * tensor.ElementSize != data.Length)
                throw new InvalidDataException($"{name}: shape does not match its byte length");
            file.Tensors[name] = tensor;
        }
        return file;
    }

    private static (JsonObject Header, long DataStart) ReadHeader(Stream stream)
    {
        Span<byte> prefix = stackalloc byte[8];
        if (stream.ReadAtLeast(prefix, 8, throwOnEndOfStream: false) < 8)
            throw new InvalidDataException("not a safetensors file: too short");
        var length = (long)BinaryPrimitives.ReadUInt64LittleEndian(prefix);
        if (length <= 0 || length > MaxHeaderBytes || 8 + length > stream.Length)
            throw new InvalidDataException("not a safetensors file: implausible header length");

        var json = new byte[length];
        stream.ReadExactly(json);
        try
        {
            return (JsonNode.Parse(json)!.AsObject(), 8 + length);
        }
        catch (Exception e) when (e is JsonException or InvalidOperationException)
        {
            throw new InvalidDataException($"not a safetensors file: {e.Message}");
        }
    }

    public void Save(string path)
    {
        var header = new JsonObject();
        if (Metadata.Count > 0)
        {
            var metadata = new JsonObject();
            foreach (var (key, value) in Metadata.OrderBy(p => p.Key, StringComparer.Ordinal)) metadata[key] = value;
            header[MetadataKey] = metadata;
        }

        var ordered = Tensors.OrderBy(p => p.Key, StringComparer.Ordinal).ToList();
        long offset = 0;
        foreach (var (name, tensor) in ordered)
        {
            header[name] = new JsonObject
            {
                ["dtype"] = tensor.Dtype,
                ["shape"] = new JsonArray(tensor.Shape.Select(d => (JsonNode?)JsonValue.Create(d)).ToArray()),
                ["data_offsets"] = new JsonArray(offset, offset + tensor.Data.Length),
            };
            offset += tensor.Data.Length;
        }

        var json = Encoding.UTF8.GetBytes(header.ToJsonString());
        // The data must start 8-byte aligned; the format pads the header with spaces to get there.
        var padded = (json.Length + 7) / 8 * 8;

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        // Written beside the target and renamed, so a crash never leaves half an adapter in place.
        var temp = path + ".tmp";
        using (var stream = File.Create(temp))
        {
            Span<byte> prefix = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)padded);
            stream.Write(prefix);
            stream.Write(json);
            for (var i = json.Length; i < padded; i++) stream.WriteByte((byte)' ');
            foreach (var (_, tensor) in ordered) stream.Write(tensor.Data);
        }
        File.Move(temp, path, overwrite: true);
    }
}
