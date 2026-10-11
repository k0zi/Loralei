using System.Runtime.InteropServices;
using Loralei.Core.Lora;
using Xunit;

namespace Loralei.Core.Tests;

public class LoraFormatTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("loralei-lora").FullName;
    private readonly Random _random = new(7);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private Tensor Randn(int rows, int columns)
    {
        var values = new float[rows * columns];
        for (var i = 0; i < values.Length; i++) values[i] = (float)(_random.NextDouble() * 2 - 1);
        return new Tensor("F32", [rows, columns], MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
    }

    private Dictionary<string, Tensor> Triple(string prefix, int rank = 4, int outFeatures = 8, int inFeatures = 6)
    {
        var state = new Dictionary<string, Tensor>();
        foreach (var part in new[] { "to_q", "to_k", "to_v" })
        {
            state[$"{prefix}.{part}.lora_A.weight"] = Randn(rank, inFeatures);
            state[$"{prefix}.{part}.lora_B.weight"] = Randn(outFeatures, rank);
        }
        return state;
    }

    [Fact]
    public void ComfyAddressesTheSameTensorsUnderItsOwnPrefix()
    {
        var state = Triple("transformer.transformer_blocks.0.attn");
        state["transformer.blocks.3.ffn.net.2.lora_A.weight"] = Randn(4, 8);
        var converted = ComfyConverter.ToComfy(state);

        Assert.Equal(state.Count, converted.Count);
        Assert.All(converted.Keys, key => Assert.StartsWith("diffusion_model.", key));
        Assert.Contains("diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight", converted.Keys);
        Assert.Contains("diffusion_model.blocks.3.ffn.net.2.lora_A.weight", converted.Keys);
        // The bytes are the adapter's own: nothing is fused, scaled or re-encoded.
        Assert.Same(state["transformer.transformer_blocks.0.attn.to_q.lora_B.weight"].Data,
            converted["diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight"].Data);
    }

    [Fact]
    public void AFileSurvivesTheRoundTripWithItsMetadata()
    {
        var path = Path.Combine(_dir, "a.safetensors");
        var original = new SafeTensorsFile { Tensors = Triple("transformer.blocks.0.attn1") };
        original.Metadata["model"] = "krea-2-raw";
        original.Metadata["config"] = "{\"rank\": 4, \"note\": \"ünïcode\"}";
        original.Save(path);

        Assert.Equal(original.Tensors.Keys.Order(), SafeTensorsFile.Keys(path).Order());
        var loaded = SafeTensorsFile.Load(path);
        Assert.Equal(original.Metadata, loaded.Metadata);
        foreach (var (name, tensor) in original.Tensors)
        {
            Assert.Equal(tensor.Shape, loaded.Tensors[name].Shape);
            Assert.Equal(tensor.Data, loaded.Tensors[name].Data);
        }

        // The layout other readers rely on: an 8-byte length, then a header padded to 8 bytes.
        var bytes = File.ReadAllBytes(path);
        var headerLength = BitConverter.ToInt64(bytes, 0);
        Assert.Equal(0, headerLength % 8);
        Assert.Equal((byte)'{', bytes[8]);
        Assert.Equal(8 + headerLength + original.Tensors.Values.Sum(t => t.Data.Length), bytes.Length);
    }

    [Fact]
    public void ConvertFileWritesTheComfyFormBesideTheSource()
    {
        var source = Path.Combine(_dir, "run.safetensors");
        var file = new SafeTensorsFile { Tensors = Triple("transformer.blocks.0.attn1") };
        file.Metadata["rank"] = "4";
        file.Save(source);

        var target = Path.Combine(_dir, "run_comfy.safetensors");
        Assert.Equal(6, ComfyConverter.ConvertFile(source, target));

        var converted = SafeTensorsFile.Load(target);
        Assert.Contains("diffusion_model.blocks.0.attn1.to_k.lora_A.weight", converted.Tensors.Keys);
        Assert.Equal(file.Tensors["transformer.blocks.0.attn1.to_k.lora_A.weight"].Data,
            converted.Tensors["diffusion_model.blocks.0.attn1.to_k.lora_A.weight"].Data);
        Assert.Equal("comfyui", converted.Metadata["format"]);
        Assert.Equal("run.safetensors", converted.Metadata["source"]);
        Assert.Equal("4", converted.Metadata["rank"]);
    }

    [Fact]
    public void GarbageIsRefusedRatherThanRead()
    {
        var path = Path.Combine(_dir, "bad.safetensors");
        File.WriteAllBytes(path, [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F, 1, 2, 3]);
        Assert.Throws<InvalidDataException>(() => SafeTensorsFile.Load(path));
        File.WriteAllText(path, "short");
        Assert.Throws<InvalidDataException>(() => SafeTensorsFile.Keys(path));
    }
}
