namespace Loralei.Core.Lora;

/// <summary>
/// Converts a diffusers-format LoRA into the key layout ComfyUI expects: the same tensors,
/// addressed under <c>diffusion_model.</c> instead of <c>transformer.</c>. None of the current
/// models needs more than that; the conversion never touches a tensor's bytes.
/// </summary>
public static class ComfyConverter
{
    public const string ComfyPrefix = "diffusion_model.";
    private const string DiffusersPrefix = "transformer.";

    public static Dictionary<string, Tensor> ToComfy(IReadOnlyDictionary<string, Tensor> state) =>
        state.ToDictionary(
            p => ComfyPrefix + (p.Key.StartsWith(DiffusersPrefix, StringComparison.Ordinal)
                ? p.Key[DiffusersPrefix.Length..]
                : p.Key),
            p => p.Value, StringComparer.Ordinal);

    /// <summary>Writes the ComfyUI form of <paramref name="source"/> and returns how many tensors it holds.</summary>
    public static int ConvertFile(string source, string target)
    {
        var file = SafeTensorsFile.Load(source);
        var converted = new SafeTensorsFile { Tensors = ToComfy(file.Tensors) };
        foreach (var (key, value) in file.Metadata) converted.Metadata[key] = value;
        converted.Metadata["format"] = "comfyui";
        converted.Metadata["source"] = Path.GetFileName(source);
        converted.Save(target);
        return converted.Tensors.Count;
    }
}
