using System.Text.RegularExpressions;

namespace Loralei.DatasetMaker.Captioning;

public static partial class PromptBuilder
{
    public const string UserPrompt = "Write the training caption for this image.";

    public static string BuildSystemPrompt(string systemPrompt, string? extraContext)
    {
        if (string.IsNullOrWhiteSpace(extraContext))
            return systemPrompt;

        return $"""
            {systemPrompt}

            Additional context: {extraContext.Trim()}
            Use this information in the caption (for example, refer to the person by that name) instead of generic terms like "a man" or "a woman".
            """;
    }

    /// <summary>Normalizes the model output into a single-paragraph caption, prefixed with the trigger word if given.</summary>
    public static string PostProcess(string raw, string? triggerWord)
    {
        var caption = raw.Trim();

        // Strip reasoning blocks some local models emit.
        caption = ThinkBlockRegex().Replace(caption, "").Trim();
        // Strip code fences / markdown emphasis.
        caption = caption.Replace("```", "").Replace("**", "").Trim();
        // Collapse all whitespace (including newlines) into single spaces.
        caption = WhitespaceRegex().Replace(caption, " ");
        caption = caption.Trim('"', '\'', '“', '”', ' ');
        // Remove "The image shows ..." style openings.
        var match = PreambleRegex().Match(caption);
        if (match.Success)
        {
            caption = caption[match.Length..];
            if (caption.Length > 0)
                caption = char.ToUpperInvariant(caption[0]) + caption[1..];
        }

        if (!string.IsNullOrWhiteSpace(triggerWord))
            caption = $"{triggerWord.Trim()}, {caption}";

        return caption;
    }

    [GeneratedRegex(@"<think>.*?</think>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex ThinkBlockRegex();

    [GeneratedRegex(@"\s+")]
    private static partial Regex WhitespaceRegex();

    [GeneratedRegex(@"^(caption:\s*|(the|this) (image|picture|photo|photograph) (shows|depicts|features|captures|is of)\s+|in (the|this) (image|picture|photo),?\s+)",
        RegexOptions.IgnoreCase)]
    private static partial Regex PreambleRegex();
}
