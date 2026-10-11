namespace Loralei.DatasetMaker.Config;

public sealed record AppConfig
{
    public OpenAiOptions OpenAI { get; init; } = new();
    public CaptionOptions Caption { get; init; } = new();
}

public sealed record OpenAiOptions
{
    public string BaseUrl { get; init; } = "https://api.openai.com/v1";
    public string ApiKey { get; init; } = "";
    public string Model { get; init; } = "gpt-4.1-mini";
    public int MaxTokens { get; init; } = 400;
    public double Temperature { get; init; } = 0.3;
    public int TimeoutSeconds { get; init; } = 120;

    /// <summary>OpenAI image detail level: low, high or auto.</summary>
    public string ImageDetail { get; init; } = "high";
}

public sealed record CaptionOptions
{
    public const string DefaultSystemPrompt =
        "You write captions for images in a LoRA training dataset for modern text-to-image and text-to-video models " +
        "(Z-Image, Qwen-Image, LTX). Write a single detailed paragraph of natural English prose in the present tense. " +
        "Do not start with phrases like \"The image shows\", \"This is a picture of\" or \"In this image\"; start directly with the main subject. " +
        "Do not use lists, headings, markdown or quotation marks around the caption. " +
        "Describe, in this order: the main subject (appearance, hair, clothing, accessories, facial expression, pose and action), " +
        "other notable objects, the setting and background, lighting, colors and atmosphere, " +
        "camera angle, framing and composition (e.g. close-up, medium shot, full body), " +
        "and finally the style or medium (e.g. photograph, digital illustration, 3D render, anime). " +
        "Be objective and specific; only describe what is visible. Do not guess identities, names, ages or emotions that are not clearly shown. " +
        "If there is readable text in the image, quote it exactly. Output only the caption.";

    public string SystemPrompt { get; init; } = DefaultSystemPrompt;

    /// <summary>Extra context appended to the system prompt, e.g. "The person in the images is John Smith."</summary>
    public string ExtraSystemPrompt { get; init; } = "";

    public string TriggerWord { get; init; } = "";
    public int MaxParallel { get; init; } = 2;
    public int MaxRetries { get; init; } = 3;
}
