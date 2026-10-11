using Loralei.DatasetMaker.Captioning;

namespace Loralei.DatasetMaker.Tests;

public class PromptBuilderTests
{
    [Fact]
    public void Without_context_the_system_prompt_is_left_alone() =>
        Assert.Equal("style", PromptBuilder.BuildSystemPrompt("style", "  "));

    [Fact]
    public void Context_is_appended_to_the_system_prompt()
    {
        var prompt = PromptBuilder.BuildSystemPrompt("style", " The person is John Smith. ");

        Assert.StartsWith("style", prompt);
        Assert.Contains("Additional context: The person is John Smith.", prompt);
    }

    [Theory]
    [InlineData("  A cat\n\nsits   on a roof. ", "A cat sits on a roof.")]
    [InlineData("<think>hmm,\nlet me see</think>A cat on a roof.", "A cat on a roof.")]
    [InlineData("\"A cat on a roof.\"", "A cat on a roof.")]
    [InlineData("**A cat** on a roof.", "A cat on a roof.")]
    [InlineData("The image shows a cat on a roof.", "A cat on a roof.")]
    [InlineData("In this picture, a cat sits.", "A cat sits.")]
    [InlineData("Caption: a cat on a roof.", "A cat on a roof.")]
    public void Model_output_is_normalised_into_one_paragraph(string raw, string expected) =>
        Assert.Equal(expected, PromptBuilder.PostProcess(raw, null));

    [Fact]
    public void The_trigger_word_goes_first() =>
        Assert.Equal("ohwx, A cat on a roof.", PromptBuilder.PostProcess("A cat on a roof.", " ohwx "));
}
