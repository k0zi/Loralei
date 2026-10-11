using Xunit;

namespace Loralei.Core.Tests;

public class FlatYamlTests
{
    /// <summary>The workspace file as it ships in this repo, comments and all.</summary>
    private const string ProjectConfig =
        """
        # Where the base models live. A single path or a list; Hugging Face cache layout
        # (models--org--repo/snapshots/...) and plain <root>/<model>/<component> both work.
        models_dir:
          - ~/Develop/ai-models/unsloth/hub
          - ~/Develop/ai-toolkit/hf-cache/huggingface/hub

        sample_every: 0
        sample_prompts:
          - "a photo of sks pattern on a wooden table"

        """;

    [Fact]
    public void ReadsScalarsSequencesAndComments()
    {
        var doc = FlatYaml.Parse(ProjectConfig);

        Assert.Equal(0, doc.Int("sample_every"));
        Assert.Equal(
            ["~/Develop/ai-models/unsloth/hub", "~/Develop/ai-toolkit/hf-cache/huggingface/hub"],
            doc.Sequence("models_dir"));
        Assert.Equal(["a photo of sks pattern on a wooden table"], doc.Sequence("sample_prompts"));
    }

    [Fact]
    public void RoundTripsAnUntouchedDocumentByteForByte()
    {
        Assert.Equal(ProjectConfig, FlatYaml.Parse(ProjectConfig).ToString());
    }

    [Fact]
    public void EditingOneKeyPreservesEverythingElse()
    {
        var doc = FlatYaml.Parse(ProjectConfig);
        doc.Set("sample_every", 250);
        var text = doc.ToString();

        // The user's comments and model roots must survive a GUI write.
        Assert.Contains("# Where the base models live.", text);
        Assert.Contains("  - ~/Develop/ai-models/unsloth/hub", text);
        Assert.Contains("sample_every: 250", text);
        Assert.DoesNotContain("sample_every: 0", text);
        Assert.Equal(["a photo of sks pattern on a wooden table"],
            FlatYaml.Parse(text).Sequence("sample_prompts"));
    }

    [Fact]
    public void RemovingAKeyLetsThePresetDefaultComeBack()
    {
        var doc = FlatYaml.Parse(ProjectConfig);
        doc.Remove("sample_every");
        var text = doc.ToString();

        Assert.DoesNotContain("sample_every", text);
        Assert.Contains("sample_prompts:", text);
    }

    [Fact]
    public void ReplacingASequenceRewritesAllItsItems()
    {
        var doc = FlatYaml.Parse(ProjectConfig);
        doc.Set("sample_prompts", new[] { "one", "two" });
        var reparsed = FlatYaml.Parse(doc.ToString());

        Assert.Equal(["one", "two"], reparsed.Sequence("sample_prompts"));
    }

    [Fact]
    public void EmptySequenceIsWrittenInFlowForm()
    {
        var doc = FlatYaml.Parse("");
        doc.Set("sample_prompts", Array.Empty<string>());
        Assert.Contains("sample_prompts: []", doc.ToString());
        Assert.Empty(FlatYaml.Parse(doc.ToString()).Sequence("sample_prompts"));
    }

    [Theory]
    [InlineData("a caption with: a colon")]
    [InlineData("#not-a-comment")]
    [InlineData("no")]          // would otherwise parse as the boolean false
    [InlineData("1.0")]         // would otherwise parse as a number
    [InlineData("he said \"hi\"")]
    [InlineData("back\\slash")]
    public void QuotesScalarsThatWouldOtherwiseChangeType(string value)
    {
        var doc = FlatYaml.Parse("");
        doc.Set("trigger_word", value);
        Assert.Equal(value, FlatYaml.Parse(doc.ToString()).Scalar("trigger_word"));
    }

    [Fact]
    public void KeepsSmallFloatsExact()
    {
        var doc = FlatYaml.Parse("");
        doc.Set("learning_rate", 5e-05);
        Assert.Equal(5e-05, FlatYaml.Parse(doc.ToString()).Double("learning_rate"));
    }

    [Fact]
    public void IgnoresTrailingComments()
    {
        var doc = FlatYaml.Parse("rank: 32  # bumped for this dataset\n");
        Assert.Equal(32, doc.Int("rank"));
    }

    [Fact]
    public void ReadsBooleansInEveryYamlSpelling()
    {
        var doc = FlatYaml.Parse("a: true\nb: false\nc: yes\nd: off\n");
        Assert.True(doc.Bool("a"));
        Assert.False(doc.Bool("b"));
        Assert.True(doc.Bool("c"));
        Assert.False(doc.Bool("d"));
    }

    [Fact]
    public void NullScalarReadsAsAbsent()
    {
        // config.resolved.yaml writes unset optionals as an explicit null.
        Assert.Equal("null", FlatYaml.Parse("alpha: null\n").Scalar("alpha"));
        Assert.Null(FlatYaml.Parse("alpha: null\n").Double("alpha"));
    }
}
