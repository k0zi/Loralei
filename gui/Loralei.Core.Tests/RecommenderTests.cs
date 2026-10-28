using Loralei.Core;
using Loralei.Core.Data;
using Xunit;

namespace Loralei.Core.Tests;

public class RecommenderTests
{
    [Fact]
    public void CustomRecommendsNothing()
    {
        Assert.Equal(Recommendation.None, Recommender.For("qwen-image", Recommender.Custom, 40));
    }

    [Fact]
    public void RunLengthFollowsTheImageCountInWholeEpochs()
    {
        var rec = Recommender.For("krea-2-raw", Recommender.Character, 20);

        Assert.Equal(100, rec.Epochs);
        Assert.Equal(2000, rec.Steps);
        Assert.Equal(0, rec.SaveEvery % 20);
        Assert.Equal(1, rec.BatchSize);
        Assert.Equal(1, rec.Repeats);
    }

    [Theory]
    [InlineData(Recommender.Character, 4, 1000)]     // floor: a tiny set still has to converge
    [InlineData(Recommender.Character, 500, 3000)]   // ceiling: a large one does not need every view
    [InlineData(Recommender.Style, 4, 1500)]
    [InlineData(Recommender.Style, 500, 4000)]
    public void RunLengthIsClamped(string kind, int images, int steps)
    {
        Assert.Equal(steps, Recommender.For("qwen-image", kind, images).Steps);
    }

    [Fact]
    public void StyleAndCharacterDiffer()
    {
        var style = Recommender.For("qwen-image", Recommender.Style, 30);
        var character = Recommender.For("qwen-image", Recommender.Character, 30);

        Assert.True(style.Rank > character.Rank);
        Assert.Equal(0.1, style.CaptionDropout);
        Assert.Null(character.CaptionDropout);
        Assert.Contains("trigger word", character.Summary);
    }

    [Fact]
    public void LearningRateIsTheModelsOwn()
    {
        Assert.Equal(5.0e-5, Recommender.For("qwen-image", Recommender.Style, 30).LearningRate);
        Assert.Equal(1.0e-4, Recommender.For("krea-2-raw", Recommender.Style, 30).LearningRate);
    }

    [Fact]
    public void WithoutImagesOnlyTheSizeIndependentValuesAreSet()
    {
        var rec = Recommender.For("qwen-image", Recommender.Style, 0);

        Assert.Equal(32, rec.Rank);
        Assert.Null(rec.Steps);
        Assert.Null(rec.Epochs);
    }

    [Theory]
    [InlineData(20, 1, 1, 20)]
    [InlineData(20, 5, 1, 100)]
    [InlineData(21, 1, 4, 6)]     // the short last batch is kept, as the trainer keeps it
    [InlineData(0, 1, 1, 0)]
    public void StepsPerEpochCountsBatches(int images, int repeats, int batch, int expected)
    {
        Assert.Equal(expected, Recommender.StepsPerEpoch(images, repeats, batch));
    }

    [Fact]
    public void EpochsAndStepsConvertBothWays()
    {
        Assert.Equal(600, Recommender.StepsForEpochs(30, 20));
        Assert.Equal(30, Recommender.EpochsForSteps(600, 20));
        Assert.Equal(31, Recommender.EpochsForSteps(601, 20));
        Assert.Equal(0, Recommender.EpochsForSteps(600, 0));
    }

    [Fact]
    public void CountsImagesBySuffixIncludingSubfolders()
    {
        var dir = Directory.CreateTempSubdirectory("loralei-count").FullName;
        try
        {
            Directory.CreateDirectory(Path.Combine(dir, "sub"));
            File.WriteAllText(Path.Combine(dir, "a.png"), "");
            File.WriteAllText(Path.Combine(dir, "a.txt"), "");
            File.WriteAllText(Path.Combine(dir, "b.JPG"), "");
            File.WriteAllText(Path.Combine(dir, "sub", "c.webp"), "");

            Assert.Equal(3, DatasetScanner.CountImages(dir));
            Assert.Equal(0, DatasetScanner.CountImages(Path.Combine(dir, "missing")));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }
}
