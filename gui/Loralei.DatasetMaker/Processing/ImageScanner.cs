namespace Loralei.DatasetMaker.Processing;

public static class ImageScanner
{
    private static readonly HashSet<string> Extensions =
        new([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"], StringComparer.OrdinalIgnoreCase);

    public static List<string> FindImages(string directory) =>
        Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories)
            .Where(f => Extensions.Contains(Path.GetExtension(f)))
            .Order(StringComparer.Ordinal)
            .ToList();
}
