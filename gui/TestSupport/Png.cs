namespace Loralei.Tests;

/// <summary>
/// Writes a minimal 8-bit greyscale PNG. The tests need real image files for loralei to scan,
/// and this avoids pulling an imaging library into the test projects for four throwaway files.
/// </summary>
internal static class Png
{
    /// <param name="seed">
    /// Varies the pixels between files. loralei's dataset scan drops exact duplicates by
    /// content hash, so images that differ only in filename are silently discarded.
    /// </param>
    public static void Write(string path, int width, int height, int seed = 0)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A });

        var header = new List<byte>();
        header.AddRange(BigEndian(width));
        header.AddRange(BigEndian(height));
        header.AddRange([8, 0, 0, 0, 0]); // 8-bit, greyscale, no interlace
        Chunk(writer, "IHDR", header.ToArray());

        var raw = new List<byte>();
        for (var y = 0; y < height; y++)
        {
            raw.Add(0); // filter type "none" for this scanline
            for (var x = 0; x < width; x++) raw.Add((byte)((x * 4 + y * 3 + seed * 37) % 256));
        }
        Chunk(writer, "IDAT", ZlibStored(raw.ToArray()));
        Chunk(writer, "IEND", []);
    }

    private static byte[] BigEndian(int value) =>
        [(byte)(value >> 24), (byte)(value >> 16), (byte)(value >> 8), (byte)value];

    private static void Chunk(BinaryWriter writer, string type, byte[] data)
    {
        var typeBytes = System.Text.Encoding.ASCII.GetBytes(type);
        writer.Write(BigEndian(data.Length));
        writer.Write(typeBytes);
        writer.Write(data);
        writer.Write(BigEndian(unchecked((int)Crc32([.. typeBytes, .. data]))));
    }

    /// <summary>Stored (uncompressed) deflate blocks in a zlib wrapper with an Adler-32 trailer.</summary>
    private static byte[] ZlibStored(byte[] data)
    {
        var output = new List<byte> { 0x78, 0x01 };
        for (var offset = 0; offset < data.Length; offset += 0xFFFF)
        {
            var length = Math.Min(0xFFFF, data.Length - offset);
            output.Add((byte)(offset + length >= data.Length ? 1 : 0));
            output.AddRange([(byte)length, (byte)(length >> 8), (byte)~length, (byte)(~length >> 8)]);
            output.AddRange(data.Skip(offset).Take(length));
        }

        uint a = 1, b = 0;
        foreach (var value in data) { a = (a + value) % 65521; b = (b + a) % 65521; }
        output.AddRange(BigEndian(unchecked((int)((b << 16) | a))));
        return output.ToArray();
    }

    private static uint Crc32(byte[] data)
    {
        var crc = 0xFFFFFFFFu;
        foreach (var value in data)
        {
            crc ^= value;
            for (var bit = 0; bit < 8; bit++) crc = (crc >> 1) ^ (0xEDB88320u & (uint)(-(crc & 1)));
        }
        return crc ^ 0xFFFFFFFFu;
    }
}
