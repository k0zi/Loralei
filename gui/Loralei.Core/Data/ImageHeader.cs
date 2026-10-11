using System.Buffers.Binary;
using System.Text;

namespace Loralei.Core.Data;

/// <summary>
/// Reads an image's pixel dimensions from its header, without decoding it.
/// <para>
/// The dataset scan needs width and height for every file, and nothing else - the pixels are
/// only ever decoded by the worker, when latents are cached. Parsing five headers here keeps
/// this assembly free of an imaging library. The formats are exactly the ones the worker's
/// Pillow can open: JPEG XL is on the suffix list but has no decoder there, so it is reported
/// as unreadable here too rather than failing later, mid-cache.
/// </para>
/// </summary>
public static class ImageHeader
{
    private static ReadOnlySpan<byte> PngSignature => [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    /// <exception cref="InvalidDataException">The file is not a readable image of a known format.</exception>
    public static (int Width, int Height) Read(string path)
    {
        using var stream = File.OpenRead(path);
        var head = new byte[32];
        var read = stream.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
        var h = head.AsSpan(0, read);

        (int Width, int Height) size;
        if (h.StartsWith(PngSignature)) size = Png(h);
        else if (h.Length >= 2 && h[0] == 0xFF && h[1] == 0xD8) size = Jpeg(stream);
        else if (h.Length >= 12 && h[..4].SequenceEqual("RIFF"u8) && h[8..12].SequenceEqual("WEBP"u8)) size = WebP(stream);
        else if (h.Length >= 2 && h[0] == 'B' && h[1] == 'M') size = Bmp(h);
        else if (h.Length >= 12 && h[4..8].SequenceEqual("ftyp"u8)) size = Avif(stream);
        else throw new InvalidDataException("cannot identify image file");

        if (size.Width <= 0 || size.Height <= 0) throw new InvalidDataException("image has no size");
        return size;
    }

    private static (int, int) Png(ReadOnlySpan<byte> h)
    {
        if (h.Length < 24 || !h[12..16].SequenceEqual("IHDR"u8)) throw new InvalidDataException("truncated PNG header");
        return (BinaryPrimitives.ReadInt32BigEndian(h[16..]), BinaryPrimitives.ReadInt32BigEndian(h[20..]));
    }

    private static (int, int) Bmp(ReadOnlySpan<byte> h)
    {
        if (h.Length < 26) throw new InvalidDataException("truncated BMP header");
        return BinaryPrimitives.ReadInt32LittleEndian(h[14..]) == 12
            ? (BinaryPrimitives.ReadUInt16LittleEndian(h[18..]), BinaryPrimitives.ReadUInt16LittleEndian(h[20..]))
            : (BinaryPrimitives.ReadInt32LittleEndian(h[18..]), Math.Abs(BinaryPrimitives.ReadInt32LittleEndian(h[22..])));
    }

    private static (int, int) Jpeg(Stream stream)
    {
        stream.Position = 2;
        Span<byte> buffer = stackalloc byte[7];
        while (true)
        {
            var lead = stream.ReadByte();
            if (lead < 0) throw new InvalidDataException("truncated JPEG: no frame header");
            if (lead != 0xFF) throw new InvalidDataException("corrupt JPEG: marker expected");

            int marker;
            do marker = stream.ReadByte(); while (marker == 0xFF); // fill bytes
            if (marker is < 0 or 0xD9) throw new InvalidDataException("truncated JPEG: no frame header");
            if (marker is 0x00 or 0x01 or (>= 0xD0 and <= 0xD8)) continue; // markers that carry no length

            Exactly(stream, buffer[..2]);
            var length = BinaryPrimitives.ReadUInt16BigEndian(buffer);
            // SOF0..SOF15, minus the three in that range that are tables rather than frames.
            if (marker is >= 0xC0 and <= 0xCF and not (0xC4 or 0xC8 or 0xCC))
            {
                Exactly(stream, buffer[..5]);
                return (BinaryPrimitives.ReadUInt16BigEndian(buffer[3..]), BinaryPrimitives.ReadUInt16BigEndian(buffer[1..]));
            }
            stream.Seek(length - 2, SeekOrigin.Current);
        }
    }

    private static (int, int) WebP(Stream stream)
    {
        stream.Position = 12;
        Span<byte> b = stackalloc byte[18];
        Exactly(stream, b);
        var chunk = b[..4];
        var data = b[8..];
        if (chunk.SequenceEqual("VP8X"u8))
            return (1 + (data[4] | data[5] << 8 | data[6] << 16), 1 + (data[7] | data[8] << 8 | data[9] << 16));
        if (chunk.SequenceEqual("VP8L"u8))
        {
            if (data[0] != 0x2F) throw new InvalidDataException("corrupt WebP: bad lossless signature");
            return (1 + (data[1] | (data[2] & 0x3F) << 8),
                    1 + (data[2] >> 6 | data[3] << 2 | (data[4] & 0x0F) << 10));
        }
        if (chunk.SequenceEqual("VP8 "u8))
        {
            if (data[3] != 0x9D || data[4] != 0x01 || data[5] != 0x2A)
                throw new InvalidDataException("corrupt WebP: bad frame start code");
            return (BinaryPrimitives.ReadUInt16LittleEndian(data[6..]) & 0x3FFF,
                    BinaryPrimitives.ReadUInt16LittleEndian(data[8..]) & 0x3FFF);
        }
        throw new InvalidDataException("unsupported WebP layout");
    }

    /// <summary>
    /// AVIF keeps its size in an <c>ispe</c> property box under <c>meta/iprp/ipco</c>. A file
    /// can hold several (tiles of a grid, an alpha plane); the image itself is the largest.
    /// </summary>
    private static (int, int) Avif(Stream stream)
    {
        stream.Position = 0;
        var ftyp = NextBox(stream, stream.Length) ?? throw new InvalidDataException("truncated AVIF");
        Span<byte> brand = stackalloc byte[4];
        Exactly(stream, brand);
        var brands = Encoding.ASCII.GetString(brand);
        if (brands is not ("avif" or "avis" or "mif1" or "msf1"))
            throw new InvalidDataException($"cannot identify image file (ftyp brand '{brands}')");
        stream.Position = ftyp.End;

        (int W, int H) best = (0, 0);
        var rotated = false;
        Walk(stream, stream.Length, 0);
        if (best.W == 0) throw new InvalidDataException("AVIF has no image size property");
        return rotated ? (best.H, best.W) : best;

        void Walk(Stream s, long end, int depth)
        {
            Span<byte> payload = stackalloc byte[12];
            while (s.Position + 8 <= end && NextBox(s, end) is { } box)
            {
                switch (box.Type)
                {
                    case "meta" when depth == 0:
                        s.Seek(4, SeekOrigin.Current); // FullBox version and flags
                        Walk(s, box.End, 1);
                        break;
                    case "iprp" when depth == 1:
                    case "ipco" when depth == 2:
                        Walk(s, box.End, depth + 1);
                        break;
                    case "ispe" when depth == 3:
                        Exactly(s, payload);
                        var w = BinaryPrimitives.ReadInt32BigEndian(payload[4..]);
                        var h = BinaryPrimitives.ReadInt32BigEndian(payload[8..]);
                        if ((long)w * h > (long)best.W * best.H) best = (w, h);
                        break;
                    case "irot" when depth == 3:
                        rotated = (s.ReadByte() & 1) == 1; // 90 or 270 degrees
                        break;
                }
                s.Position = box.End;
            }
        }
    }

    private readonly record struct Box(string Type, long End);

    private static Box? NextBox(Stream stream, long limit)
    {
        Span<byte> header = stackalloc byte[8];
        if (stream.ReadAtLeast(header, 8, throwOnEndOfStream: false) < 8) return null;
        long size = BinaryPrimitives.ReadUInt32BigEndian(header);
        var start = stream.Position - 8;
        if (size == 1)
        {
            Span<byte> large = stackalloc byte[8];
            Exactly(stream, large);
            size = (long)BinaryPrimitives.ReadUInt64BigEndian(large);
        }
        else if (size == 0) size = limit - start;
        if (size < 8) throw new InvalidDataException("corrupt box size");
        return new Box(Encoding.ASCII.GetString(header[4..]), Math.Min(start + size, limit));
    }

    private static void Exactly(Stream stream, Span<byte> buffer)
    {
        if (stream.ReadAtLeast(buffer, buffer.Length, throwOnEndOfStream: false) < buffer.Length)
            throw new InvalidDataException("image file is truncated");
    }
}
