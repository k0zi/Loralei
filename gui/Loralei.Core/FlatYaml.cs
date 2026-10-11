using System.Globalization;
using System.Text;

namespace Loralei.Core;

/// <summary>
/// A minimal reader/writer for the one YAML shape loralei uses: a flat mapping of scalars
/// and plain string sequences. <c>loralei.yaml</c> and <c>config.resolved.yaml</c> contain
/// nothing else - no nesting, anchors, flow collections or multi-line scalars.
/// <para>
/// Editing is line-preserving on purpose. The workspace <c>loralei.yaml</c> is the user's
/// file, often with comments and hand-tuned <c>models_dir</c> entries; the GUI rewrites
/// only the keys it manages and leaves everything else exactly as it found it.
/// </para>
/// </summary>
public sealed class FlatYaml
{
    private abstract record Node;
    private sealed record Verbatim(string Line) : Node;
    private sealed record Entry(string Key, List<string> Lines) : Node;

    private readonly List<Node> _nodes = [];

    public static FlatYaml Parse(string text)
    {
        var doc = new FlatYaml();
        var lines = text.Replace("\r\n", "\n").Split('\n');
        Entry? current = null;

        foreach (var line in lines)
        {
            var trimmed = line.TrimStart();

            // A sequence item or an indented continuation belongs to the key above it.
            if (current is not null && line.Length > 0 && char.IsWhiteSpace(line[0]))
            {
                current.Lines.Add(line);
                continue;
            }

            current = null;
            if (trimmed.Length == 0 || trimmed.StartsWith('#'))
            {
                doc._nodes.Add(new Verbatim(line));
                continue;
            }

            var colon = FindKeySeparator(line);
            if (colon < 0)
            {
                doc._nodes.Add(new Verbatim(line));
                continue;
            }

            var key = line[..colon].Trim();
            current = new Entry(key, [line]);
            doc._nodes.Add(current);
        }
        return doc;
    }

    /// <summary>Index of the <c>:</c> that ends the key, ignoring one inside a quoted key.</summary>
    private static int FindKeySeparator(string line)
    {
        var quote = '\0';
        for (var i = 0; i < line.Length; i++)
        {
            var ch = line[i];
            if (quote != '\0') { if (ch == quote) quote = '\0'; continue; }
            if (ch is '"' or '\'') { quote = ch; continue; }
            if (ch == '#') return -1;
            if (ch == ':' && (i + 1 == line.Length || line[i + 1] is ' ' or '\t')) return i;
        }
        return -1;
    }

    public bool Has(string key) => _nodes.OfType<Entry>().Any(e => e.Key == key);

    /// <summary>Every key in the document, in file order, each once.</summary>
    public IReadOnlyList<string> Keys => _nodes.OfType<Entry>().Select(e => e.Key).Distinct().ToList();

    /// <summary>True when the key holds a block sequence (or an inline <c>[]</c>) rather than a scalar.</summary>
    public bool IsSequence(string key)
    {
        var entry = _nodes.OfType<Entry>().LastOrDefault(e => e.Key == key);
        if (entry is null) return false;
        var inline = Scalar(key);
        return inline is null ? entry.Lines.Skip(1).Any(l => l.TrimStart().StartsWith("- ")) : inline == "[]";
    }

    /// <summary>The raw scalar after <c>key:</c>, unquoted; null when absent or a sequence.</summary>
    public string? Scalar(string key)
    {
        var entry = _nodes.OfType<Entry>().LastOrDefault(e => e.Key == key);
        if (entry is null) return null;
        var head = entry.Lines[0];
        var value = head[(FindKeySeparator(head) + 1)..].Trim();
        value = StripComment(value);
        return value.Length == 0 ? null : Unquote(value);
    }

    public IReadOnlyList<string> Sequence(string key)
    {
        var entry = _nodes.OfType<Entry>().LastOrDefault(e => e.Key == key);
        if (entry is null) return [];

        var inline = Scalar(key);
        if (inline is not null)
            return inline is "[]" ? [] : [inline];

        return entry.Lines.Skip(1)
            .Select(l => l.TrimStart())
            .Where(l => l.StartsWith("- "))
            .Select(l => Unquote(StripComment(l[2..].Trim())))
            .ToList();
    }

    public bool? Bool(string key) => Scalar(key)?.ToLowerInvariant() switch
    {
        "true" or "yes" or "on" => true,
        "false" or "no" or "off" => false,
        _ => null,
    };

    public int? Int(string key) =>
        int.TryParse(Scalar(key), NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v : null;

    public double? Double(string key) =>
        double.TryParse(Scalar(key), NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : null;

    /// <summary>Replaces the key in place, or appends it when new.</summary>
    public void Set(string key, object? value)
    {
        var lines = Render(key, value);
        var index = _nodes.FindLastIndex(n => n is Entry e && e.Key == key);
        if (index >= 0) _nodes[index] = new Entry(key, lines);
        else _nodes.Add(new Entry(key, lines));
    }

    public void Remove(string key) => _nodes.RemoveAll(n => n is Entry e && e.Key == key);

    private static List<string> Render(string key, object? value)
    {
        switch (value)
        {
            case null:
                return [$"{key}: null"];
            case bool b:
                return [$"{key}: {(b ? "true" : "false")}"];
            case int or long:
                return [$"{key}: {Convert.ToString(value, CultureInfo.InvariantCulture)}"];
            case float or double or decimal:
            {
                var d = Convert.ToDouble(value, CultureInfo.InvariantCulture);
                // "R" keeps 5e-05 exact; YAML reads both plain and exponent forms as floats.
                return [$"{key}: {d.ToString("R", CultureInfo.InvariantCulture)}"];
            }
            case IEnumerable<string> seq:
            {
                var items = seq.ToList();
                if (items.Count == 0) return [$"{key}: []"];
                var lines = new List<string> { $"{key}:" };
                lines.AddRange(items.Select(i => $"  - {Quote(i)}"));
                return lines;
            }
            default:
                return [$"{key}: {Quote(Convert.ToString(value, CultureInfo.InvariantCulture) ?? "")}"];
        }
    }

    /// <summary>
    /// Always double-quotes. Plain scalars would need loralei's own type coercion rules to
    /// emit safely (a caption of "1.0", a trigger word of "no"), and quoting sidesteps all
    /// of it: the config binder coerces the string back to the declared field type anyway.
    /// </summary>
    private static string Quote(string value)
    {
        var sb = new StringBuilder("\"");
        foreach (var ch in value)
        {
            sb.Append(ch switch
            {
                '"' => "\\\"",
                '\\' => "\\\\",
                '\n' => "\\n",
                '\r' => "\\r",
                '\t' => "\\t",
                _ => ch.ToString(),
            });
        }
        return sb.Append('"').ToString();
    }

    private static string StripComment(string value)
    {
        if (value.StartsWith('"') || value.StartsWith('\'')) return value;
        var hash = value.IndexOf(" #", StringComparison.Ordinal);
        return hash < 0 ? value : value[..hash].TrimEnd();
    }

    private static string Unquote(string value)
    {
        if (value.Length < 2) return value;
        if (value[0] == '\'' && value[^1] == '\'') return value[1..^1].Replace("''", "'");
        if (value[0] != '"' || value[^1] != '"') return value;

        var sb = new StringBuilder();
        for (var i = 1; i < value.Length - 1; i++)
        {
            if (value[i] != '\\') { sb.Append(value[i]); continue; }
            i++;
            if (i >= value.Length - 1) break;
            sb.Append(value[i] switch
            {
                'n' => '\n', 'r' => '\r', 't' => '\t', '0' => '\0',
                var other => other,
            });
        }
        return sb.ToString();
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        foreach (var node in _nodes)
        {
            switch (node)
            {
                case Verbatim v: sb.Append(v.Line).Append('\n'); break;
                case Entry e: foreach (var l in e.Lines) sb.Append(l).Append('\n'); break;
            }
        }
        var text = sb.ToString();
        // Parsing a trailing newline yields one empty final line; do not let it accumulate.
        return text.EndsWith("\n\n", StringComparison.Ordinal) ? text[..^1] : text;
    }
}
