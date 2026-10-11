using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.Json.Nodes;

namespace Loralei.Core.Config;

/// <summary>
/// Maps between <see cref="RunConfig"/> properties and their snake_case keys, and coerces the
/// loosely typed values that arrive from YAML, the environment and command-line flags.
/// <para>
/// Unknown keys are an error, not a warning: a typo in <c>loralei.yaml</c> would otherwise
/// silently train with the default it was meant to replace.
/// </para>
/// </summary>
internal static class ConfigBinder
{
    private static readonly Dictionary<string, PropertyInfo> Properties = typeof(RunConfig)
        .GetProperties(BindingFlags.Public | BindingFlags.Instance)
        .Where(p => p.CanWrite)
        .ToDictionary(p => SnakeCase(p.Name), StringComparer.Ordinal);

    public static IReadOnlyCollection<string> Keys => Properties.Keys;

    public static string SnakeCase(string pascal)
    {
        var sb = new StringBuilder();
        for (var i = 0; i < pascal.Length; i++)
        {
            var ch = pascal[i];
            // "SwapH2dOnly" -> "swap_h2d_only": a digit stays attached to the letter before it.
            if (char.IsUpper(ch) && i > 0) sb.Append('_');
            sb.Append(char.ToLowerInvariant(ch));
        }
        return sb.ToString();
    }

    public static void Apply(RunConfig cfg, IReadOnlyDictionary<string, object?> values)
    {
        foreach (var (key, value) in values)
        {
            if (!Properties.TryGetValue(key, out var property))
                throw new ConfigException($"unknown config key '{key}'");
            property.SetValue(cfg, Coerce(key, property.PropertyType, value));
        }
    }

    private static object? Coerce(string key, Type type, object? value)
    {
        if (type == typeof(List<string>)) return ToList(value);

        var nullable = Nullable.GetUnderlyingType(type);
        var target = nullable ?? type;
        var optional = nullable is not null ||
                       (type == typeof(string) && key is "model_path" or "trigger_word" or "target_regex" or "exclude_regex");

        if (value is null)
            return optional ? null : throw new ConfigException($"{key} must not be null");

        var text = value switch
        {
            string s => s,
            bool b => b ? "true" : "false",
            IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
            _ => value.ToString() ?? "",
        };

        if (target == typeof(string)) return key == "model_path" ? PathText.ExpandUser(text) : text;

        if (target == typeof(bool))
            return text.ToLowerInvariant() switch
            {
                "true" or "yes" or "on" or "1" => true,
                "false" or "no" or "off" or "0" => false,
                _ => throw new ConfigException($"{key} must be true or false, got '{text}'"),
            };

        if (target == typeof(int))
        {
            if (int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var i)) return i;
            // YAML writes 2000.0 as readily as 2000; accept it when nothing is lost.
            if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var whole) &&
                whole == Math.Floor(whole) && Math.Abs(whole) < int.MaxValue)
                return (int)whole;
            throw new ConfigException($"{key} must be an integer, got '{text}'");
        }

        if (target == typeof(double))
            return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var d)
                ? d
                : throw new ConfigException($"{key} must be a number, got '{text}'");

        throw new ConfigException($"{key} has an unsupported type {target.Name}");
    }

    private static List<string> ToList(object? value) => value switch
    {
        null => [],
        string s => [s],
        IEnumerable<string> many => many.ToList(),
        _ => [value.ToString() ?? ""],
    };

    public static JsonObject ToJson(RunConfig cfg)
    {
        var json = new JsonObject();
        foreach (var (key, property) in Properties)
        {
            json[key] = property.GetValue(cfg) switch
            {
                null => null,
                string s => JsonValue.Create(s),
                bool b => JsonValue.Create(b),
                int i => JsonValue.Create(i),
                double d => JsonValue.Create(d),
                List<string> list => new JsonArray(list.Select(s => (JsonNode?)JsonValue.Create(s)).ToArray()),
                var other => throw new InvalidOperationException($"{key}: {other.GetType().Name}"),
            };
        }
        return json;
    }
}

/// <summary>Path spellings users type: <c>~</c>, and several roots joined by the path separator.</summary>
public static class PathText
{
    public static string Home => Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    public static string ExpandUser(string path)
    {
        if (path == "~") return Home;
        return path.StartsWith("~/", StringComparison.Ordinal) ? Path.Combine(Home, path[2..]) : path;
    }

    /// <summary>`repo @ /home/me/models/hub` -> `repo @ ~/models/hub`, so listings stay readable.</summary>
    public static string Shorten(string label) => label.Replace(Home, "~");

    /// <summary>
    /// True for a regular file, or a link that leads to one. .NET's own <c>File.Exists</c>
    /// answers true for a dangling link, and the Hugging Face cache is made of links - one
    /// whose blob was pruned must count as missing.
    /// </summary>
    public static bool IsFile(FileSystemInfo entry)
    {
        if (entry is not FileInfo) return false;
        if (entry.LinkTarget is null) return entry.Exists;
        try { return entry.ResolveLinkTarget(returnFinalTarget: true) is FileInfo { Exists: true, LinkTarget: null }; }
        catch (IOException) { return false; } // a link cycle
    }

    /// <summary>Accepts one path, several joined by the path separator, or a list of either.</summary>
    public static List<string> AsRoots(IEnumerable<string>? values)
    {
        var roots = new List<string>();
        foreach (var value in values ?? [])
        foreach (var part in value.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
            roots.Add(ExpandUser(part));
        return roots;
    }
}
