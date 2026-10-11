using System.Globalization;

namespace Loralei.Cli;

/// <summary>A mistake on the command line: reported with the command's usage, exit code 2.</summary>
public sealed class UsageException(string message) : Exception(message);

/// <summary>
/// The options one command accepts. Hand-rolled on purpose: the commands are few and flat,
/// and this keeps the dependency list short enough to restore from the local NuGet cache.
/// </summary>
public sealed class OptionSet
{
    private readonly Dictionary<string, (string Name, bool? Fixed)> _names = new(StringComparer.Ordinal);
    private readonly HashSet<string> _flags = new(StringComparer.Ordinal);

    /// <summary>An option that takes a value: <c>--rank 32</c> or <c>--rank=32</c>.</summary>
    public OptionSet Value(string name, params string[] aliases)
    {
        foreach (var spelling in aliases.Prepend(name)) _names[spelling] = (name, null);
        return this;
    }

    /// <summary>A switch. <paramref name="negation"/> names the spelling that turns it off.</summary>
    public OptionSet Flag(string name, string? negation = null, params string[] aliases)
    {
        _flags.Add(name);
        foreach (var spelling in aliases.Prepend(name)) _names[spelling] = (name, true);
        if (negation is not null) _names[negation] = (name, false);
        return this;
    }

    public Args Parse(IReadOnlyList<string> argv)
    {
        var args = new Args();
        for (var i = 0; i < argv.Count; i++)
        {
            var token = argv[i];
            if (token == "--")
            {
                args.Positional.AddRange(argv.Skip(i + 1));
                break;
            }
            // A lone "-" and negative numbers are values, not options.
            if (!token.StartsWith('-') || token.Length == 1 || char.IsDigit(token[1]))
            {
                args.Positional.Add(token);
                continue;
            }

            string? inline = null;
            var equals = token.IndexOf('=');
            if (token.StartsWith("--", StringComparison.Ordinal) && equals > 0)
            {
                inline = token[(equals + 1)..];
                token = token[..equals];
            }

            if (token is "--help" or "-h") { args.Help = true; continue; }
            if (!_names.TryGetValue(token, out var option)) throw new UsageException($"unknown option {token}");

            if (_flags.Contains(option.Name))
            {
                if (inline is not null) throw new UsageException($"{token} does not take a value");
                args.Set(option.Name, option.Fixed == true ? "true" : "false");
                continue;
            }

            if (inline is null)
            {
                if (++i >= argv.Count) throw new UsageException($"{token} needs a value");
                inline = argv[i];
            }
            args.Set(option.Name, inline);
        }
        return args;
    }
}

public sealed class Args
{
    private readonly Dictionary<string, List<string>> _values = new(StringComparer.Ordinal);

    public List<string> Positional { get; } = [];
    public bool Help { get; set; }

    internal void Set(string name, string value)
    {
        if (!_values.TryGetValue(name, out var list)) _values[name] = list = [];
        list.Add(value);
    }

    public string? Get(string name) => _values.TryGetValue(name, out var list) ? list[^1] : null;

    public IReadOnlyList<string> All(string name) => _values.TryGetValue(name, out var list) ? list : [];

    /// <summary>Null when the switch was not given either way, so a config layer can still decide.</summary>
    public bool? Flag(string name) => Get(name) is { } value ? value == "true" : null;

    public bool On(string name, bool fallback = false) => Flag(name) ?? fallback;

    public int? Int(string name)
    {
        if (Get(name) is not { } text) return null;
        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
            ? value
            : throw new UsageException($"{name} must be an integer, got '{text}'");
    }

    public double? Double(string name)
    {
        if (Get(name) is not { } text) return null;
        return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var value)
            ? value
            : throw new UsageException($"{name} must be a number, got '{text}'");
    }

    /// <summary>The single positional argument a command requires.</summary>
    public string Required(int index, string what) =>
        index < Positional.Count ? Positional[index] : throw new UsageException($"missing {what}");

    public void NoMoreThan(int count)
    {
        if (Positional.Count > count) throw new UsageException($"unexpected argument '{Positional[count]}'");
    }
}
