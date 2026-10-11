using System.Text.RegularExpressions;

namespace Loralei.Core;

/// <summary>Removes terminal control sequences from the CLI's output.</summary>
public static partial class AnsiStripper
{
    // CSI (colour, cursor, erase-line), plus OSC ... BEL/ST used for hyperlinks.
    [GeneratedRegex(@"\x1B\[[0-9;?:]*[ -/]*[@-~]|\x1B\][^\x07\x1B]*(?:\x07|\x1B\\)|\x1B[@-Z\\-_]")]
    private static partial Regex ControlSequences();

    public static string Strip(string text) =>
        string.IsNullOrEmpty(text) ? text : ControlSequences().Replace(text, "");
}
