using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Loralei.Core;

namespace Loralei.Gui.ViewModels;

/// <summary>
/// The shared output pane. Every command's stdout and stderr lands here, so the exact
/// thing loralei printed is always one glance away - the GUI is a front end to the CLI,
/// not a replacement for seeing what it did.
/// </summary>
public sealed partial class LogViewModel : ObservableObject
{
    private const int MaxLines = 4000;

    public ObservableCollection<string> Lines { get; } = [];

    [ObservableProperty] private bool _autoScroll = true;

    private bool _lastWasTransient;

    /// <summary>
    /// Appends a line, or overwrites the previous one when the child is redrawing a
    /// progress bar in place. Without this a 2000-step run would add 2000 near-identical
    /// entries and bury everything else.
    /// </summary>
    public void Append(ConsoleLine line)
    {
        if (_lastWasTransient && Lines.Count > 0) Lines[^1] = line.Text;
        else Lines.Add(line.Text);

        _lastWasTransient = line.Transient;
        while (Lines.Count > MaxLines) Lines.RemoveAt(0);
    }

    public void AppendNote(string text)
    {
        Lines.Add(text);
        _lastWasTransient = false;
        while (Lines.Count > MaxLines) Lines.RemoveAt(0);
    }

    /// <summary>Marks the start of a command with the exact argv, so it can be reproduced.</summary>
    public void AppendCommand(string preview)
    {
        if (Lines.Count > 0) AppendNote("");
        AppendNote($"$ {preview}");
    }

    [RelayCommand]
    private void Clear()
    {
        Lines.Clear();
        _lastWasTransient = false;
    }

    public string AllText => string.Join('\n', Lines);
}
