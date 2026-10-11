using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Templates;
using Avalonia.Data;
using Avalonia.Layout;
using Loralei.Gui.ViewModels;

namespace Loralei.Gui;

/// <summary>
/// Maps <c>…ViewModels.XViewModel</c> to <c>…Views.XView</c> by name, and wraps the result in
/// the chrome every page shares: heading, blurb, and the setup-problem strip. It also tells
/// the page when its view goes away, so the page can save what it holds.
/// <para>
/// Putting the chrome in the locator keeps the page views free of it.
/// </para>
/// </summary>
public class ViewLocator : IDataTemplate
{
    public Control Build(object? data)
    {
        if (data is not PageViewModel page) return new TextBlock { Text = "no page" };

        var name = page.GetType().FullName!
            .Replace("ViewModels", "Views", StringComparison.Ordinal)
            .Replace("ViewModel", "View", StringComparison.Ordinal);

        var type = Type.GetType(name);
        Control view = type is not null
            ? (Control)Activator.CreateInstance(type)!
            : new TextBlock { Text = $"view not found: {name}" };

        var root = Chrome(page, view);
        // The page is captured rather than read back from DataContext, which is already gone
        // by the time the view is detached.
        root.DetachedFromVisualTree += (_, _) => page.Persist();
        return root;
    }

    private static Control Chrome(PageViewModel page, Control view)
    {
        var heading = new TextBlock { Text = page.Title };
        heading.Classes.Add("h1");

        var blurb = new TextBlock { Text = page.Blurb, Margin = new Avalonia.Thickness(0, 0, 0, 10) };
        blurb.Classes.Add("subtle");

        var problemText = new TextBlock { TextWrapping = Avalonia.Media.TextWrapping.Wrap };
        problemText.Bind(TextBlock.TextProperty,
            new Binding(nameof(MainWindowViewModel.SetupProblem)) { Source = page.Shell });

        var problem = new Border
        {
            Margin = new Avalonia.Thickness(0, 0, 0, 12),
            Child = problemText,
        };
        problem.Classes.Add("panel");
        problem.Classes.Add("problem");
        problem.Bind(Visual.IsVisibleProperty, new Binding(nameof(MainWindowViewModel.SetupProblem))
        {
            Source = page.Shell,
            Converter = Avalonia.Data.Converters.StringConverters.IsNotNullOrEmpty,
        });

        var header = new StackPanel { Spacing = 4 };
        header.Children.Add(heading);
        header.Children.Add(blurb);
        header.Children.Add(problem);

        if (page.FillsPage)
        {
            // No scroller: the view gets a finite height and scrolls whatever it needs to itself.
            var dock = new DockPanel { Margin = new Avalonia.Thickness(18) };
            DockPanel.SetDock(header, Dock.Top);
            dock.Children.Add(header);
            dock.Children.Add(view);
            return dock;
        }

        var stack = new StackPanel { Spacing = 4 };
        stack.Children.Add(header);
        stack.Children.Add(view);

        return new ScrollViewer
        {
            Padding = new Avalonia.Thickness(18),
            HorizontalAlignment = HorizontalAlignment.Stretch,
            Content = stack,
        };
    }

    public bool Match(object? data) => data is PageViewModel;
}
