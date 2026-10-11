namespace Loralei.ImageResizer.ViewModels;

/// <summary>An enum value paired with the wording shown in a combo box.</summary>
public sealed record EnumOption<T>(T Value, string Label)
{
    public override string ToString() => Label;
}
