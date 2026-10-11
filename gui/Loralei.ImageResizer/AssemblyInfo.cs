using System.Runtime.CompilerServices;

// The crop overlay exposes its view transform to the tests so they can aim at its corner handles.
[assembly: InternalsVisibleTo("Loralei.ImageResizer.Tests")]
