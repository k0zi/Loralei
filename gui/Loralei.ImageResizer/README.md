# Image Resizer

The **Resize** page of the loralei GUI. It began as a standalone Avalonia app and was imported
here as a library: `Loralei.Gui` supplies the application, the window and the theme, and hosts
`MainView` as one of its pages (`Loralei.Gui/ViewModels/ResizeViewModel.cs`).

It is for turning a folder of images into exact **A × A** squares,
where A is 512, 768, 1024, 1280 or 2048. Built for preparing image sets where every file has
to come out the same square size.

Originals are never modified. Results are written to a `saved_resized` folder inside the folder
you chose.

## What it does

Pick a folder and it loads every image directly inside it (sub-folders are ignored), showing a
thumbnail, file name, dimensions and file size for each one. Each image then gets a square region
picked in one of three ways:

| Mode       | What it does                                                                   |
|------------|--------------------------------------------------------------------------------|
| **Crop**   | Largest centred square inside the image; the long edge is trimmed away.         |
| **Pad**    | The whole image scaled to fit, with the remaining bands filled in.              |
| **Manual** | A square you position yourself in the crop editor.                              |

The pad fill is your choice of a blurred copy of the image (the default), white, black, or
transparent. Images smaller than the target are upscaled, with a badge on the card warning you.

Output keeps the source format — JPEG stays JPEG at your chosen quality, PNG stays PNG, WebP stays
WebP. BMP and GIF become PNG because Skia cannot write them, and a transparent fill forces PNG.

### Theme

The views are plain Avalonia (Fluent) dressed by
[KD.Avalonia.Rice](https://github.com/k0zi/KD.Avalonia.Rice). The light/dark switch, the colour
themes and the settings and about dialogs are the host's; nothing here owns a theme.

The crop editor's canvas deliberately stays dark in both themes, the way photo editors do, so an
image's colours are always judged against the same surround.

### Settings

Output size, default mode, pad fill, quality and the last folder you opened are
remembered in `~/.config/loralei-gui/image-resizer.json`, beside the GUI's own `settings.json`
(and moved with it by `LORALEI_GUI_SETTINGS`). They are written when a folder loads, when you
leave the page and when the window closes. The file is plain,
indented JSON and safe to edit; anything missing, corrupt or out of range falls back to the default
rather than stopping the app from starting.

### The crop editor

Drag inside the square to move it, drag a corner to resize it (it always stays square), scroll to
zoom. The square may hang off the edge of the image; whatever hangs off is filled with the pad fill,
so one editor covers both "crop in tighter" and "pad and reposition".

## Running it

```sh
dotnet run --project Loralei.Gui -- --page Resize    # from gui/
dotnet test Loralei.ImageResizer.Tests
```

## Layout

```
Loralei.ImageResizer/
  Models/      plain types: SquareSelection, ResizeSettings, AppSettings, the enums
  Services/    ResizeGeometry (pure maths), OutputFormatPolicy (pure), decoding, rendering, saving,
               AppSettingsStore
  ViewModels/  MainViewModel, ImageItemViewModel, CropEditorViewModel
  Views/       MainView (the page content), CropEditorWindow, the folder picker
  Controls/    CropOverlay, the draggable square
Loralei.ImageResizer.Tests/
```

`MainView` loads the remembered folder when it is attached and writes the settings when it is
detached - what the old main window did on open and close. `DialogService` looks its owner window
up on each call, because the host builds its view models before its window exists.

Paint your own surfaces with the `Rice*Brush` resources (`RiceSurfaceBrush`, `RiceBorderBrush`,
`RiceForegroundMutedBrush`, ...) through `DynamicResource`, so they follow both the theme and the
light/dark switch. `ThemeTests` renders the view in both variants and fails if the frames
come out identical, which is the guard against a pinned brush creeping in.

The tests bring their own minimal `Application` (`TestApp`, just `RiceStyles`) and share it for the whole run (`AvaloniaTestIsolationLevel.PerAssembly`):
Rice's `ThemeManager` is a process-wide singleton whose brush dictionary can only be attached once.

Crop, pad and manual all reduce to "map this `SquareSelection` onto the whole A × A canvas", so
there is a single render path. The geometry lives in `ResizeGeometry` as pure functions over plain
numbers, which is what makes it testable without a window or a GPU.
