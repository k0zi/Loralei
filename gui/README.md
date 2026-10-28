# loralei - the .NET side

Everything in loralei that is not torch lives here: the `loralei` command itself
(`Loralei.Cli` over `Loralei.Core`) and a desktop front end for it (`Loralei.Gui`, Avalonia on
.NET 10). The GUI drives the whole workflow — doctor → models → data → cache → train →
sample → export → runs — with live progress, a loss chart and a sample gallery.

```bash
cd gui
dotnet run --project Loralei.Cli -- models ls      # the CLI, straight from source
```

```bash
cd gui
dotnet run --project Loralei.Gui            # or: --project Loralei.Gui -- --page Train
```

## How it works

There are three layers, and each one only talks to the next:

```
Loralei.Gui  --argv-->  loralei (Loralei.Cli + Loralei.Core)  --one JSON job-->  python -m loralei
  the pages               config, models, dataset, planning                        torch
```

The GUI does not reimplement anything. It spawns the `loralei` CLI - built beside it, since
`Loralei.Cli` is a project reference - as a child process, and reads the files a run writes.
The CLI stays the source of truth, and everything the GUI does is visible: each page shows
the exact `argv` before it runs it, and the shared output pane at the bottom carries the
child's real stdout and stderr on every page.

The CLI in turn does everything it can itself and starts the Python worker only for the
part that needs torch, handing it a finished job (`Loralei.Core/Worker/`). The root README
describes that seam.

| What the GUI needs | Where it comes from |
|---|---|
| Live step, loss, VRAM | `runs/<name>/logs/train.jsonl` |
| Run state, peak VRAM, final path | `runs/<name>/manifest.json` |
| Settings a past run used | `runs/<name>/config.resolved.yaml` |
| ROCm health | `loralei doctor --json` |
| Model inventory, bucket preview, reclaimable caches | `models ls --json` / `data check --json` / `models gc --json` |
| Validation renders | `runs/<name>/samples/**.png` |

### The workspace and `loralei.yaml`

The `--json` payloads are serialized from types in `Loralei.Core/Dto` and read back into the
same types, so the CLI and the GUI cannot drift apart the way a printed table and a scraper
for it can.

`loralei train` has CLI flags for only about a third of `RunConfig`. There is no `--config`
flag, and the CLI reads `loralei.yaml` **from its working directory**. So the GUI:

1. treats one directory as the *workspace* — the one holding `loralei.yaml` (or, on a fresh
   checkout, the `loralei.example.yaml` it is copied from), found by walking up from the
   executable, and overridable in settings;
2. writes the flagless settings into `<workspace>/loralei.yaml`;
3. launches every child process with that directory as its working directory.

Anything with a real flag is still passed as a flag, so it is visible in the command preview.

The Train form is **seeded from the existing `loralei.yaml`** when the page first opens. This is
load-bearing, not a nicety: an empty field means "inherit from the preset", and saving writes
a cleared field as a removal. Without seeding, the first save would delete settings you had
put in that file by hand.

### Recommended settings and epochs

*LoRA type* on the Train page is `custom` by default, which recommends nothing. Choosing
`style` or `character` fills in rank, learning rate, run length, warmup, checkpoint interval,
batch size, repeats and caption dropout from the model and the number of images in the dataset
folder (`Recommender` in `Loralei.Core`). The values are rules of thumb, not measurements.
A field only follows a new recommendation while it is empty or still holds the last recommended
value; *Reset to recommended* overwrites the hand-edited ones too. Memory settings and the
timestep schedule are left to the preset and profile.

*Run length* can be given in steps or in epochs. Loralei itself only takes `--steps`, so epochs
are converted in the form: `ceil(images × repeats ÷ batch size)` steps per epoch, which is how
the trainer counts them for a single bucket (with several buckets and a batch size above 1 it
is a lower bound). The image count is by file suffix, so it includes files the dataset scan
would later reject as duplicates or unreadable.

### Interrupting a run

Interrupt sends **SIGINT**, not a kill. The CLI catches it and passes it on to its worker,
where `train/loop.py` stamps `manifest.json` with `state: "failed"` and releases the block
swapper; the CLI stays until the worker is done. A `Process.Kill()` would skip all of that.
.NET has no API for sending a signal, so `Native.Signal` P/Invokes `kill(2)`.

Two things worth knowing:

- **Interrupting loses the adapter** unless a checkpoint exists — the final save happens
  before the handler that stamps the manifest. The Train page warns when *Save every* is 0.
- Shutdown can take a while if a validation render is in flight; the trainer waits up to
  600 s per sampler child.

### Theming

The look is plain Avalonia (Fluent) dressed by
[KD.Avalonia.Rice](https://github.com/k0zi/KD.Avalonia.Rice) 26.10.6: `RiceWindow` draws the
frameless chrome and title bar, `RiceSideMenu` is the page navigation, and `Border.panel` is
a card painted from the active Rice theme. The title bar carries Rice's own buttons: the
light/dark toggle, settings and about. Settings opens `SettingsWindow`, which holds a
**System / Light / Dark** start-up picker and Rice's `ThemeSelector` (Ubuntu, Fedora, Nord,
Dracula, …); about opens Rice's built-in dialog, fed from the `RiceAppInfo` in `Program.cs`.

Three things are worth knowing before changing any colour here:

- **Light/dark has a remembered half and a session half.** The start-up variant is ours:
  `ThemeService` (`Loralei.Gui/Theming/`) writes it to `settings.json` and sets
  `Application.RequestedThemeVariant`, where System is simply `ThemeVariant.Default`. The
  title-bar toggle then flips it for the session only and is not saved. The colour theme is
  Rice's: `ThemeManager` applies it and saves it to `~/.config/loralei-gui/rice.json`.
- **Use `DynamicResource`, never `StaticResource`, for anything theme-dependent.**
  `StaticResource` resolves once, so a live switch would leave the old colours on screen. Our
  own palette (`OkBrush`, `WarnBrush`, `FailBrush`, `MutedBrush`, `LogPaneBrush`,
  `ProblemBrush` and the four `Chart*Brush` keys) lives in a `ThemeDictionaries` block in
  `Styles.axaml` with a value per variant; Rice's keys (`RiceSurfaceBrush`,
  `RiceBorderBrush`, `RiceAccentBrush`, …) are rebuilt by `ThemeManager` per colour theme.
- **Rice's `ThemeManager` is a process-wide singleton** whose brush dictionary can be
  parented by one `RiceStyles` only. That is why the GUI tests run with
  `AvaloniaTestIsolationLevel.PerAssembly` (`TestApp.cs`): a fresh `App` per test throws
  "The ResourceDictionary already has a parent".

`LossChart` paints itself, so it is the one control that cannot inherit any of this: it
resolves its brushes against `ActualThemeVariant` — trying its own resource chain, then the
Application's, because a control does not always reach a `Styles.Resources` dictionary — and
repaints on `ActualThemeVariantChanged`. `ThemeTests.TheChartResolvesItsOwnBrushesPerVariant`
guards exactly that, and it does fail if the Application-level lookup is removed.

`ViewLocator` does double duty: it maps view model to view *and* wraps the result in the
shared page chrome (heading, blurb, setup-problem strip), which keeps the page views
free of it. It also calls the page's `Persist()` when the view goes away, which is how Caption
saves its fields on a page switch or a window close. A page that sets `FillsPage` - Resize, whose image grid scrolls itself - gets the
space below the chrome instead of the page scroller. Because it matches every page view model application-wide, the side menu sets
each entry's `Content` and `Icon` through a `RiceSideMenuItem` style rather than leaving the
item to default content — `ShellTests` guards that.

### Icons

[Material.Icons.Avalonia](https://github.com/SKProCH/Material.Icons) 3.0.2, registered by
`<rice:RiceStyles />` in `App.axaml`. Each page declares its own glyph
(`PageViewModel.Icon`, a `MaterialIconKind`) which the side menu renders, and the Doctor
table pairs its status colour with a shape — a tick, a warning, a cross — so a failed check
does not depend on hue alone to be noticed.

`IconTests` keeps two things honest: that the status-to-glyph mapping is what it claims, and
that no two pages share an icon.

### Deliberate choices

- **Doctor does not run at launch.** Even the quick checks allocate VRAM for a bf16 matmul and
  an 8-bit Adam step, which could push a training run — possibly started from a terminal —
  over the edge of a 16 GiB card.
- **One command at a time.** There is a single box behind all of this; two GPU commands would
  just fight over VRAM.
- **stdin is closed** on every child, so a confirmation the GUI did not anticipate is
  answered "no" at once instead of hanging. `runs rm` always passes `--yes` after the GUI's
  own confirmation, and `models gc` is only ever run in its listing form.
- **`FORCE_COLOR` is set per command, never globally.** It makes the CLI emit colour and
  live progress frames into a pipe, which the `--json` commands must never get.
- **No chart, YAML, argument-parsing or imaging library in the core.** The loss chart is
  ~150 lines of `StreamGeometry`; the YAML files involved are flat scalar maps; the CLI has
  a dozen flat commands; and the dataset scan needs only image *headers*, which
  `ImageHeader` reads itself. `Loralei.Core` and `Loralei.Cli` have no package references at
  all, which keeps the whole thing restorable from the local NuGet cache - that suits a
  machine that is offline by design.

## Privacy

With one exception, the GUI itself opens no network connection: it spawns the CLI and reads files.
(Two of the CLI commands it can spawn download: **Pull** on the Models page and **Install
worker** on the Doctor page, each only when its button is pressed.)
The exception is the **Caption** page, which is there to send images somewhere: pressing
Caption posts every image in the chosen folder, with the prompts, to the endpoint in its Base
URL field. That is a local model if the URL is local, and a third party if it is not. Nothing
is sent until Caption is pressed; Preview sends nothing.

The only things the GUI ever writes are
`~/.config/loralei-gui/settings.json` (paths, light/dark preference, window size),
`~/.config/loralei-gui/rice.json` (the colour theme, written by Rice),
`~/.config/loralei-gui/image-resizer.json` (the Resize page's options and last folder),
`~/.config/loralei-gui/dataset-maker.json` (the Caption page's fields, API key included if you
typed one; mode 0600), the workspace `loralei.yaml`, the resized copies the Resize page saves
into a `saved_resized` folder inside the folder you picked, and the `.txt` captions the Caption
page writes beside the images. Everything else under `runs/` is written by loralei itself. Child processes get
only `FORCE_COLOR`/`NO_COLOR` added to their environment; the CLI then adds the offline
switches (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `DO_NOT_TRACK`, …) for its Python worker
unless you have set them yourself.

Verified with `strace -f -e trace=connect`, not just by reading the code - but against
loralei 0.1, when the CLI was still Python. The GUI's own behaviour has not changed since;
the trace of a training run has not been repeated with the C# CLI in the middle:

| Traced | Result |
|---|---|
| App start, Models page, side menu | 8 `connect()`, all `AF_UNIX` (syslog, session DBus, at-spi, LTTng) |
| A real training run driven through the view models | 48 `AF_UNIX` + 1 loopback; **no non-loopback IP** |
| `dotnet build -t:Rebuild` | **no IP connection** (see below) |

`ss` also shows the running app with no listening socket and no established connection, and
Avalonia's TCP-capable assemblies (`Avalonia.Remote.Protocol`, `Avalonia.DesignerSupport` —
the XAML previewer transport) are not even mapped into the process.

**One thing was found and fixed.** Avalonia pulls in `Avalonia.BuildServices`, a *build-time*
telemetry collector that posted anonymised build metadata to Avalonia's servers during
compilation. It never sees source, file paths, datasets or anything from a running app — but
a project that is otherwise offline by design should not phone home when you build it, so
`Directory.Build.targets` overrides its `AvaloniaStats` target. Re-check after any Avalonia
upgrade, since a renamed target would silently restore it:

```bash
strace -f -qq -e trace=connect -o /tmp/b.trace dotnet build Loralei.Gui -t:Rebuild
grep AF_INET /tmp/b.trace        # expect nothing
```

The other way out, as on the CLI, is `loralei models pull` — the Models page downloads
base weights through the CLI, which fetches them from the Hugging Face Hub itself
(`Loralei.Core/Models/ModelDownloader.cs`). Nothing else reaches the network.

## Layout

```
Loralei.Core/                no package references at all. Two things live here:
                               what loralei *is* - Config/ (layers, presets), Models/ (registry,
                               lookup, download), Data/ (scan, buckets), Caching/, Lora/
                               (safetensors, ComfyUI), Diagnostics/ (doctor), Worker/ (the
                               Python worker and its jobs)
                               and what the GUI needs to drive it - LoraleiClient, RunStore,
                               the event-log tailer
Loralei.Cli/                 the `loralei` command: arguments in, text or JSON out
Loralei.Gui/                 the Avalonia app — Views/, ViewModels/, Controls/, Theming/
Loralei.Core.Tests/          unit tests, plus integration tests against the real CLI and worker
Loralei.Gui.Tests/           headless UI tests, including a real end-to-end training run
Loralei.ImageResizer/        the Resize page: squares up a folder of images; see its own README
Loralei.ImageResizer.Tests/  its tests - geometry, the save pipeline, the crop editor
Loralei.DatasetMaker/        the Caption page's engine: captions images through a vision endpoint
Loralei.DatasetMaker.Tests/  its tests, against a loopback stand-in for the endpoint
TestSupport/                 shared test helpers
```

## Tests

```bash
dotnet test                                          # everything
dotnet test --filter "FullyQualifiedName!~Live&FullyQualifiedName!~EndToEnd&FullyQualifiedName!~Interrupting"
```

The CLI is built beside the tests, so the ones that run it always can. The ones that need
the GPU skip themselves when the base weights are not on the machine. Those train for real:
`TrainsEndToEndAndReportsLiveProgress` runs 12 steps of Qwen-Image at 256px through the
view model and asserts progress arrives *while the child is still running*, and
`InterruptingStopsTheRunAndSaysWhatWasLost` starts a long run and interrupts it.

Each test builds a throwaway workspace - its own directory, its own `loralei.yaml` (carrying
over only `models_dir`), its own settings file via `LORALEI_GUI_SETTINGS`. Nothing under test
can reach your project config or your `runs/`, even if a run is killed before its cleanup
gets to run. A lock file serialises the tests that spawn loralei, because there is one GPU.

One thing these tests deliberately do not cover: the `DispatcherTimer` that drives the live
readout in the running app never ticks under the headless harness, so the tests call
`TrainViewModel.PollLiveState()` - the same drain the timer calls - directly.

## Requires

.NET 10 SDK. For anything that touches the GPU, also the Python worker: the project
`.venv` with torch and `pip install -e .` (see the root README), which the CLI finds by
walking up from the workspace, or wherever `LORALEI_PYTHON` points.

Packages: Avalonia 12.1.3, KD.Avalonia.Rice 26.10.6 (which asks for exactly that Avalonia),
CommunityToolkit.Mvvm 8.4.2, Material.Icons.Avalonia 3.0.2 (asks for Avalonia 12.0.0 — the
explicit references unify it upward; a restore that reports NU1605 means that stopped
working).

All of them are in the local NuGet cache after one network restore, so the project remains
restorable offline.
