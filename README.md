<p align="center">
  <img src="src/assets/loralei.png" alt="Loralei" width="320">
</p>

# Loralei

**A LoRA trainer for AMD GPUs.** Train a LoRA from a folder of images and captions on
ROCm, with no CUDA anywhere in the stack.

```bash
loralei train ./my-dataset --name my-style --model krea-2-raw
```

The dataset is just paired files - `a.jpg` next to `a.txt`, `b.png` next to `b.txt`.
Sub-directories are searched too.

## What is in the box

- **A command line**, `loralei`, that covers the whole path: check the ROCm stack, find or
  download the base weights, check the dataset, cache, train, sample, export, manage runs.
- **A desktop app** (Avalonia) with a page for each of those steps, live loss and VRAM and
  a sample gallery while a run is going - plus two pages for preparing a dataset: **Resize**
  squares up a folder of images, **Caption** writes the `.txt` captions through an
  OpenAI-compatible vision endpoint.
- **Three base models**: Qwen-Image, Krea 2 Raw and LTX 2.3.
- **Block swap and fp8 storage** of the frozen base, to fit large models into less VRAM.
- **Offline by default.** Only `loralei models pull` opens a network connection, and in
  the app the Caption page, which posts images to the endpoint you point it at.
- **Two output layouts**: diffusers and ComfyUI, each loaded back into the real
  transformer to prove it attaches.

## Requirements

- Linux, and an AMD GPU that ROCm supports. Developed and measured on a Radeon RX 9060 XT
  (gfx1200, 16 GiB) with ROCm 7.1.
- Integrated GPUs are detected and skipped: they share system memory and run out of it at
  once with models this size.
- There is no CUDA, Vulkan or CPU training path. `LORALEI_DEVICE=cpu` exists only to
  smoke-test the plumbing on tiny models (see Testing).
- The .NET 10 SDK and Python 3.12.
- Disk for the base weights: 33 to 95 GiB per model.

## Base models

| `--model` | what | size | notes |
|---|---|---|---|
| `qwen-image` (default) | Qwen-Image 2512, 20B | 57 GiB | |
| `krea-2-raw` | Krea 2 Raw, 12.8B | 33 GiB | gated: accept the licence on the Hub and set `HF_TOKEN` |
| `ltx-2.3` | LTX 2.3 dev, 19B | 95 GiB | 45 GiB of that is its Gemma text encoder; 32 px grid |

**How far each one is proven.** For every model, the worker's training code is checked
against the model's own diffusers pipeline: the prompt is encoded to the same tensors, the
latents are normalised the way the pipeline un-normalises them, and the transformer is
called with the same arguments (`tests/test_families.py`). And the whole path - cache,
train, checkpoint, resume, sample, export - has been run for each on tiny random-weight
models of the same architecture. What has **not** happened yet is a full run on real
weights judged by eye. Treat the presets as starting values and look at early samples
before committing to a long run.

**LTX 2.3 is a video model; a dataset here is images.** It is trained on images as
one-frame clips, which it accepts natively. That teaches a subject or a style, not motion.
It also has an audio stream: that is fed noise and only the video side is trained.

---

## How it is put together

loralei is two halves with a narrow seam between them.

| | lives in | does |
|---|---|---|
| **`loralei` CLI** (C#, .NET 10) | `gui/Loralei.Cli`, `gui/Loralei.Core` | everything that does not need torch: the config layers, finding and downloading base models, the dataset scan and bucket plan, cache keys, the VRAM estimate, the non-GPU half of `doctor`, the ComfyUI export, runs |
| **worker** (Python) | `src/loralei` | everything that does: VAE and text-encoder caching, the training loop, block swap, fp8, sampling, the load-back check, the GPU half of `doctor` |

The CLI decides; the worker computes. For a GPU command the CLI resolves every setting and
every path, then starts `python -m loralei -` and hands it one finished JSON job on stdin.
The worker reports back as JSON lines on stdout and never reads `loralei.yaml`, searches for
a model or scans a dataset itself. `models ls`, `data check`, `runs`, `export --no-verify`
and `train --dry-run`'s planning never start Python at all.

## Install

Needs the .NET 10 SDK for the CLI and Python 3.12 for the worker.

```bash
# the worker: torch comes from the ROCm wheel index, pinned to match the installed ROCm
uv venv --python 3.12 .venv
uv pip install --index-url https://download.pytorch.org/whl/rocm7.1 \
    torch==2.13.0+rocm7.1 torchvision==0.28.0+rocm7.1
uv pip install -e .
uv pip install bitsandbytes          # optional: 8-bit Adam

# your settings: loralei.yaml is yours and ignored by git
cp loralei.example.yaml loralei.yaml

# the CLI
dotnet publish gui/Loralei.Cli -c Release -o ~/.local/share/loralei
ln -s ~/.local/share/loralei/loralei ~/.local/bin/loralei
```

The CLI finds the worker's interpreter by itself: `$LORALEI_PYTHON` if set, else the nearest
`.venv` walking up from the working directory (so running from this checkout just works),
else `python3` on `PATH`.

Then check the stack before trusting it with a six-hour run:

```bash
loralei doctor
```

## Quick start

```bash
# 1. is the ROCm stack sane?  (exits non-zero if not)
loralei doctor

# 2. are the base weights on this machine?
loralei models ls
loralei models pull krea-2-raw       # only if something is missing

# 3. does the dataset pair up?  (images <-> captions, bucket preview)
loralei data check ./my-dataset

# 4. train.  latents and text embeddings are cached automatically first.
loralei train ./my-dataset --name my-style --model krea-2-raw

# 5. look at it, then ship it
loralei sample -p "a photo of sks style, a cat on a roof" --lora runs/my-style/my-style.safetensors
loralei export my-style --format both
```

Add `--dry-run` to step 4 to see the resolved config, the VRAM estimate and the ETA
without starting anything.

### Or use the GUI

Every one of those steps is also a page in a desktop app, with live loss, VRAM and a
sample gallery while a run is going:

```bash
cd gui && dotnet run --project Loralei.Gui
```

It drives this same CLI as a child process and shows you the exact command before it runs
it. See [gui/README.md](gui/README.md).

Two more pages help before training starts and have no CLI counterpart:

- **Resize** squares up a folder of images, with a crop editor, and saves the copies into
  a `saved_resized` folder inside it; the originals are never modified. See
  [gui/Loralei.ImageResizer/README.md](gui/Loralei.ImageResizer/README.md).
- **Caption** writes a `.txt` caption next to every image through an OpenAI-compatible
  vision endpoint (LM Studio, Ollama, vLLM, llama.cpp, OpenAI, ...). This is the one page
  that sends your images somewhere: to the Base URL you give it. See
  [gui/Loralei.DatasetMaker/README.md](gui/Loralei.DatasetMaker/README.md).

### The dataset

A directory of paired files, searched recursively:

```
my-dataset/
├── a.jpg   a.txt
├── b.png   b.txt
└── nested/ c.webp  c.txt
```

The `.txt` holds the caption for the image of the same name. `loralei data check` reports
images with no caption, captions with no image, duplicates and unreadable files before
they waste a training run. `--trigger sks` prepends a trigger word to every caption that
does not already contain it.

Images are never squashed to a square: they are assigned to aspect-ratio buckets of
roughly equal latent token count, then scaled to cover and centre-cropped.

---

## Commands

### `loralei doctor [--json] [--fix] [--thorough]`
Validates the ROCm stack: driver access, the torch build, which devices are visible and
their architectures, the SDPA backend (the important one - see ROCm notes), bf16 matmul,
8-bit Adam, allocator config, free disk and host RAM. `--fix` prints a shell snippet with
the environment variables; it never changes anything itself. A check that takes the worker
process down with it - a driver abort inside a matmul - is reported as that check failing,
not as a missing table.

### `loralei models ls` / `pull <model>...` / `gc`
`ls` shows each model, whether it can be trained on or only downloaded, whether its weights
are already on this machine, and how many GiB are missing. `pull` takes one model or
several, downloads only the missing components of each, and refuses to start one if
the filesystem cannot take it (`--dry-run` checks without downloading); an interrupted
download resumes from its `.incomplete` file, and `HF_TOKEN` is sent for gated repos. `gc`
ranks large reclaimable caches by size; it lists by default and only deletes with
`--no-dry-run`, one confirmation per entry. `ls` and `gc` take `--json`.

### `loralei data check <dir> [--resolution N] [--caption-extension .txt] [--json]`
Pairs images with captions and prints the bucket histogram with per-bucket token counts.
Only image headers are read, so this is quick even on a large folder; JPEG, PNG, WebP, BMP
and AVIF are understood.

### `loralei cache <dir> [-m model] [--resolution N] [--vae-device cpu|cuda] [--gpu N] [--force]`
Pre-computes VAE latents and text-encoder embeddings. `train` does this for you; run it
separately when you want the cost paid up front. The cache is keyed by model, VAE, text
encoder and geometry, so it can never go stale silently, and changing resolution does not
throw the text embeddings away.

### `loralei train <dir> --name <n> [flags]`
| flag | |
|---|---|
| `-m, --model` | `qwen-image` (default), `krea-2-raw`, `ltx-2.3` |
| `--profile` | `balanced` (default), `quality`, `speed`, `lowmem` |
| `--rank`, `--lr`, `--steps`, `--batch-size`, `--seed` | the usual knobs |
| `--resolution` | a multiple of 16; of 32 for `ltx-2.3` |
| `--blocks-to-swap N` | frozen blocks parked in host RAM (see Memory) |
| `--fp8-base` | store frozen Linear weights as float8 |
| `--optimizer` | `adamw8bit` (default), `adamw`, `adafactor` |
| `--trigger WORD` | prepend a trigger word to captions |
| `--gpu N` | which card to train on |
| `--save-every N` | checkpoint interval (0 = only at the end) |
| `--resume` | continue from the newest checkpoint |
| `--vae-device` | `cpu` (default) or `cuda` |
| `--dry-run` | resolve, estimate, stop |

### `loralei sample -p "..." [--lora F] [-m model] [--steps] [--cfg] [--width] [--height] [--seed] [--gpu] [--out DIR]`
Renders from the base model, optionally with an adapter. `-p` repeats for several prompts.
When `--lora` is given it also verifies the adapter attached to every module it names.

### `loralei export <run> [--format comfyui|diffusers|both] [--no-verify]`
Writes `<run>.safetensors` (diffusers layout) and `<run>_comfy.safetensors` (ComfyUI
layout, with q/k/v fused into one rank-3r adapter), then loads the result back into the
real transformer and fails if any key is unexpected or any target went unadapted. The
conversion itself is done by the CLI, straight on the safetensors bytes; only the load-back
check starts the worker, and `--no-verify` skips it.

### `loralei runs ls` / `show <name>` / `rm <name> [--yes]`
List, inspect and delete runs.

---

## Where everything lives

**Base models** are read from wherever you keep them. By default that is the standard
Hugging Face cache, but `models_dir` in the config points loralei anywhere:

```yaml
# loralei.yaml
models_dir:
  - ~/Develop/ai-models/unsloth/hub          # downloads land in the first root
  - ~/Develop/ai-toolkit/hf-cache/huggingface/hub
```

A root may be in either layout, detected automatically:

```
<root>/models--<org>--<repo>/snapshots/<rev>/<component>/    Hugging Face cache
<root>/<model key>/<component>/                              plain folders
```

Point at a relocated `HF_HOME` and the `hub` inside it is found too. Equivalent on the
command line: `--models-dir PATH` (repeat a colon-separated list), or the environment
variable `LORALEI_MODELS_DIR`.

Each **component** is resolved separately, so partial copies combine instead of forcing a
re-download: if one root holds only a transformer and another has the text encoder and
VAE, that model resolves as complete. `loralei models ls` prints where each one came from.

For a single model in one specific folder, `model_path` (or `--model-path`) takes priority
over everything; components it does not contain still resolve normally:

```yaml
model_path: ~/models/krea-2-raw      # holds transformer/, vae/, text_encoder/, ...
```

`loralei models pull` writes into the first `models_dir` entry as a browsable
`<root>/<model key>/<component>/` tree, or into the Hugging Face cache when no root is
configured. The free-space guard checks whichever of those it is about to use.

**Latent and text caches** live next to the data, so they are reused across runs and
deleted with the dataset:

```
my-dataset/.loralei_cache/latents/<signature>/  and  /text/<signature>/
```

If the dataset directory is read-only, they fall back to `~/.cache/loralei/datasets/`.

**Runs** are in the project:

```
runs/<name>/
├── <name>.safetensors          final LoRA (diffusers layout)
├── <name>_comfy.safetensors    after `loralei export`
├── config.resolved.yaml        every setting this run actually used
├── manifest.json               model, step, state, peak VRAM, dataset fingerprint
├── logs/train.jsonl            one JSON object per logged step: loss, VRAM, checkpoints
├── checkpoints/                <name>-000500.safetensors + .state.pt (for --resume)
└── samples/                    periodic validation renders
```

`logs/train.jsonl` is written for anything watching a run from outside the terminal - the
GUI tails it for the live loss curve. `log_every` (default 10) sets how often a step is
recorded; `manifest.json` only moves at checkpoint boundaries, so it is the wrong thing to
poll for progress.

**Config** is layered, lowest priority first: built-in defaults, `$LORALEI_MODELS_DIR`, then
`gui/Loralei.Core/Presets/<model>.yaml` (compiled into the CLI), then the `--profile` overlay,
then `~/.config/loralei/config.yaml`, then `./loralei.yaml`, then CLI flags. A key the config
does not know is an error, not a warning - a typo would otherwise train with the default it
was meant to replace. The fully resolved config is written to
`runs/<name>/config.resolved.yaml` **and** embedded in the output safetensors metadata, so a
run can be reproduced from its own artifacts.

Per-model defaults worth knowing:

| | qwen-image | krea-2-raw | ltx-2.3 |
|---|---|---|---|
| rank / lr | 16 / 5e-5 | 32 / 1e-4 | 32 / 1e-4 |
| flow shift | 2.2 | 2.5 | 2.6 |
| blocks to swap | 42 of 60 | 14 of 28 | 36 of 48 |
| fp8 base | on | on | on |

None of these rows is tuned on a real run yet. The flow shifts follow each pipeline's own
schedule shift at 1024x1024; the memory settings are sized from the weights.

---

## Offline

loralei runs fully offline. `loralei models pull` is the only command that opens a network
connection, and only while it downloads - and it is the CLI that does it, talking to the
Hugging Face Hub directly (`HF_ENDPOINT` points it at a mirror). The Python worker never
downloads anything.

The worker is offline by default, not just offline in practice: the CLI starts it with
`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY` and `DO_NOT_TRACK` set
before the interpreter runs, so a string that diffusers or transformers mistake for a
Hugging Face repo id fails on the spot instead of being looked up online. The all-Python
loralei 0.1 was checked with `strace` on `connect()` across `doctor`, `cache`, `train`,
`sample` and `export`: zero IPv4 or IPv6 connections, only local Unix sockets between
DataLoader workers. The worker loads the same libraries under the same switches, but that
trace has not been repeated since the CLI moved to C#.

To allow network access anyway, set `HF_HUB_OFFLINE=0` yourself; loralei never overrides
a variable that is already set.

---

## Memory

Measured on a Radeon RX 9060 XT (gfx1200, 16 GiB) with Z-Image, a 6B model loralei no
longer ships, at 1024x1024, rank 32, batch 1, gradient checkpointing on, latents and text
embeddings pre-cached. The mechanism is model-independent; the absolute numbers are not,
and nothing comparable has been measured for the current models yet:

| configuration | peak VRAM | loss after 6 steps |
|---|---|---|
| no block swap | 14.54 GiB | 0.2440 |
| `--blocks-to-swap 14` (default) | 11.14 GiB | 0.2440 |
| `--blocks-to-swap 24` | 7.77 GiB | 0.2440 |
| `--fp8-base --blocks-to-swap 14` | 7.43 GiB | 0.2395 |

The loss is unchanged by block swapping - that is the point. Swapping moves frozen
blocks to pinned host RAM (~0.36 GiB per Z-Image block) and streams them back on a
prefetch stream; `--fp8-base` additionally stores frozen Linear weights as
`float8_e4m3fn` and dequantizes them inside `forward` (no `torch._scaled_mm`, which
upcasts on gfx1200).

---

## ROCm notes

Four gfx1200-specific problems were found while building this, all measured on this
hardware. They are handled automatically; this section is so the behaviour is not
mistaken for a bug in loralei.

### 1. SDPA silently falls back to a 12x slower kernel

gfx1200 is on AOTriton's *experimental* list, so PyTorch refuses both the flash and
the mem-efficient attention backends unless `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`
is set **before torch is imported** (it is latched into a `static const bool` on the
first SDPA dispatch). Measured at B=1, H=30, S=4608, D=128, bf16, forward+backward:

| | time | peak |
|---|---|---|
| flash | 49.7 ms | 0.40 GiB |
| math fallback | 619 ms | 10.07 GiB |

The CLI puts it in the worker's environment before the interpreter starts
(`Loralei.Core/Worker/RocmEnvironment.cs`), which is as early as "before" gets.
`loralei doctor` fails loudly if the flag did not take.

### 2. The integrated GPU enumerates first

`rocm_agent_enumerator` reports `gfx1150` (the 890M) as HIP device 0 on this machine,
ahead of both discrete cards, and `rocm-smi`'s numbering disagrees with HIP's. Training
that lands on the iGPU OOMs instantly against shared system memory. loralei detects APU
gfx targets and sets `HIP_VISIBLE_DEVICES` accordingly; `loralei doctor` prints what it
chose.

### 3. The VAE is unreliable on the GPU - so it runs on the CPU

This one is nasty because it corrupts data instead of crashing:

* **bf16 encode is simply wrong** - 14-21% relative error against a CPU fp32 reference
  on every image and every repeat, far beyond bf16's ~0.4% per-element precision.
* **The first convolution of each tensor shape returns garbage** - usually all-NaN,
  sometimes finite but off by more than 100%, which no cheap sanity check catches.
  512x512 needs one throwaway pass, 1024x1024 needs two.
* **At 1024-class shapes it mostly does not work at all** - most passes return NaN, and
  some fault the queue outright with `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` or
  `hipErrorIllegalAddress`, preceded by `MIOpen(HIP): Error [EvaluateInvokers] ... Empty
  code object path`. Neither card behaves differently, and none of
  `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0`, `MIOPEN_DEBUG_CONV_WINOGRAD=0`,
  `MIOPEN_DEBUG_CONV_DIRECT=0`, `MIOPEN_FIND_MODE=NORMAL`, `vae.enable_tiling()`,
  forcing the math SDPA backend, or clearing the MIOpen kernel cache avoided it.

So the VAE runs on the CPU by default: ~7.5 s per 1024x1024 image, once per dataset,
and always correct. `--vae-device cuda` re-enables the GPU path, which verifies each new
shape by requiring two passes to agree and falls back to the CPU if they never do.

The sampler decodes on the GPU, but warms up at the exact latent shape it is about to
use - without that, the first image comes out as a coloured band on black. It also
re-runs any decode that returns NaN or a uniformly flat frame, falling back to the CPU
VAE after two attempts, because that still happens occasionally even with the warmup.

### 4. Block swap needs its own eviction

PyTorch's non-reentrant gradient checkpointing aborts a block's recomputation with an
internal exception as soon as it has the saved tensors it needs, so **post-forward hooks
never fire during backward**. A hook-based swapper therefore loads blocks during backward
and never unloads them: measured 34 loads / 33 evictions in the forward, then 29 loads and
*zero* evictions in the backward, with 450 parameters stranded on the GPU and peak VRAM
back where it started. loralei bounds residency explicitly instead, evicting on the next
block's pre-hook.

Relatedly, the prefetch stream needs a real `cuda.Event` and `record_stream`. Without
them training still runs, converges, and produces a LoRA - just against half-copied
weights, which showed up only as a loss of 1.93 where 1.03 was expected.

---

## Testing

```bash
cd gui && dotnet test                    # the CLI, the core and the GUI
.venv/bin/python -m pytest tests -q      # the worker
```

The C# tests cover everything that moved out of Python - config layering, model lookup, the
downloader (against a loopback stand-in for the Hub), the dataset scan and bucket plan, the
safetensors reader and the ComfyUI conversion - plus the worker protocol against a stand-in
interpreter. The ones that train for real skip themselves when the base weights are not on
the machine.

The worker's block swap tests need a ROCm device and skip without one. Everything else
there is CPU-only.

`tests/test_families.py` is the one that matters most: it runs each model family against
its real diffusers pipeline. It needs tiny random-weight models (about 100 MB in all),
fetched once with `.venv/bin/python tests/fetch_tiny_models.py`, and skips without them.

The whole path can also be run on a machine with no usable GPU, against those tiny models:
point `models_dir` at a folder holding them under the model keys, set `dtype: fp32` and
`optimizer: adamw` in `loralei.yaml`, and run the CLI with `LORALEI_DEVICE=cpu`. That is a
smoke test of the plumbing, not a way to train.

---

## License

MIT - see [LICENSE](LICENSE).
