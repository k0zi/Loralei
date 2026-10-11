"""Render images from a base model, optionally with a LoRA applied.

Runs as its own process so it can be pinned to the second card while training holds
the first, and so the text encoder and VAE it needs never inflate the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .cache.latents import VAE_DTYPE
from .families import family_of
from .job import Components, Prepared, Spec


@dataclass
class SampleResult:
    paths: list[Path]
    verify: object | None = None


def build_pipeline(
    spec: Spec,
    components: Components,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    offload: bool = True,
):
    """Assemble the pipeline from resolved components.

    Which classes, and which components beyond the usual five, is the model family's
    business. The VAE is kept in fp32 everywhere: the bf16 path was measured to be
    numerically broken on gfx1200 (see loralei/cache/latents.py), and a wrong decode
    looks like a bad LoRA.
    """
    pipe = family_of(spec).build_pipeline(components, dtype, VAE_DTYPE)
    if device.type != "cuda":
        return pipe.to(device)
    if offload:
        pipe.enable_model_cpu_offload(device=str(device))
    else:
        pipe.to(device)
    return pipe


def guard_vae_decode(pipe) -> None:
    """Re-run a decode that came back degenerate.

    Even with a shape-matched warmup, a decode occasionally returns NaN or a flat
    image - the same gfx1200 fault that makes latent caching run on the CPU. A
    rendered frame that is uniformly one colour is never what the model produced, so
    retrying (and finally falling back to the CPU VAE) costs nothing and removes a
    class of "the LoRA broke it" false alarms.
    """
    if getattr(pipe.vae, "_loralei_guarded", False):
        return
    original = pipe.vae.decode

    def guarded(latents, *args, **kwargs):
        for attempt in range(3):
            output = original(latents, *args, **kwargs)
            sample = output[0] if isinstance(output, tuple) else output.sample
            if torch.isfinite(sample).all() and sample.float().std() > 1e-3:
                return output
            if attempt == 1:
                pipe.vae.to("cpu")
                return original(latents.to("cpu"), *args, **kwargs)
        return output

    pipe.vae.decode = guarded
    pipe.vae._loralei_guarded = True


def warm_vae(pipe, family, device: torch.device, height: int, width: int) -> None:
    """Burn the cold-start conv pass at the exact shape we are about to decode.

    The gfx1200 fault is per tensor shape, so a warmup at some other size does not
    help - a 16x16 primer followed by a real 128x128 decode still produces garbage
    (a red band on black, in the first run that caught this).
    """
    if device.type != "cuda":
        return
    with torch.no_grad():
        dummy = torch.zeros(family.latent_shape(pipe.vae, height, width), device=device, dtype=VAE_DTYPE)
        pipe.vae.to(device)
        pipe.vae.decode(dummy)
    torch.cuda.synchronize(device)


def generate(
    spec: Spec,
    components: Components,
    prompts: list[str],
    output_dir: Path,
    lora: Path | None = None,
    steps: int = 20,
    guidance_scale: float = 4.0,
    width: int = 1024,
    height: int = 1024,
    seed: int = 0,
    device: torch.device | None = None,
    negative_prompt: str = "",
    prefix: str = "sample",
    verify_lora: bool = True,
    on_image=None,
    dtype: torch.dtype = torch.bfloat16,
) -> SampleResult:
    from .hw import pick_training_device, probe_devices

    family = family_of(spec)
    device = device or torch.device(f"cuda:{pick_training_device(probe_devices()).index}")
    pipe = build_pipeline(spec, components, device, dtype)

    verification = None
    if lora is not None:
        pipe.load_lora_weights(str(lora), adapter_name="loralei")
        if verify_lora:
            from .lora.verify import verify_against_transformer

            verification = verify_against_transformer(lora, pipe.transformer, "loralei")

    warm_vae(pipe, family, device, height, width)
    guard_vae_decode(pipe)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, prompt in enumerate(prompts):
        generator = torch.Generator(device="cpu").manual_seed(seed + index)
        image = family.render(
            pipe, prompt, negative_prompt, height, width, steps, guidance_scale, generator
        )
        path = output_dir / f"{prefix}_{index:02d}.png"
        image.save(path)
        paths.append(path)
        if on_image:
            on_image(path)

    return SampleResult(paths=paths, verify=verification)


def device_override(hip_index: int) -> dict[str, str]:
    """Pin a child process to a single *already-visible* device.

    The index is relative to the current HIP_VISIBLE_DEVICES list, matching what
    torch reports as ``cuda:<index>``.
    """
    import os

    visible = os.environ.get("HIP_VISIBLE_DEVICES")
    if not visible:
        return {"HIP_VISIBLE_DEVICES": str(hip_index)}
    parts = visible.split(",")
    if hip_index >= len(parts):
        raise IndexError(f"device {hip_index} not in HIP_VISIBLE_DEVICES={visible}")
    return {"HIP_VISIBLE_DEVICES": parts[hip_index]}


def spawn_sampler(
    prepared: Prepared,
    lora: Path,
    step: int,
    output_dir: Path,
    device_index: int | None = None,
) -> "subprocess.Popen | None":
    """Fire off validation renders in a detached process.

    Sampling needs the text encoder and the VAE, which the trainer has deliberately
    unloaded - reloading them in-process would undo the whole memory plan. Running on
    the other card instead costs the training loop nothing at all.

    The child is another worker, handed a `sample` job built from this one: the model
    is already resolved, so it gets the same component folders rather than searching.
    """
    import json
    import os
    import subprocess
    import sys

    cfg = prepared.cfg
    if not cfg.sample_prompts:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "job": "sample",
        "spec": prepared.spec.as_dict(),
        "components": prepared.components.as_dict(),
        "prompts": list(cfg.sample_prompts),
        "out": str(output_dir),
        "lora": str(lora),
        "steps": cfg.sample_steps,
        "cfg": cfg.sample_cfg,
        "width": cfg.resolution,
        "height": cfg.resolution,
        "seed": cfg.seed,
        "gpu": None,
        "dtype": cfg.dtype,
    }
    job_path = output_dir / f"sample-{step:06d}.job.json"
    job_path.write_text(json.dumps(job, indent=2))

    env = dict(os.environ)
    if device_index is not None:
        env.update(device_override(device_index))
    log = output_dir / f"sample-{step:06d}.log"
    handle = log.open("w")
    return subprocess.Popen(
        [sys.executable, "-m", "loralei", str(job_path)],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
