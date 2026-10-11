"""VAE latent pre-caching.

We store the *distribution* (mean and std), not a single sample, so training can
draw fresh latent noise every epoch while the expensive VAE pass happens once. Both are
stored already normalised into the transformer's latent space; which VAE that is, and
how it normalises, is the model family's business (`families.Family.encode`).

Two ROCm 7.1 / gfx1200 faults are worked around here, both measured against a CPU
fp32 reference on that hardware with a 2D `AutoencoderKL`. The 3D VAEs the current
models use run the same MIOpen convolutions, so the guards stay:

1. **bf16 encode is wrong.** Not cold-start, not noise: every image, every repeat,
   14-21% relative error - far beyond bf16's ~0.4% per-element precision. So we encode
   in fp32 (VAE_DTYPE) and cast the *result* to bf16.
2. **The first conv of a given shape returns garbage** - usually all-NaN, sometimes
   finite-but-wrong, which is worse because nothing catches it. MIOpen appears to
   return the output buffer of its algorithm search. We therefore encode each new
   shape twice and keep the second result, and still verify finiteness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from ..families import Family
from ..job import Item, Plan
from .store import CacheLayout

#: Always encode in fp32 - see the module docstring. Costs ~0.5 GiB of VRAM.
VAE_DTYPE = torch.float32

#: Two passes count as agreeing within this relative tolerance. Well above MIOpen's
#: non-deterministic reduction noise, far below a failed cold-start pass.
AGREEMENT_TOLERANCE = 1e-3


@dataclass
class LatentStats:
    written: int = 0
    reused: int = 0
    failed: int = 0
    discarded_warmups: int = 0
    """Cold-start passes thrown away - one per distinct tensor shape."""
    cpu_fallbacks: int = 0


def resize_to_bucket(image: Image.Image, bucket: tuple[int, int]) -> Image.Image:
    """Cover the bucket, then centre-crop - never distort the aspect ratio."""
    target_w, target_h = bucket
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = max(target_w, round(src_w * scale)), max(target_h, round(src_h * scale))
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    import numpy as np

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def load_vae(family: Family, path: Path, dtype: torch.dtype, device: str, warmup: bool = True):
    """Load the VAE and burn its cold-start pass.

    Measured on gfx1200 / ROCm 7.1: the first convolution forward for a given shape
    returns garbage (all-NaN, or finite-but-wrong by >100%) while MIOpen selects a
    kernel. This warmup primes the general kernel cache; cache_latents() additionally
    burns one pass per distinct tensor shape, which is what actually covers it.
    """
    vae = family.load_vae(path, dtype, device)
    if warmup and device.startswith("cuda"):
        with torch.no_grad():
            family.encode(vae, torch.zeros(1, 3, 64, 64, device=device, dtype=dtype))
        torch.cuda.synchronize(device)
    return vae


def cache_latents(
    family: Family,
    items: list[Item],
    plan: Plan,
    layout: CacheLayout,
    vae_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = VAE_DTYPE,
    batch_size: int = 1,
    force: bool = False,
    fallback_to_cpu: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> LatentStats:
    layout.ensure()
    stats = LatentStats()

    pending: list[Item] = []
    for item in items:
        bucket = plan.bucket_for(item)
        if not force and layout.latent_path(item.key, *bucket).is_file():
            stats.reused += 1
        else:
            pending.append(item)

    if not pending:
        return stats

    vae = load_vae(family, vae_path, dtype, device)
    cpu_vae = None
    trusted_shapes: set[torch.Size] = set()

    try:
        for index, group in enumerate(_grouped_by_bucket(pending, plan, batch_size)):
            bucket, batch = group
            tensors = []
            for item in batch:
                with Image.open(item.image) as im:
                    tensors.append(image_to_tensor(resize_to_bucket(im, bucket)))
            pixels = torch.stack(tensors).to(device=device, dtype=dtype)

            verify = pixels.shape not in trusted_shapes
            try:
                mean, std, discarded = _encode_verified(family, vae, pixels, verify)
            except RuntimeError:
                if device == "cpu" or not fallback_to_cpu:
                    stats.failed += len(batch)
                    raise
                # The GPU path is a best-effort optimisation; correctness is not
                # negotiable, so drop to the CPU VAE and keep going.
                if cpu_vae is None:
                    cpu_vae = load_vae(family, vae_path, VAE_DTYPE, "cpu", warmup=False)
                stats.cpu_fallbacks += len(batch)
                mean, std, discarded = _encode_verified(family, cpu_vae, pixels.cpu(), verify=False)
            stats.discarded_warmups += discarded
            # The cold-start fault is per tensor shape, not per process: a new bucket
            # gets its own bad first call. Verify once per shape, then trust it - later
            # calls with that shape are bit-identical.
            trusted_shapes.add(pixels.shape)

            for item, m, s in zip(batch, mean, std):
                save_file(
                    {"mean": m.to(torch.bfloat16).cpu(), "std": s.to(torch.bfloat16).cpu()},
                    layout.latent_path(item.key, *bucket),
                    metadata={"image": item.image.name, "bucket": f"{bucket[0]}x{bucket[1]}"},
                )
                stats.written += 1
            if progress:
                progress(stats.written, len(pending))
    finally:
        del vae, cpu_vae
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    layout.write_meta(
        "latents",
        {
            "vae": str(vae_path),
            "family": family.spec.family,
            "resolution": plan.resolution,
            "buckets": {f"{w}x{h}": len(v) for (w, h), v in plan.by_bucket.items()},
        },
    )
    return stats


def _encode_verified(
    family: Family, vae, pixels: torch.Tensor, verify: bool, attempts: int = 8
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Encode, and when `verify` is set, only accept a result two passes agree on.

    The first real conv of a process on gfx1200 returns garbage - sometimes all-NaN,
    sometimes finite but off by >100%, which no cheap sanity check would catch. Later
    calls agree with each other, so "two consecutive passes agree" is both a sufficient
    test and a self-terminating one.

    The comparison is a relative tolerance rather than `torch.equal`: at 1024^2 MIOpen
    picks a non-deterministic reduction, so repeated passes differ in the last bits.
    A bad pass is wrong by more than 100%, so 0.1% separates them with room to spare.

    How many passes it takes to settle depends on the shape: 512^2 needs one throwaway,
    1024^2 needs two, hence the generous attempt budget.
    """
    discarded = 0
    previous: torch.Tensor | None = None
    for _ in range(attempts):
        with torch.no_grad():
            mean, std = family.encode(vae, pixels)
        finite = torch.isfinite(mean).all() and torch.isfinite(std).all()
        if finite and not verify:
            return mean, std, discarded
        if finite and previous is not None:
            spread = (previous - mean).abs().max()
            magnitude = mean.abs().max().clamp(min=1e-6)
            if spread <= AGREEMENT_TOLERANCE * magnitude:
                return mean, std, discarded
        previous = mean if finite else None
        discarded += 1
    raise RuntimeError(
        f"the VAE did not produce a reproducible result in {attempts} passes for shape "
        f"{tuple(pixels.shape)} - this is a ROCm/MIOpen fault, not bad input. Re-run, or "
        "encode on the other card with --gpu."
    )


def _grouped_by_bucket(
    items: Iterable[Item], plan: Plan, batch_size: int
) -> Iterable[tuple[tuple[int, int], list[Item]]]:
    """Batch only within a bucket - mixed shapes cannot be stacked."""
    by_bucket: dict[tuple[int, int], list[Item]] = {}
    for item in items:
        by_bucket.setdefault(plan.bucket_for(item), []).append(item)
    for bucket, group in by_bucket.items():
        for start in range(0, len(group), batch_size):
            yield bucket, group[start : start + batch_size]


def load_latent(layout: CacheLayout, key: str, bucket: tuple[int, int]) -> dict[str, torch.Tensor]:
    return load_file(layout.latent_path(key, *bucket))
