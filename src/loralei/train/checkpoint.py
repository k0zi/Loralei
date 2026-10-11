"""Checkpoint and resume.

A 2000-step Z-Image run is a couple of hours; losing it to a reboot or a stray OOM
is avoidable. Each checkpoint is the LoRA weights (a normal, loadable safetensors)
plus a sidecar holding the optimizer, schedule and RNG state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch

STATE_SUFFIX = ".state.pt"


@dataclass
class Checkpoint:
    step: int
    weights: Path
    state: Path


def checkpoint_paths(checkpoints_dir: Path, name: str, step: int) -> tuple[Path, Path]:
    stem = f"{name}-{step:06d}"
    return checkpoints_dir / f"{stem}.safetensors", checkpoints_dir / f"{stem}{STATE_SUFFIX}"


def find_latest(checkpoints_dir: Path, name: str) -> Checkpoint | None:
    if not checkpoints_dir.is_dir():
        return None
    pattern = re.compile(rf"^{re.escape(name)}-(\d+)\.safetensors$")
    best: Checkpoint | None = None
    for path in checkpoints_dir.iterdir():
        match = pattern.match(path.name)
        if not match:
            continue
        step = int(match.group(1))
        state = path.with_name(path.stem + STATE_SUFFIX)
        if not state.is_file():
            continue  # weights without a sidecar cannot resume, only be used
        if best is None or step > best.step:
            best = Checkpoint(step=step, weights=path, state=state)
    return best


def save_state(
    path: Path,
    step: int,
    epoch: int,
    optimizer,
    scheduler,
    generator: torch.Generator,
    loss_ema: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "loss_ema": loss_ema,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "generator": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
        },
        path,
    )
    return path


def load_state(
    checkpoint: Checkpoint, transformer, optimizer, scheduler, generator: torch.Generator
) -> tuple[int, int, float | None]:
    """Restore weights and optimizer state. Returns (step, epoch, loss_ema)."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    weights = load_file(checkpoint.weights)
    # Undo the export-time renaming: `transformer.<path>.lora_A.weight` -> PEFT's layout.
    restored = {
        key.removeprefix("transformer."): value
        for key, value in weights.items()
    }
    set_peft_model_state_dict(transformer, restored)

    blob = torch.load(checkpoint.state, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(blob["optimizer"])
    scheduler.load_state_dict(blob["scheduler"])
    generator.set_state(blob["generator"])
    torch.set_rng_state(blob["torch_rng"])
    return int(blob["step"]), int(blob["epoch"]), blob.get("loss_ema")
