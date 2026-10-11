"""Write the trained adapter out, with enough metadata to reproduce the run."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .. import __version__

DIFFUSERS_PREFIX = "transformer."


def extract_lora_state_dict(
    transformer, adapter_name: str = "default", scale: float = 1.0
) -> dict[str, torch.Tensor]:
    """PEFT weights, renamed to what `pipe.load_lora_weights()` expects.

    PEFT stores `layers.0.attention.to_q.lora_A.default.weight`; diffusers wants
    `transformer.layers.0.attention.to_q.lora_A.weight`.

    The diffusers-native format carries no alpha, so loaders assume `alpha == rank`
    and apply a scale of 1.0. When the run used a different alpha we fold the
    difference into B here - otherwise the exported LoRA would be quietly stronger
    or weaker than the one that was trained.
    """
    from peft import get_peft_model_state_dict

    raw = get_peft_model_state_dict(transformer, adapter_name=adapter_name)
    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        clean = key.replace(f".{adapter_name}.weight", ".weight").replace(
            f".{adapter_name}.", "."
        )
        if clean.startswith("base_model.model."):
            clean = clean[len("base_model.model.") :]
        tensor = value.detach().to(torch.float32).cpu()
        if scale != 1.0 and ".lora_B." in clean:
            tensor = tensor * scale
        out[DIFFUSERS_PREFIX + clean] = tensor
    return out


def build_metadata(cfg, extra: dict | None = None) -> dict[str, str]:
    meta = {
        "loralei_version": __version__,
        "model": cfg.model,
        "rank": str(cfg.rank),
        "alpha": str(cfg.effective_alpha),
        "resolution": str(cfg.resolution),
        "learning_rate": str(cfg.learning_rate),
        "optimizer": cfg.optimizer,
        "timestep_sampling": cfg.timestep_sampling,
        "flow_shift": str(cfg.flow_shift),
        "weighting_scheme": cfg.weighting_scheme,
        "config": json.dumps(cfg.dump(), sort_keys=True, default=str),
    }
    meta.update({k: str(v) for k, v in (extra or {}).items()})
    return meta


def save_lora(
    transformer,
    path: Path,
    cfg,
    extra_metadata: dict | None = None,
    adapter_name: str = "default",
) -> Path:
    state = extract_lora_state_dict(
        transformer, adapter_name, scale=cfg.effective_alpha / cfg.rank
    )
    if not state:
        raise RuntimeError("no LoRA weights found on the transformer - was an adapter added?")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, path, metadata=build_metadata(cfg, extra_metadata))
    return path
