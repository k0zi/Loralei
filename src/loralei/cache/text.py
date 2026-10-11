"""Text encoder pre-caching.

Each model family embeds prompts differently, and getting it wrong is silent - the
LoRA simply learns against the wrong conditioning. How a prompt becomes tensors is
therefore not decided here: `families.Family.encode_text` mirrors the model's own
diffusers pipeline. This module only runs it once per unique caption and files the
result, which for some models is more than one tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from safetensors.torch import load_file, save_file

from ..families import Family, Text
from .store import CacheLayout


@dataclass
class TextStats:
    written: int = 0
    reused: int = 0


def cache_text_embeddings(
    family: Family,
    captions: list[str],
    layout: CacheLayout,
    components,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    force: bool = False,
    include_empty: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> TextStats:
    """Embed every unique caption once. The empty prompt is included for CFG sampling."""
    layout.ensure()
    stats = TextStats()

    unique = list(dict.fromkeys(captions))
    if include_empty and "" not in unique:
        unique.append("")

    pending = []
    for caption in unique:
        if not force and layout.text_path(caption).is_file():
            stats.reused += 1
        else:
            pending.append(caption)
    if not pending:
        return stats

    ctx = family.load_text(components, dtype, device)
    size = family.text_batch_size
    try:
        for start in range(0, len(pending), size):
            batch = pending[start : start + size]
            for caption, text in zip(batch, family.encode_text(ctx, batch)):
                save_file(
                    {
                        name: (tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor)
                        .cpu()
                        .contiguous()
                        for name, tensor in text.items()
                    },
                    layout.text_path(caption),
                    metadata={"caption": caption[:2000]},
                )
                stats.written += 1
            if progress:
                progress(stats.written, len(pending))
    finally:
        del ctx
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    layout.write_meta(
        "text",
        {
            "text_encoder": str(components.path("text_encoder")),
            "model_key": family.spec.key,
            "family": family.spec.family,
            "max_sequence_length": family.spec.max_sequence_length,
            "count": len(unique),
        },
    )
    return stats


def load_text_embedding(layout: CacheLayout, caption: str) -> Text:
    return load_file(layout.text_path(caption))
