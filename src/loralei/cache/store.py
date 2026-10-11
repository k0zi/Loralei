"""Where pre-computed latents and text embeddings live.

Caching both means the VAE and the text encoder are never loaded in the training process
at all - which is what makes a 6B (or 20B) transformer fit on a 16 GiB card.

The two directories are chosen upstream and arrive in the job: each is keyed by everything
that would change its contents (model, VAE, geometry; text encoder, truncation length), so
a stale cache is impossible rather than merely unlikely. Only the per-file names are
derived here, and they must stay in step with `CacheLayout` on the C# side.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def caption_hash(caption: str) -> str:
    return hashlib.sha256(caption.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class CacheLayout:
    latents: Path
    text: Path

    def latent_path(self, key: str, width: int, height: int) -> Path:
        return self.latents / f"{key}_{width}x{height}.safetensors"

    def text_path(self, caption: str) -> Path:
        return self.text / f"{caption_hash(caption)}.safetensors"

    def ensure(self) -> None:
        self.latents.mkdir(parents=True, exist_ok=True)
        self.text.mkdir(parents=True, exist_ok=True)

    def write_meta(self, which: str, meta: dict) -> None:
        target = (self.latents if which == "latents" else self.text) / "meta.json"
        target.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
