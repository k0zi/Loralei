"""On-disk layout of a training run, as far as the training loop writes it."""

from __future__ import annotations

import json
import time
from pathlib import Path


def prepare(output_root: Path, name: str) -> dict[str, Path]:
    base = Path(output_root) / name
    paths = {
        "root": base,
        "logs": base / "logs",
        "checkpoints": base / "checkpoints",
        "samples": base / "samples",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_manifest(base: Path, **fields) -> Path:
    """Merge fields into manifest.json.

    Written from inside the loop rather than by whatever started it, so that an
    interrupted run still gets stamped `failed` even if its parent is already gone.
    """
    path = base / "manifest.json"
    existing = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except ValueError:
            existing = {}
    existing.update(fields)
    existing["updated_at"] = time.time()
    path.write_text(json.dumps(existing, indent=2, sort_keys=True, default=str))
    return path
