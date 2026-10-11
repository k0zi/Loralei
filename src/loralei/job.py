"""The job handed to this worker, as objects the torch code can use.

Everything here is decided before Python starts: the C# side reads the config layers,
finds the model on disk, scans the dataset, plans the buckets and names the cache
directories. This module only gives that JSON document attribute access - there is no
resolving, validating or searching left to do on this side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache.store import CacheLayout

#: Config keys that are filesystem locations and should be `Path`s, not strings.
_PATH_KEYS = ("dataset", "output_root", "model_path")


class Config:
    """The resolved run configuration: one attribute per `config.resolved.yaml` key."""

    def __init__(self, values: dict[str, Any]) -> None:
        self._values = dict(values)
        for key, value in values.items():
            if key in _PATH_KEYS and value is not None:
                value = Path(value)
            setattr(self, key, value)

    @property
    def effective_alpha(self) -> float:
        return float(self.alpha) if self.alpha is not None else float(self.rank)

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.name

    def dump(self) -> dict[str, Any]:
        """Exactly what was resolved, for the safetensors metadata."""
        return dict(self._values)


@dataclass(frozen=True)
class Spec:
    """The architecture facts the trainer needs about a base model."""

    key: str
    pipeline_class: str
    transformer_class: str
    blocks_attr: str
    extra_block_attrs: tuple[str, ...] = ()
    max_sequence_length: int = 512
    family: str = ""
    """Which entry of `families.FAMILIES` knows how to drive this model."""

    @classmethod
    def of(cls, raw: dict[str, Any]) -> Spec:
        return cls(
            key=raw["key"],
            pipeline_class=raw["pipeline_class"],
            transformer_class=raw["transformer_class"],
            blocks_attr=raw["blocks_attr"],
            extra_block_attrs=tuple(raw.get("extra_block_attrs") or ()),
            max_sequence_length=int(raw.get("max_sequence_length", 512)),
            family=raw.get("family", ""),
        )

    def as_dict(self) -> dict[str, Any]:
        """The JSON this was read from - what a sampler child is handed."""
        return {
            "key": self.key,
            "pipeline_class": self.pipeline_class,
            "transformer_class": self.transformer_class,
            "blocks_attr": self.blocks_attr,
            "extra_block_attrs": list(self.extra_block_attrs),
            "max_sequence_length": self.max_sequence_length,
            "family": self.family,
        }


@dataclass(frozen=True)
class Item:
    image: Path
    caption: str
    key: str
    """Stable identifier used for cache filenames."""
    bucket: tuple[int, int]


@dataclass
class Plan:
    """Which bucket each image was assigned to. Assigned upstream; only looked up here."""

    resolution: int
    by_bucket: dict[tuple[int, int], list[Item]] = field(default_factory=dict)

    @classmethod
    def of(cls, items: list[Item], resolution: int) -> Plan:
        plan = cls(resolution=resolution)
        for item in items:
            plan.by_bucket.setdefault(item.bucket, []).append(item)
        return plan

    def bucket_for(self, item: Item) -> tuple[int, int]:
        return item.bucket

    def summary(self) -> str:
        return ", ".join(f"{w}x{h}:{len(items)}" for (w, h), items in self.by_bucket.items())


class Components:
    """Resolved model folders. Asking for one that was not found is the error."""

    def __init__(self, model_key: str, paths: dict[str, str]) -> None:
        self._model_key = model_key
        self._paths = {name: Path(path) for name, path in paths.items()}

    def as_dict(self) -> dict[str, str]:
        return {name: str(path) for name, path in self._paths.items()}

    def path(self, component: str) -> Path:
        try:
            return self._paths[component]
        except KeyError:
            raise FileNotFoundError(
                f"{self._model_key}/{component} is not available locally. "
                f"Run `loralei models pull {self._model_key}` first."
            ) from None


@dataclass
class Prepared:
    """A dataset job: everything `cache` and `train` share."""

    cfg: Config
    spec: Spec
    components: Components
    items: list[Item]
    plan: Plan
    layout: CacheLayout
    fingerprint: str

    @classmethod
    def of(cls, job: dict[str, Any]) -> Prepared:
        cfg = Config(job["config"])
        spec = Spec.of(job["spec"])
        items = [
            Item(
                image=Path(raw["image"]),
                caption=raw["caption"],
                key=raw["key"],
                bucket=(int(raw["bucket"][0]), int(raw["bucket"][1])),
            )
            for raw in job["items"]
        ]
        return cls(
            cfg=cfg,
            spec=spec,
            components=Components(spec.key, job["components"]),
            items=items,
            plan=Plan.of(items, cfg.resolution),
            layout=CacheLayout(latents=Path(job["cache"]["latents"]), text=Path(job["cache"]["text"])),
            fingerprint=job.get("fingerprint", ""),
        )
