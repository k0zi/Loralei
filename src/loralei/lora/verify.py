"""Prove a saved LoRA actually loads before anyone tries it in ComfyUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from safetensors.torch import safe_open


@dataclass
class VerifyResult:
    file_keys: int = 0
    attached_modules: int = 0
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unexpected and self.attached_modules > 0

    def summary(self) -> str:
        if self.ok:
            return f"{self.file_keys} tensors -> {self.attached_modules} adapted modules"
        parts = []
        if self.unexpected:
            parts.append(f"{len(self.unexpected)} unexpected (e.g. {self.unexpected[0]})")
        if self.missing:
            parts.append(f"{len(self.missing)} target(s) not adapted (e.g. {self.missing[0]})")
        if not self.attached_modules:
            parts.append("no adapter attached at all")
        return "; ".join(parts)


def lora_keys(path: Path) -> list[str]:
    with safe_open(path, framework="pt") as f:
        return list(f.keys())


def verify_against_transformer(path: Path, transformer, adapter_name: str = "default") -> VerifyResult:
    """Every `lora_A` key in the file must land on a real module, and vice versa."""
    result = VerifyResult()
    keys = lora_keys(path)
    result.file_keys = len(keys)

    wanted = set()
    for key in keys:
        if ".lora_A." not in key:
            continue
        name = key.split(".lora_A.")[0]
        wanted.add(name.removeprefix("transformer."))

    attached = set()
    for name, module in transformer.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is not None and adapter_name in lora_a:
            attached.add(name)
    result.attached_modules = len(attached)

    result.unexpected = sorted(wanted - attached)
    result.missing = sorted(attached - wanted)
    return result
