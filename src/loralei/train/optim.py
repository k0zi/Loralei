"""Optimizer and schedule selection, with a working fallback on every path."""

from __future__ import annotations

import math
from typing import Iterable

import torch


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    kind: str,
    learning_rate: float,
    weight_decay: float = 0.01,
) -> tuple[torch.optim.Optimizer, str]:
    """Returns the optimizer and the name actually used (it may have fallen back)."""
    params = list(params)
    if kind == "adamw8bit":
        try:
            import bitsandbytes as bnb

            return (
                bnb.optim.AdamW8bit(params, lr=learning_rate, weight_decay=weight_decay),
                "adamw8bit",
            )
        except (ImportError, RuntimeError):
            kind = "adamw"
    if kind == "adafactor":
        from transformers.optimization import Adafactor

        return (
            Adafactor(
                params,
                lr=learning_rate,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            ),
            "adafactor",
        )
    return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay), "adamw"


def build_scheduler(
    optimizer: torch.optim.Optimizer, kind: str, warmup_steps: int, total_steps: int
):
    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return step / max(1, warmup_steps)
        if kind == "constant_with_warmup" or kind == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if kind == "linear":
            return 1.0 - progress
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
