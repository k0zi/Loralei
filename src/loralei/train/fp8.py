"""Store frozen Linear weights as float8, dequantizing inside forward.

This is *storage* quantization, not an fp8 GEMM. `torch._scaled_mm` upcasts to fp32
on gfx1200 and hipBLASLt's fp8 support for RDNA4 is too fresh to depend on, so the
matmul stays bf16 - only the resident copy of the weights shrinks, roughly halving
the transformer's footprint (12.3 GiB -> ~6.4 GiB for Z-Image).

Per-output-channel scales keep the error small: e4m3 has a 448 maximum, so each row
is scaled to use the full range instead of being clipped by a single global factor.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import torch
from torch import nn

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


@dataclass
class QuantStats:
    layers: int = 0
    bytes_before: int = 0
    bytes_after: int = 0

    @property
    def saved_gib(self) -> float:
        return (self.bytes_before - self.bytes_after) / 1024**3


def _fp8_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    weight = self.weight.to(x.dtype) * self._fp8_scale.to(x.dtype)
    return nn.functional.linear(x, weight, self.bias)


def quantize_linear_(linear: nn.Linear) -> bool:
    """Replace a frozen Linear's weight with fp8 + per-row scale. Idempotent."""
    if getattr(linear, "_fp8_quantized", False):
        return False
    weight = linear.weight
    if weight.requires_grad or weight.dtype == FP8_DTYPE:
        return False

    data = weight.data.float()
    scale = data.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / FP8_MAX
    quantized = (data / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

    linear.weight = nn.Parameter(quantized, requires_grad=False)
    linear.register_buffer("_fp8_scale", scale.to(torch.bfloat16), persistent=False)
    linear.forward = types.MethodType(_fp8_forward, linear)
    linear._fp8_quantized = True
    return True


def quantize_module_(
    module: nn.Module,
    skip_patterns: tuple[str, ...] = ("t_embedder", "cap_embedder", "x_embedder", "final_layer"),
    min_numel: int = 1 << 18,
) -> QuantStats:
    """Quantize the big frozen Linears, leaving precision-sensitive ones alone.

    `skip_patterns` mirrors ZImageTransformer2DModel._skip_layerwise_casting_patterns -
    timestep and caption embedders feed every block's modulation and are the first
    place low precision shows up as colour drift.
    """
    stats = QuantStats()
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        if any(pattern in name for pattern in skip_patterns):
            continue
        if child.weight.numel() < min_numel or child.weight.requires_grad:
            continue
        before = child.weight.numel() * child.weight.element_size()
        if quantize_linear_(child):
            stats.layers += 1
            stats.bytes_before += before
            stats.bytes_after += child.weight.numel() + child._fp8_scale.numel() * 2
    return stats
