import torch
from torch import nn

from loralei.train.fp8 import FP8_DTYPE, quantize_linear_, quantize_module_


def test_quantized_linear_keeps_outputs_close():
    torch.manual_seed(0)
    linear = nn.Linear(512, 512, bias=False).to(torch.bfloat16).requires_grad_(False)
    x = torch.randn(16, 512, dtype=torch.bfloat16)
    reference = linear(x).float()

    assert quantize_linear_(linear)
    assert linear.weight.dtype == FP8_DTYPE

    out = linear(x).float()
    relative = (out - reference).norm() / reference.norm()
    # e4m3 with per-row scales: a few percent on the norm, not a few tens.
    assert relative < 0.05


def test_quantization_is_idempotent():
    linear = nn.Linear(256, 256).to(torch.bfloat16).requires_grad_(False)
    assert quantize_linear_(linear)
    assert not quantize_linear_(linear)


def test_trainable_layers_are_left_alone():
    linear = nn.Linear(256, 256).to(torch.bfloat16)
    linear.weight.requires_grad_(True)
    assert not quantize_linear_(linear)
    assert linear.weight.dtype == torch.bfloat16


def test_skip_patterns_are_honoured():
    module = nn.ModuleDict(
        {
            "t_embedder": nn.Linear(512, 512),
            "block": nn.Linear(512, 512),
        }
    ).to(torch.bfloat16)
    module.requires_grad_(False)
    stats = quantize_module_(module, skip_patterns=("t_embedder",), min_numel=1)
    assert stats.layers == 1
    assert module["t_embedder"].weight.dtype == torch.bfloat16
    assert module["block"].weight.dtype == FP8_DTYPE
