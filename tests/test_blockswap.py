import pytest
import torch
from torch import nn

from loralei.train.blockswap import BlockSwapper

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a ROCm device")


class Block(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.linear(x)


class Stack(nn.Module):
    def __init__(self, count: int = 8, dim: int = 256):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim) for _ in range(count))

    def forward(self, x):
        for block in self.blocks:
            x = block(x) + x
        return x


def test_swapping_does_not_change_the_result():
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    model = Stack().to(device).requires_grad_(False)
    x = torch.randn(2, 256, device=device)

    reference = model(x).clone()

    swapper = BlockSwapper(list(model.blocks), num_swap=6, device=device, prefetch=1)
    try:
        for _ in range(3):
            assert torch.equal(model(x), reference)
    finally:
        swapper.close()


def test_residency_stays_inside_the_budget():
    device = torch.device("cuda:0")
    model = Stack(count=12).to(device).requires_grad_(False)
    swapper = BlockSwapper(list(model.blocks), num_swap=12, device=device, prefetch=1)
    try:
        model(torch.randn(1, 256, device=device))
        assert len(swapper._resident_order) <= swapper.residency_budget
        assert swapper.stats.blocks_swapped == 12
        assert swapper.stats.host_bytes > 0
    finally:
        swapper.close()


def test_close_returns_every_block_to_the_host():
    device = torch.device("cuda:0")
    model = Stack(count=6).to(device).requires_grad_(False)
    swapper = BlockSwapper(list(model.blocks), num_swap=6, device=device, prefetch=0)
    model(torch.randn(1, 256, device=device))
    swapper.close()
    for block in model.blocks:
        assert not block.linear.weight.is_cuda
