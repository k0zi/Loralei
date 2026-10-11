"""Stream frozen transformer blocks between host RAM and VRAM.

A 6B Z-Image is 12.3 GiB of bf16 weights; a 16 GiB card has no room left for
activations. Since LoRA training freezes the base weights, most blocks can live in
pinned host memory and be copied in just before they run - and, because they never
change, they never need to be copied back (`h2d_only`). With 122 GiB of RAM the host
side is free; the cost is PCIe bandwidth, which a prefetch stream hides behind the
previous block's compute.

Trainable parameters (the LoRA adapters) are deliberately left on the GPU: they are
tiny, the optimizer holds state for them, and moving them would break the graph.

Eviction is driven by a bounded residency window rather than by forward hooks,
because **post-forward hooks do not fire during a checkpointed backward**: PyTorch's
non-reentrant checkpointing aborts the recomputation with an internal exception as
soon as it has the saved tensors it needs, so the module's forward never returns.
Measured here: 34 loads and 33 evictions in the forward, then 29 more loads and *zero*
evictions in the backward, ending with 450 parameters stranded on the GPU and peak
VRAM back at the unswapped 10.7 GiB.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class SwapStats:
    blocks_swapped: int = 0
    host_bytes: int = 0
    loads: int = 0
    evictions: int = 0

    @property
    def host_gib(self) -> float:
        return self.host_bytes / 1024**3


@dataclass
class _BlockState:
    module: nn.Module
    cpu_params: dict[str, torch.Tensor] = field(default_factory=dict)
    gpu_params: dict[str, torch.Tensor] = field(default_factory=dict)
    resident: bool = False
    event: torch.cuda.Event | None = None
    """Set when the copy was issued on the prefetch stream and not yet waited on."""


class BlockSwapper:
    """Keeps the last `keep_on_gpu` blocks resident and streams the rest on demand."""

    def __init__(
        self,
        blocks: list[nn.Module],
        num_swap: int,
        device: torch.device,
        pin_memory: bool = True,
        prefetch: int = 1,
    ) -> None:
        self.device = device
        self.prefetch = max(0, prefetch)
        #: How many swapped blocks may be resident at once. One for the block being
        #: run, plus the ones being prefetched, plus a little slack so a block whose
        #: gradient is still being consumed is not evicted from under autograd.
        self.residency_budget = self.prefetch + 3
        self._resident_order: list[int] = []
        self.stats = SwapStats()
        self._states: dict[int, _BlockState] = {}
        self._order: list[nn.Module] = list(blocks)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        if num_swap <= 0:
            return

        # Swap the *last* N blocks: the first ones run again soonest during the
        # backward recompute, so keeping them resident costs the fewest transfers.
        to_swap = self._order[len(self._order) - num_swap :]
        self._stream = torch.cuda.Stream(device=device) if self.prefetch else None

        for position, block in enumerate(self._order):
            if block not in to_swap:
                continue
            state = _BlockState(module=block)
            for name, param in block.named_parameters(recurse=True):
                if param.requires_grad:
                    continue  # LoRA adapters stay put
                cpu = param.data.detach().to("cpu", copy=True)
                if pin_memory:
                    try:
                        cpu = cpu.pin_memory()
                    except RuntimeError:
                        pass
                state.cpu_params[name] = cpu
                param.data = cpu
                self.stats.host_bytes += cpu.numel() * cpu.element_size()
            self._states[position] = state
            self.stats.blocks_swapped += 1

        for position, state in self._states.items():
            self._handles.append(
                state.module.register_forward_pre_hook(self._make_pre_hook(position))
            )
            self._handles.append(
                state.module.register_forward_hook(self._make_post_hook(position))
            )

        torch.cuda.empty_cache()

    # -- hooks ---------------------------------------------------------------

    def _make_pre_hook(self, position: int):
        def hook(module, args):
            self._ensure_resident(position)
            for offset in range(1, self.prefetch + 1):
                self._prefetch(position + offset)
            self._trim()

        return hook

    def _make_post_hook(self, position: int):
        def hook(module, args, output):
            # Fires in the ordinary forward but not during a checkpointed backward,
            # so it is an optimisation, not the mechanism - see _trim().
            self._evict(position)

        return hook

    def _trim(self) -> None:
        """Keep the resident set inside its budget, oldest first.

        Autograd holds its own references to anything it saved, so moving a
        parameter's `.data` back to the host here cannot invalidate a pending
        backward - it only drops *our* reference to the device copy.
        """
        while len(self._resident_order) > self.residency_budget:
            self._evict(self._resident_order[0])

    # -- movement ------------------------------------------------------------

    def _ensure_resident(self, position: int) -> None:
        """Block until this block's weights are actually on the device.

        Getting this wrong is silent: an unsynchronised prefetch lets the block run
        against a half-copied buffer, and training converges to a visibly different
        (wrong) loss instead of crashing. Measured: 1.93 vs 1.03 on the same seed.
        """
        state = self._states.get(position)
        if state is None:
            return
        if state.resident:
            self._join_prefetch(state)
            return
        self._copy_in(state)

    def _join_prefetch(self, state: _BlockState) -> None:
        if state.event is None:
            return
        current = torch.cuda.current_stream(self.device)
        current.wait_event(state.event)
        # Memory allocated on the prefetch stream must be tied to the consuming
        # stream, or the caching allocator may hand it out again while it is in use.
        for tensor in state.gpu_params.values():
            tensor.record_stream(current)
        state.event = None

    def _prefetch(self, position: int) -> None:
        state = self._states.get(position)
        if state is None or state.resident or self._stream is None:
            return
        self._stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._stream):
            self._copy_in(state)
            event = torch.cuda.Event()
            event.record(self._stream)
            state.event = event

    def _copy_in(self, state: _BlockState) -> None:
        params = dict(state.module.named_parameters(recurse=True))
        state.gpu_params = {}
        for name, cpu in state.cpu_params.items():
            gpu = cpu.to(self.device, non_blocking=True)
            params[name].data = gpu
            state.gpu_params[name] = gpu
        state.resident = True
        self.stats.loads += 1
        position = self._position_of(state)
        if position is not None and position not in self._resident_order:
            self._resident_order.append(position)

    def _evict(self, position: int) -> None:
        state = self._states.get(position)
        if state is None or not state.resident:
            return
        self._join_prefetch(state)
        params = dict(state.module.named_parameters(recurse=True))
        for name, cpu in state.cpu_params.items():
            params[name].data = cpu
        state.gpu_params = {}
        state.resident = False
        self.stats.evictions += 1
        if position in self._resident_order:
            self._resident_order.remove(position)

    def _position_of(self, state: _BlockState) -> int | None:
        for position, candidate in self._states.items():
            if candidate is state:
                return position
        return None

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for position in list(self._states):
            self._evict(position)


def collect_blocks(transformer, spec) -> list[nn.Module]:
    """Every swappable block of a transformer, in execution order."""
    blocks: list[nn.Module] = []
    for attr in (spec.blocks_attr, *spec.extra_block_attrs):
        container = transformer
        for part in attr.split("."):
            container = getattr(container, part, None)
            if container is None:
                break
        if isinstance(container, nn.ModuleList):
            blocks.extend(container)
    return blocks


def max_swappable(transformer, spec) -> int:
    return len(collect_blocks(transformer, spec))
