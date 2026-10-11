"""The training loop.

By the time this runs there is no VAE and no text encoder in the process - only the
transformer, a LoRA adapter, and two caches on disk. That is deliberate: it is what
makes 12.3 GiB of frozen weights plus activations fit in 16 GiB.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.dataset import BucketBatchSampler, CachedDataset, collate
from ..families import family_of
from ..job import Prepared, Spec
from ..lora.save import save_lora
from ..run import layout as run_layout
from .blockswap import BlockSwapper, collect_blocks
from .checkpoint import checkpoint_paths, find_latest, load_state, save_state
from .fp8 import quantize_module_
from .objective import sample_sigmas
from .optim import build_optimizer, build_scheduler

GIB = 1024**3


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    loss_ema: float | None = None
    started_at: float = field(default_factory=time.time)
    peak_vram: int = 0

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at


def select_lora_targets(
    transformer: nn.Module, target_regex: str | None, exclude_regex: str | None
) -> list[str]:
    """Explicit module names, so the choice is auditable instead of a regex surprise."""
    import re

    include = re.compile(target_regex) if target_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    names: list[str] = []
    for name, module in transformer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if include and not include.match(name):
            continue
        if exclude and exclude.match(name):
            continue
        names.append(name)
    return names


def load_transformer(spec: Spec, path: Path, dtype: torch.dtype = torch.bfloat16):
    """Load onto CPU. Placement happens later so swapped blocks never touch VRAM."""
    import diffusers

    cls = getattr(diffusers, spec.transformer_class)
    model = cls.from_pretrained(path, dtype=dtype)
    model.requires_grad_(False)
    return model


def place_model(
    transformer: nn.Module,
    blocks: list[nn.Module],
    blocks_to_swap: int,
    device: torch.device,
) -> set[int]:
    """Move everything to the GPU except the frozen weights of the swapped blocks.

    Doing this before any `.to(device)` matters: moving the whole model first and
    swapping afterwards would need the full 12-41 GiB resident for a moment, which is
    exactly what we are trying to avoid.
    """
    swapped = set()
    if blocks_to_swap > 0:
        swapped = {id(b) for b in blocks[len(blocks) - blocks_to_swap :]}

    swapped_modules: set[int] = set()
    for block in blocks:
        if id(block) in swapped:
            for module in block.modules():
                swapped_modules.add(id(module))

    for module in transformer.modules():
        keep_on_host = id(module) in swapped_modules
        for param in module.parameters(recurse=False):
            if keep_on_host and not param.requires_grad:
                continue
            param.data = param.data.to(device)
        for name, buffer in list(module.named_buffers(recurse=False)):
            if buffer is not None:
                setattr(module, name, buffer.to(device))
    return swapped


def train(
    prepared: Prepared,
    device: torch.device,
    on_step: Callable[[TrainState, float], None] | None = None,
    on_save: Callable[[Path, int], None] | None = None,
    resume: bool = False,
) -> TrainState:
    cfg = prepared.cfg
    spec = prepared.spec
    family = family_of(spec)
    # Everything below assumes a ROCm card. A CPU device exists only so the whole path can
    # be smoke-tested with a tiny model on a machine without one.
    on_gpu = device.type == "cuda"
    paths = run_layout.prepare(cfg.output_root, cfg.name)

    dataset = CachedDataset(
        prepared.items,
        prepared.plan,
        prepared.layout,
        repeats=cfg.repeats,
        caption_dropout=cfg.caption_dropout,
    )
    sampler = BucketBatchSampler(dataset, cfg.batch_size, shuffle=True, seed=cfg.seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=on_gpu,
    )

    dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32
    transformer = load_transformer(spec, prepared.components.path("transformer"), dtype)

    targets = select_lora_targets(transformer, cfg.target_regex, cfg.exclude_regex)
    if not targets:
        raise RuntimeError(
            f"no Linear modules matched target_regex={cfg.target_regex!r} "
            f"(exclude={cfg.exclude_regex!r})"
        )

    from peft import LoraConfig

    transformer.add_adapter(
        LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.effective_alpha,
            lora_dropout=cfg.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=targets,
        )
    )

    quant_stats = None
    if cfg.fp8_base:
        quant_stats = quantize_module_(transformer)

    blocks = collect_blocks(transformer, spec)
    blocks_to_swap = min(cfg.blocks_to_swap, max(0, len(blocks) - 1)) if on_gpu else 0
    place_model(transformer, blocks, blocks_to_swap, device)

    swapper = None
    if blocks_to_swap:
        swapper = BlockSwapper(
            blocks,
            blocks_to_swap,
            device,
            pin_memory=cfg.swap_pin_memory,
            prefetch=cfg.swap_prefetch,
        )

    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer.train()

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    for param in trainable:
        param.data = param.data.to(device=device, dtype=torch.float32)

    optimizer, optimizer_name = build_optimizer(trainable, cfg.optimizer, cfg.learning_rate)
    scheduler = build_scheduler(optimizer, cfg.lr_scheduler, cfg.warmup_steps, cfg.max_steps)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    state = TrainState()

    if resume:
        checkpoint = find_latest(paths["checkpoints"], cfg.name)
        if checkpoint is None:
            raise FileNotFoundError(
                f"--resume asked for, but no resumable checkpoint in {paths['checkpoints']}"
            )
        state.step, state.epoch, state.loss_ema = load_state(
            checkpoint, transformer, optimizer, scheduler, generator
        )
        if on_save:
            on_save(checkpoint.weights, state.step)

    run_layout.write_manifest(
        paths["root"],
        model=cfg.model,
        state="running",
        step=0,
        lora_targets=len(targets),
        trainable_params=sum(p.numel() for p in trainable),
        blocks_total=len(blocks),
        blocks_swapped=blocks_to_swap,
        swap_host_gib=round(swapper.stats.host_gib, 2) if swapper else 0.0,
        fp8_layers=quant_stats.layers if quant_stats else 0,
        optimizer=optimizer_name,
        dataset=str(cfg.dataset),
        dataset_fingerprint=prepared.fingerprint,
        images=len(prepared.items),
    )

    samplers: list = []
    if on_gpu:
        torch.cuda.reset_peak_memory_stats(device)
    try:
        while state.step < cfg.max_steps:
            sampler.set_epoch(state.epoch)
            for batch in loader:
                if state.step >= cfg.max_steps:
                    break

                mean = batch["latent_mean"].to(device=device, dtype=dtype, non_blocking=True)
                std = batch["latent_std"].to(device=device, dtype=dtype, non_blocking=True)
                noise = torch.randn(
                    mean.shape, device=device, dtype=dtype, generator=generator
                )
                latents = mean + std * noise  # resample the posterior every epoch

                embeddings = [
                    {
                        # Masks ride along with the embeddings and keep their own dtype.
                        name: tensor.to(device=device, non_blocking=True).to(
                            dtype if tensor.is_floating_point() else tensor.dtype
                        )
                        for name, tensor in text.items()
                    }
                    for text in batch["embeddings"]
                ]
                sigmas = sample_sigmas(
                    latents.shape[0],
                    cfg.timestep_sampling,
                    cfg.flow_shift,
                    cfg.weighting_scheme,
                    cfg.logit_mean,
                    cfg.logit_std,
                    cfg.mode_scale,
                    device,
                    generator,
                )

                loss = family.loss(
                    transformer, latents, embeddings, sigmas, generator, cfg.weighting_scheme
                )
                (loss / cfg.gradient_accumulation_steps).backward()

                if (state.step + 1) % cfg.gradient_accumulation_steps == 0:
                    if cfg.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                value = loss.detach().float().item()
                if not (value == value):  # NaN
                    raise RuntimeError(
                        f"loss became NaN at step {state.step} on {batch['names']}"
                    )
                state.loss_ema = (
                    value if state.loss_ema is None else 0.98 * state.loss_ema + 0.02 * value
                )
                state.step += 1
                state.peak_vram = torch.cuda.max_memory_allocated(device) if on_gpu else 0
                if on_step:
                    on_step(state, value)

                if cfg.save_every and state.step % cfg.save_every == 0:
                    weights_path, state_path = checkpoint_paths(
                        paths["checkpoints"], cfg.name, state.step
                    )
                    save_lora(transformer, weights_path, cfg, {"step": state.step})
                    save_state(
                        state_path,
                        state.step,
                        state.epoch,
                        optimizer,
                        scheduler,
                        generator,
                        state.loss_ema,
                    )
                    run_layout.write_manifest(paths["root"], step=state.step, state="running")
                    if on_save:
                        on_save(weights_path, state.step)

                if (
                    cfg.sample_every
                    and cfg.sample_prompts
                    and state.step % cfg.sample_every == 0
                ):
                    _spawn_sample(prepared, transformer, paths, state, samplers, device)
            state.epoch += 1

        final = paths["root"] / f"{cfg.name}.safetensors"
        save_lora(
            transformer,
            final,
            cfg,
            {
                "step": state.step,
                "loss": f"{state.loss_ema:.5f}" if state.loss_ema else "",
                "peak_vram_gib": f"{state.peak_vram / GIB:.2f}",
            },
        )
        run_layout.write_manifest(
            paths["root"],
            step=state.step,
            state="done",
            final=str(final),
            peak_vram_gib=round(state.peak_vram / GIB, 2),
            seconds=round(state.elapsed, 1),
        )
        if on_save:
            on_save(final, state.step)
    except BaseException:
        run_layout.write_manifest(paths["root"], step=state.step, state="failed")
        raise
    finally:
        if swapper:
            swapper.close()
        for handle in samplers:
            if handle.poll() is None:
                handle.wait(timeout=600)
    return state


def _spawn_sample(
    prepared: Prepared,
    transformer,
    paths: dict[str, Path],
    state,
    samplers: list,
    training_device: torch.device,
):
    """Render validation images from the current weights, off the training card."""
    from ..sample import spawn_sampler

    cfg = prepared.cfg

    samplers[:] = [h for h in samplers if h.poll() is None]
    if samplers:
        return  # a previous batch is still rendering; skip rather than pile up

    device_index = cfg.sample_gpu
    if device_index is None and training_device.type == "cuda" and torch.cuda.device_count() > 1:
        # The sampler needs the text encoder, the transformer and the VAE at once -
        # roughly 20 GiB of loads. Put it on a card the trainer is not occupying.
        device_index = 1 - (training_device.index or 0)

    preview = paths["samples"] / f"step-{state.step:06d}.safetensors"
    save_lora(transformer, preview, cfg, {"step": state.step})
    handle = spawn_sampler(
        prepared,
        preview,
        state.step,
        paths["samples"] / f"step-{state.step:06d}",
        device_index=device_index,
    )
    if handle is not None:
        samplers.append(handle)
