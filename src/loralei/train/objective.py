"""Which noise levels a step trains on.

The objective itself - what the transformer is called with and what its output means -
is per model family and lives in `loralei.families`. What is shared is the choice of
sigma: every supported model is a rectified flow, sampled the same ways.
"""

from __future__ import annotations

import torch
from diffusers.training_utils import compute_density_for_timestep_sampling


def apply_flow_shift(u: torch.Tensor, shift: float) -> torch.Tensor:
    """Flux-style shift: push sampling density towards higher noise levels."""
    if shift == 1.0:
        return u
    return (shift * u) / (1.0 + (shift - 1.0) * u)


def sample_sigmas(
    batch_size: int,
    timestep_sampling: str,
    flow_shift: float,
    weighting_scheme: str,
    logit_mean: float,
    logit_std: float,
    mode_scale: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one noise level per sample, in [0, 1]."""
    if timestep_sampling == "logit_normal":
        u = compute_density_for_timestep_sampling(
            "logit_normal",
            batch_size,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
            device=device,
            generator=generator,
        )
    elif timestep_sampling == "uniform":
        u = torch.rand(batch_size, device=device, generator=generator)
    elif timestep_sampling == "shift":
        u = torch.rand(batch_size, device=device, generator=generator)
        u = apply_flow_shift(u, flow_shift)
    else:
        raise ValueError(f"unknown timestep_sampling {timestep_sampling!r}")
    # Keep away from the exact endpoints: sigma=0 has no noise to predict and
    # sigma=1 has no signal.
    return u.clamp(1e-4, 1.0 - 1e-4)
