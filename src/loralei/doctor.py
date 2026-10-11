"""The half of `loralei doctor` that needs torch.

Driver access, the allocator and MIOpen settings, disk and host RAM are checked by the
C# side, which also assembles the report. What is left here is what only torch can
answer: is this a ROCm build, which devices does it see, and do the three things a
training run leans on hardest - flash SDPA, bf16 matmul, 8-bit Adam - actually work.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

GIB = 1024**3

Status = Literal["ok", "warn", "fail", "skip"]


@dataclass
class Check:
    name: str
    status: Status
    detail: str = ""
    remedy: str = ""
    data: dict = field(default_factory=dict)


def check_torch() -> Check:
    try:
        import torch
    except ImportError:
        return Check(
            "torch",
            "fail",
            "torch is not importable",
            "uv pip install --index-url https://download.pytorch.org/whl/rocm7.1 "
            "torch==2.13.0+rocm7.1 torchvision==0.28.0+rocm7.1",
        )

    hip = getattr(torch.version, "hip", None)
    if hip is None:
        return Check(
            "torch",
            "fail",
            f"torch {torch.__version__} is a CUDA/CPU build, not ROCm",
            "reinstall from the ROCm wheel index (see README); this machine has no CUDA device",
        )
    status: Status = "ok"
    remedy = ""
    if "dev" in torch.__version__ or "git" in torch.__version__:
        status = "warn"
        remedy = (
            "nightly builds have been reported to fault during block swap; "
            "prefer the pinned torch==2.13.0+rocm7.1"
        )
    return Check(
        "torch",
        status,
        f"torch {torch.__version__} (HIP {hip})",
        remedy,
        {"torch": torch.__version__, "hip": hip},
    )


def check_devices(agents: list[str], apu_targets: frozenset[str]) -> Check:
    from .hw import probe_devices

    devices = probe_devices()
    if not devices:
        return Check(
            "devices",
            "fail",
            f"torch sees no device (ROCm agents on this host: {agents or 'none'})",
            "check HIP_VISIBLE_DEVICES and driver access",
        )

    lines = [
        f"[{d.index}] {d.name} {d.arch} {d.total_gib:.1f} GiB "
        f"({d.free_gib:.1f} GiB free, {d.multi_processor_count} CU)"
        for d in devices
    ]
    apus = [d for d in devices if d.arch in apu_targets]
    status: Status = "ok"
    remedy = ""
    if apus:
        status = "warn"
        remedy = (
            f"integrated GPU(s) {[d.arch for d in apus]} are visible to torch; "
            "training that lands there will OOM against shared system memory. "
            "loralei excludes them automatically - only an explicit HIP_VISIBLE_DEVICES "
            "in your shell can reintroduce them."
        )
    return Check(
        "devices",
        status,
        "; ".join(lines),
        remedy,
        {
            "visible": os.environ.get("HIP_VISIBLE_DEVICES", "<unset>"),
            "apu_targets": sorted(apu_targets & {d.arch for d in devices}),
            "devices": [
                {"index": d.index, "arch": d.arch, "name": d.name, "total_gib": round(d.total_gib, 1)}
                for d in devices
            ],
        },
    )


def check_sdpa_backend(quick: bool = True) -> Check:
    """The single most consequential check on RDNA4.

    gfx1200 is on AOTriton's *experimental* list, so both the flash and the
    mem-efficient SDPA backends refuse to run unless
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 was set before torch was imported.
    Measured here on a RX 9060 XT (B=1 H=30 S=4608 D=128, bf16, fwd+bwd):
    flash 49.7 ms / 0.40 GiB vs math 619 ms / 10.07 GiB.
    """
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available():
        return Check("SDPA backend", "skip", "no device")

    flag = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "<unset>")
    b, h, s, d = 1, 8 if quick else 30, 4096, 128
    iters = 3 if quick else 5

    def run(backend) -> tuple[float, float] | None:
        q, k, v = (
            torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        )
        torch.cuda.reset_peak_memory_stats()
        try:
            with sdpa_kernel([backend]):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                out.sum().backward()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                    out.sum().backward()
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / iters
        except RuntimeError:
            return None
        return dt, torch.cuda.max_memory_allocated() / GIB

    flash = run(SDPBackend.FLASH_ATTENTION)
    if flash is None:
        math = run(SDPBackend.MATH)
        detail = f"flash/mem-efficient SDPA unavailable (TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL={flag})"
        if math:
            detail += f"; math fallback costs {math[0] * 1000:.0f} ms / {math[1]:.2f} GiB per step"
        return Check(
            "SDPA backend",
            "fail",
            detail,
            "export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 *before* importing torch - "
            "loralei sets it on the worker's environment, so seeing this means it was "
            "overridden in your shell",
        )

    math = run(SDPBackend.MATH)
    detail = f"flash available: {flash[0] * 1000:.1f} ms / {flash[1]:.2f} GiB"
    data = {"flash_ms": round(flash[0] * 1000, 1), "flash_gib": round(flash[1], 2)}
    if math:
        detail += f" (math fallback would be {math[0] * 1000:.0f} ms / {math[1]:.2f} GiB)"
        data |= {
            "math_ms": round(math[0] * 1000, 1),
            "math_gib": round(math[1], 2),
            "speedup": round(math[0] / flash[0], 1),
        }
    return Check("SDPA backend", "ok", detail, "", data)


def check_bf16() -> Check:
    import torch

    if not torch.cuda.is_available():
        return Check("bf16 matmul", "skip", "no device")
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    out = (a @ a).float()
    if not torch.isfinite(out).all():
        return Check("bf16 matmul", "fail", "non-finite values in a plain bf16 matmul", "")
    return Check("bf16 matmul", "ok", "2048^2 bf16 matmul is finite")


def check_optimizer_8bit(steps: int = 200) -> Check:
    """Torture-test bitsandbytes' blockwise 8-bit Adam.

    This exercises bnb's own kernels, not the hipBLASLt 4-bit GEMM path that is
    known to VM-fault on gfx1200 - but if 8-bit Adam is broken here, we want to
    find out in 10 seconds rather than 6 hours into a run.
    """
    import torch

    if not torch.cuda.is_available():
        return Check("8-bit Adam", "skip", "no device")
    try:
        import bitsandbytes as bnb
    except ImportError:
        return Check(
            "8-bit Adam",
            "warn",
            "bitsandbytes not installed",
            "uv pip install bitsandbytes   # or use --optimizer adamw / adafactor",
        )

    param = torch.nn.Parameter(torch.randn(4096, 1024, device="cuda", dtype=torch.float32) * 0.01)
    opt = bnb.optim.AdamW8bit([param], lr=1e-4)
    try:
        for _ in range(steps):
            loss = (param * param).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        return Check(
            "8-bit Adam",
            "fail",
            f"bitsandbytes AdamW8bit failed after warmup: {exc}"[:300],
            "run with --optimizer adamw (or adafactor) instead",
        )
    if not torch.isfinite(param).all():
        return Check(
            "8-bit Adam",
            "fail",
            "parameters went non-finite during the 8-bit Adam torture test",
            "run with --optimizer adamw (or adafactor) instead",
        )
    return Check(
        "8-bit Adam",
        "ok",
        f"bitsandbytes {bnb.__version__}: {steps} steps on 4M params, parameters finite",
    )


def run_all(
    quick: bool,
    agents: list[str],
    apu_targets: frozenset[str],
    report: Callable[[Check], None],
) -> None:
    """Run every torch check, reporting each as it finishes."""
    torch_check = check_torch()
    report(torch_check)
    if torch_check.status == "fail":
        return

    checks: list[tuple[str, Callable[[], Check]]] = [
        ("devices", lambda: check_devices(agents, apu_targets)),
        ("SDPA backend", lambda: check_sdpa_backend(quick=quick)),
        ("bf16 matmul", check_bf16),
        ("8-bit Adam", lambda: check_optimizer_8bit(200 if quick else 512)),
    ]
    for name, fn in checks:
        try:
            report(fn())
        except Exception as exc:  # a broken check must not hide the other results
            report(Check(name, "fail", repr(exc)[:300]))
