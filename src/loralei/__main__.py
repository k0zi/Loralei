"""The loralei worker: the part of loralei that needs torch.

This is not a command line for people. The `loralei` CLI (C#) reads the config, finds
the model, scans the dataset and plans the run, then starts this process with one job:

    python -m loralei -            # the job as JSON on stdin
    python -m loralei job.json     # or from a file (how the trainer starts a sampler)

Progress is reported as one JSON object per line on stdout. stderr is left to torch,
diffusers and tqdm. The environment - including TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL,
which must be set before torch is imported - is prepared by whoever starts us.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import traceback
from pathlib import Path

GIB = 1024**3


def emit(event: str, **fields) -> None:
    sys.stdout.write(json.dumps({"event": event, **fields}, default=str) + "\n")
    sys.stdout.flush()


def _interrupt_once(signum, frame) -> None:
    """Raise KeyboardInterrupt for the first SIGINT only.

    From a terminal, Ctrl-C reaches the whole process group and the parent forwards it
    as well. A second interrupt landing inside the handler that stamps the manifest and
    releases the block swapper would undo exactly the cleanup the first one asked for.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise KeyboardInterrupt


def _apu_targets(job: dict) -> frozenset[str]:
    return frozenset(job.get("apu_targets") or ())


def _pick_device(job: dict, requested: int | None) -> str:
    """The torch device a job runs on, announced before any model is loaded.

    `LORALEI_DEVICE=cpu` is for smoke tests with a tiny model on a machine without a
    ROCm card. It is not a way to train: nothing here is tuned for it.
    """
    if os.environ.get("LORALEI_DEVICE") == "cpu":
        emit("device", index=-1, name="CPU (smoke test)", arch="cpu", total_gib=0.0, free_gib=0.0)
        return "cpu"

    from .hw import pick_training_device, probe_devices

    device = pick_training_device(probe_devices(), requested, _apu_targets(job))
    emit("device", **device.as_dict())
    return f"cuda:{device.index}"


def _dtype(name: str | None):
    import torch

    # bf16 is a GPU format here; the CPU smoke-test path runs everything in fp32.
    if name == "fp32" or os.environ.get("LORALEI_DEVICE") == "cpu":
        return torch.float32
    return torch.bfloat16


def _run_caches(prepared, device: str, force: bool) -> None:
    from .cache.latents import VAE_DTYPE, cache_latents
    from .cache.text import cache_text_embeddings
    from .families import family_of

    cfg = prepared.cfg
    family = family_of(prepared.spec)
    latents = cache_latents(
        family,
        items=prepared.items,
        plan=prepared.plan,
        layout=prepared.layout,
        vae_path=prepared.components.path("vae"),
        device="cpu" if cfg.vae_device == "cpu" else device,
        dtype=VAE_DTYPE,  # bf16 VAE encode is numerically broken on gfx1200
        force=force,
        progress=lambda done, total: emit("progress", task="latents", done=done, total=total),
    )
    text = cache_text_embeddings(
        family,
        captions=[item.caption for item in prepared.items],
        layout=prepared.layout,
        components=prepared.components,
        # A text encoder too large for the card runs where its family says it must.
        device=os.environ.get("LORALEI_TEXT_DEVICE") or family.text_device or device,
        dtype=_dtype(cfg.dtype),
        force=force,
        progress=lambda done, total: emit("progress", task="text", done=done, total=total),
    )
    emit(
        "cached",
        latents={"written": latents.written, "reused": latents.reused,
                 "cpu_fallbacks": latents.cpu_fallbacks},
        text={"written": text.written, "reused": text.reused},
    )


def job_probe(job: dict) -> None:
    _pick_device(job, job.get("gpu"))


def job_doctor(job: dict) -> None:
    from .doctor import run_all

    def report(check) -> None:
        emit("check", name=check.name, status=check.status, detail=check.detail,
             remedy=check.remedy, **({"data": check.data} if check.data else {}))

    run_all(bool(job.get("quick", True)), list(job.get("agents") or ()), _apu_targets(job), report)


def job_cache(job: dict) -> None:
    from .job import Prepared

    prepared = Prepared.of(job)
    device = _pick_device(job, prepared.cfg.gpu)
    _run_caches(prepared, device, bool(job.get("force")))


def job_train(job: dict) -> None:
    import torch

    from .job import Prepared
    from .train.loop import train

    prepared = Prepared.of(job)
    cfg = prepared.cfg
    device = _pick_device(job, cfg.gpu)
    if job.get("cache_first"):
        _run_caches(prepared, device, force=False)

    def on_step(state, value: float) -> None:
        emit(
            "step",
            step=state.step,
            epoch=state.epoch,
            loss=round(state.loss_ema or value, 6),
            vram_gib=round(state.peak_vram / GIB, 3),
        )

    state = train(
        prepared,
        torch.device(device),
        on_step=on_step,
        on_save=lambda path, step: emit("save", step=step, path=str(path)),
        resume=bool(job.get("resume")),
    )
    emit(
        "done",
        step=state.step,
        loss=state.loss_ema,
        vram_gib=round(state.peak_vram / GIB, 3),
        seconds=round(state.elapsed, 1),
        final=str(cfg.run_dir / f"{cfg.name}.safetensors"),
    )


def job_sample(job: dict) -> None:
    import torch

    from .job import Components, Spec
    from .sample import generate

    spec = Spec.of(job["spec"])
    device = _pick_device(job, job.get("gpu"))
    result = generate(
        spec=spec,
        components=Components(spec.key, job["components"]),
        prompts=list(job["prompts"]),
        output_dir=Path(job["out"]),
        lora=Path(job["lora"]) if job.get("lora") else None,
        steps=int(job["steps"]),
        guidance_scale=float(job["cfg"]),
        width=int(job["width"]),
        height=int(job["height"]),
        seed=int(job["seed"]),
        device=torch.device(device),
        on_image=lambda path: emit("image", path=str(path)),
        dtype=_dtype(job.get("dtype")),
    )
    if result.verify is not None:
        emit("verify", ok=result.verify.ok, summary=result.verify.summary())
        if not result.verify.ok:
            raise SystemExit(1)


def job_verify(job: dict) -> None:
    """Load an adapter back into the real transformer and check every key landed."""
    import diffusers
    import torch

    from .job import Components, Spec
    from .lora.verify import verify_against_transformer
    from .train.loop import load_transformer

    spec = Spec.of(job["spec"])
    components = Components(spec.key, job["components"])
    lora = Path(job["lora"])

    transformer = load_transformer(spec, components.path("transformer"), _dtype(None))
    pipeline_cls = getattr(diffusers, spec.pipeline_class)
    state_dict = pipeline_cls.lora_state_dict(str(lora))
    pipeline_cls.load_lora_into_transformer(state_dict, transformer, adapter_name="loralei")
    result = verify_against_transformer(lora, transformer, "loralei")
    emit("verify", ok=result.ok, summary=result.summary())
    if not result.ok:
        raise SystemExit(1)


JOBS = {
    "probe": job_probe,
    "doctor": job_doctor,
    "cache": job_cache,
    "train": job_train,
    "sample": job_sample,
    "verify": job_verify,
}


def main() -> None:
    signal.signal(signal.SIGINT, _interrupt_once)

    if len(sys.argv) != 2:
        sys.stderr.write(
            "this is loralei's torch worker, started by the `loralei` command.\n"
            "usage: python -m loralei <job.json | ->\n"
        )
        raise SystemExit(2)

    source = sys.argv[1]
    job = json.loads(sys.stdin.read() if source == "-" else Path(source).read_text())
    try:
        JOBS[job["job"]](job)
    except KeyboardInterrupt:
        emit("error", message="interrupted")
        raise SystemExit(130) from None
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        emit("error", message=f"{type(exc).__name__}: {exc}"[:2000])
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
