"""Hardware probing that needs torch."""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3


@dataclass
class Device:
    index: int
    """Index as torch sees it (i.e. after HIP_VISIBLE_DEVICES filtering)."""
    name: str
    arch: str
    total_bytes: int
    free_bytes: int
    multi_processor_count: int

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB

    @property
    def free_gib(self) -> float:
        return self.free_bytes / GIB

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "arch": self.arch,
            "total_gib": round(self.total_gib, 2),
            "free_gib": round(self.free_gib, 2),
        }


def _bare_arch(gcn_arch_name: str) -> str:
    """'gfx1200:xnack-' -> 'gfx1200'."""
    return gcn_arch_name.split(":", 1)[0]


def probe_devices(wake: bool = True) -> list[Device]:
    """Enumerate visible devices.

    The cards sit in a runtime-PM low-power state when idle and report nonsense for
    free memory until touched, so `wake` allocates a byte on each first.
    """
    import torch

    if not torch.cuda.is_available():
        return []

    devices: list[Device] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        if wake:
            try:
                torch.empty(1, device=f"cuda:{i}")
            except RuntimeError:
                pass
        try:
            free, total = torch.cuda.mem_get_info(i)
        except RuntimeError:
            free, total = 0, props.total_memory
        devices.append(
            Device(
                index=i,
                name=props.name,
                arch=_bare_arch(getattr(props, "gcnArchName", "")),
                total_bytes=total,
                free_bytes=free,
                multi_processor_count=props.multi_processor_count,
            )
        )
    return devices


def pick_training_device(
    devices: list[Device], requested: int | None = None, apu_targets: frozenset[str] = frozenset()
) -> Device:
    """Choose the card to train on; refuse the iGPU unless explicitly asked."""
    if not devices:
        raise RuntimeError(
            "no ROCm device visible to torch - run `loralei doctor` to find out why"
        )
    if requested is not None:
        if requested >= len(devices):
            raise IndexError(
                f"--gpu {requested} requested but torch sees only {len(devices)} device(s)"
            )
        return devices[requested]
    usable = [d for d in devices if d.arch not in apu_targets] or devices
    return max(usable, key=lambda d: d.free_bytes)
