"""Where each stage of the pipeline runs.

Measured on an M4 Pro (USR 2.0 Large, beam 1, 100 Clips of the LRS3 test split):

    hybrid (encoder MPS / decoder CPU)   RTF 0.099
    mps    (everything on GPU)           RTF 0.128
    cpu    (everything on CPU)           RTF 0.232

all at identical WER. The split wins because the encoder is dense 3D
convolution — what a GPU is for — while beam search is hundreds of tiny
sequential steps, where per-kernel launch overhead on MPS outweighs any
throughput gain. Transfer cost between the two is ~0.004 RTF.

See VoxLens issue #7.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["DevicePlan", "available_devices", "resolve_device"]

DEFAULT_DEVICE = "hybrid"


@dataclass(frozen=True)
class DevicePlan:
    """Which torch device each stage of the pipeline runs on."""

    name: str
    encoder: torch.device
    decoder: torch.device

    @property
    def needs_transfer(self) -> bool:
        """True when encoder output must move devices before decoding."""
        return self.encoder.type != self.decoder.type


def available_devices() -> tuple[str, ...]:
    """Device names usable on this machine, best first."""
    if torch.backends.mps.is_available():
        return ("hybrid", "mps", "cpu")
    return ("cpu",)


def resolve_device(name: str) -> DevicePlan:
    """Turn a --device name into a concrete plan.

    Raises ValueError for an unknown name, and RuntimeError for a name that
    is known but unusable on this machine.
    """
    cpu = torch.device("cpu")

    if name not in ("hybrid", "mps", "cpu"):
        raise ValueError(
            f"Unknown device {name!r}. Choose from: hybrid, mps, cpu."
        )

    if name == "cpu":
        return DevicePlan(name="cpu", encoder=cpu, decoder=cpu)

    if not torch.backends.mps.is_available():
        reason = (
            "PyTorch reports no MPS backend. On a Mac migrated from Intel "
            "hardware this usually means an x86_64 Python — see docs/setup.md."
            if not torch.backends.mps.is_built()
            else "This machine has no Apple Silicon GPU."
        )
        raise RuntimeError(f"--device {name} is unavailable. {reason} Use --device cpu.")

    mps = torch.device("mps")
    if name == "mps":
        return DevicePlan(name="mps", encoder=mps, decoder=mps)
    return DevicePlan(name="hybrid", encoder=mps, decoder=cpu)
