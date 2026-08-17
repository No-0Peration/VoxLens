"""Smoke test: does this machine actually run the recogniser?

Requires the vendored upstream (scripts/vendor.py) and a checkpoint. The
checkpoint is ~4 GB, caller-supplied, and never downloaded automatically —
so these tests skip with an actionable message when it is absent.
"""
import os
import platform

import pytest
import torch

from voxlens.devices import resolve_device
from voxlens.upstream import is_vendored, load_encoder, vendored_path

CHECKPOINT = os.environ.get("VOXLENS_CHECKPOINT")

needs_upstream = pytest.mark.skipif(
    not is_vendored(), reason="upstream not vendored — run: uv run python scripts/vendor.py"
)
needs_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT and os.path.exists(CHECKPOINT)),
    reason="set VOXLENS_CHECKPOINT to a USR 2.0 Large .pth to run this",
)


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-only guarantee")
def test_the_interpreter_is_native_arm64():
    """Guards the migrated-Mac failure, where an x86_64 Python arrives silently.

    `docs/setup.md` tells the reader this test will catch a wrong interpreter,
    so it has to check the thing that actually goes wrong: the architecture.
    """
    assert platform.machine() == "arm64", (
        f"This interpreter is {platform.machine()}, not arm64. On a Mac migrated "
        "from Intel hardware, python3 and uv are often still x86_64 builds and "
        "torch ships no x86 macOS wheels above 2.4.1. See docs/setup.md."
    )


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-only guarantee")
def test_the_gpu_is_actually_usable():
    """`is_built()` is a compile-time flag and passes on machines that cannot
    run a single kernel. Availability is the property setup.md promises."""
    assert torch.backends.mps.is_built(), (
        "PyTorch was built without MPS — almost certainly an x86_64 Python. "
        "See docs/setup.md."
    )
    assert torch.backends.mps.is_available(), (
        "PyTorch has MPS support but reports it unavailable on this machine. "
        "See docs/setup.md."
    )


@needs_upstream
def test_vendored_upstream_is_importable():
    assert vendored_path().is_dir()
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E  # noqa: F401


@needs_upstream
def test_ctc_prefix_scorer_honours_its_input_device():
    """Patch 0001. Upstream hardcoded cuda-or-cpu, silently choosing cpu on MPS."""
    import inspect

    from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH

    src = inspect.getsource(CTCPrefixScoreTH.__init__)
    assert "x.device" in src, "patch 0001 is not applied to the vendored tree"


@needs_upstream
@needs_checkpoint
def test_encoder_runs_on_the_hybrid_plan():
    """The end-to-end claim of this ticket: the model loads and encodes here."""
    plan = resolve_device("hybrid" if torch.backends.mps.is_available() else "cpu")
    encoder = load_encoder(CHECKPOINT, plan)

    frames = 40
    dummy = torch.zeros(1, frames, 88, 88, device=plan.encoder)
    with torch.no_grad():
        out = encoder(xs_v=dummy)

    assert out.shape[0] == 1
    assert out.shape[1] == frames, "the encoder must not downsample in time"
    assert out.dim() == 3
