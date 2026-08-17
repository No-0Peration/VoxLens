"""Access to the vendored USR 2.0 recogniser.

The upstream repository is not a package and does not run unpatched on Apple
Silicon, so it is vendored at a pinned revision by ``scripts/vendor.py`` and
patched from ``patches/``. Nothing here downloads anything: both the upstream
tree and the checkpoint are caller-supplied.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from voxlens.devices import DevicePlan

__all__ = ["UPSTREAM_REPO", "UPSTREAM_REV", "is_vendored", "load_encoder", "vendored_path"]

UPSTREAM_REPO = "https://github.com/ahaliassos/usr2.git"
UPSTREAM_REV = "df0c78b7a3807e625a0fcdadd14b1cf674d21c91"

BACKBONE = "resnet_transformer_large"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vendored_path() -> Path:
    return repo_root() / "vendor" / "usr2"


def is_vendored() -> bool:
    path = vendored_path()
    if not (path / "espnet").is_dir():
        return False
    _ensure_importable()
    return True


def _ensure_importable() -> None:
    """Put the vendored tree on sys.path — it is a script tree, not a package."""
    path = str(vendored_path())
    if path not in sys.path:
        sys.path.insert(0, path)


def load_config():
    """Compose the upstream Hydra config for the chosen backbone."""
    _ensure_importable()
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(vendored_path() / "conf"), version_base="1.3"):
        return compose(config_name="config", overrides=[f"model/backbone={BACKBONE}"])


def load_encoder(checkpoint_path: str | Path, plan: DevicePlan):
    """Load the recogniser and return its encoder, placed per ``plan``.

    Only the encoder is returned: this is the smoke-test surface, and keeping
    the encoder separable is a standing constraint (it is the streaming-critical
    component and the natural Core ML conversion unit).
    """
    _ensure_importable()
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
    from utils.utils import UNIGRAM1000_LIST

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. VoxLens never downloads it — "
            "see docs/setup.md for where to obtain USR 2.0 Large."
        )

    cfg = load_config()
    model = E2E(len(UNIGRAM1000_LIST), cfg.model.backbone)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if any(k.startswith("model.backbone.") for k in state):
        state = {
            k.replace("model.backbone.", "", 1): v
            for k, v in state.items()
            if k.startswith("model.backbone.")
        }
    model.load_state_dict(state)
    model.eval()

    encoder = model.encoder.to(plan.encoder)
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder
