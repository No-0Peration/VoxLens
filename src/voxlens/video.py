"""Decoding a Clip into Frames.

Deliberately does not use ``torchvision.io.read_video``: it was removed in
torchvision 0.28, and depending on it is what breaks the upstream demo.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["DecodedClip", "decode_clip"]

TARGET_FPS = 25.0


class UnreadableClipError(RuntimeError):
    """The file is missing, or contains no decodable Frames."""


@dataclass(frozen=True)
class DecodedClip:
    """Frames of a Clip, as RGB uint8, with the rate they were shot at."""

    frames: np.ndarray  # (T, H, W, 3) uint8 RGB
    fps: float

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps


def decode_clip(path: str | Path) -> DecodedClip:
    """Read a Clip into Frames.

    Raises UnreadableClipError if the file is absent or yields no Frames.
    """
    path = Path(path)
    if not path.exists():
        raise UnreadableClipError(f"No such file: {path}")

    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()

    if not frames:
        raise UnreadableClipError(f"No decodable frames in: {path}")

    # A container that reports no rate is assumed to be at the rate the
    # recogniser expects; guessing anything else would silently resample.
    if not fps or fps <= 0:
        fps = TARGET_FPS

    return DecodedClip(frames=np.stack(frames), fps=float(fps))
