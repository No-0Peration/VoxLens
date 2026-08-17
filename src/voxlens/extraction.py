"""Frames to Mouth Regions.

Produces the crop contract the recogniser expects — 96x96 RGB at 25 fps,
warped onto a shared mean face — and, alongside it, the per-Frame record of
where no face was found.

That record is the point. Upstream interpolates across undetected Frames and
then asserts every Frame has landmarks, so by the time anything reaches the
encoder the gap is invisible. Occlusion can only be reported if the mask is
captured here, before that happens (ADR-0008).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voxlens.upstream import ensure_importable

__all__ = ["ExtractedMouthRegions", "MouthRegionError", "extract_mouth_regions"]

DETECTOR = "mediapipe"


class MouthRegionError(RuntimeError):
    """The Speaker's mouth could not be read in enough Frames."""


@dataclass(frozen=True)
class ExtractedMouthRegions:
    """Mouth Region crops, plus which Frames had no detectable face."""

    crops: np.ndarray  # (T, 96, 96, 3) uint8 RGB
    undetected: tuple[bool, ...]  # per Frame: True == no face found

    @property
    def undetected_count(self) -> int:
        return sum(self.undetected)


def extract_mouth_regions(frames: np.ndarray) -> ExtractedMouthRegions:
    """Detect the Speaker's face and crop the Mouth Region from every Frame.

    Raises MouthRegionError when too few Frames yield a face for the upstream
    cropper to produce anything.
    """
    ensure_importable()
    from preprocessing.landmarks_detector import LandmarksDetector
    from preprocessing.video_preprocess import VideoProcess

    detector = LandmarksDetector(detector=DETECTOR)
    try:
        landmarks = detector(frames)
    finally:
        detector.close()

    undetected = tuple(mark is None for mark in landmarks)

    # convert_gray=False: crops are stored in colour. The recogniser converts
    # to greyscale itself during normalisation.
    crops = VideoProcess(convert_gray=False)(frames, landmarks)
    if crops is None:
        raise MouthRegionError(
            "Could not read the Speaker's mouth in enough Frames. "
            "The video needs a clearly visible face."
        )

    return ExtractedMouthRegions(crops=crops, undetected=undetected)
