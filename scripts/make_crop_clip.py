#!/usr/bin/env python
"""Produce the pre-cropped Mouth Region clip the transport tests want.

`VOXLENS_CROP_CLIP` has to be 96x96 Mouth Regions rather than whole Frames —
what evaluation corpora ship, and what a phone would send. Corpora are gated
or no longer distributed (ADR-0005), so this makes one from what you have.

    make_crop_clip.py face.mp4 --out mouth96.mp4     extraction, a real mouth
    make_crop_clip.py --synthetic 50 --out plumb.mp4 no source video needed

The synthetic form exists for testing plumbing, and only plumbing: it carries
no mouth, so any Transcript from it is meaningless. What it does establish is
that two routes into the recogniser agree on identical input, which is exactly
what the crop transport has to prove (ADR-0009) and needs no speech to show.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

CROP = 96
FPS = 25


def synthetic(frames: int) -> np.ndarray:
    """Temporally smooth texture: deterministic, and not a uniform block.

    Smooth in time because a run of identical Frames is the one input a
    per-Frame pipeline can mishandle without anyone noticing, and deterministic
    so a disagreement between two routes is never the fixture's fault.
    """
    grid_y, grid_x = np.meshgrid(np.arange(CROP), np.arange(CROP), indexing="ij")
    clip = np.empty((frames, CROP, CROP, 3), dtype=np.uint8)
    for index in range(frames):
        phase = index / max(frames - 1, 1) * 2 * np.pi
        field = (
            np.sin(grid_x / 7.0 + phase) + np.cos(grid_y / 11.0 - phase)
        ) / 2.0  # -1..1
        value = ((field + 1) * 127.5).astype(np.uint8)
        clip[index, :, :, 0] = value
        clip[index, :, :, 1] = np.roll(value, 3, axis=0)
        clip[index, :, :, 2] = np.roll(value, 6, axis=1)
    return clip


def extracted(path: Path) -> np.ndarray:
    """Real Mouth Regions, through the same extraction the CLI runs."""
    from voxlens.extraction import MouthRegionError, extract_mouth_regions
    from voxlens.video import decode_clip

    clip = decode_clip(path)
    try:
        regions = extract_mouth_regions(clip.frames)
    except MouthRegionError as exc:
        raise SystemExit(f"make_crop_clip: {path}: {exc}")
    return regions.crops


def write(crops: np.ndarray, out: Path) -> None:
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (CROP, CROP))
    if not writer.isOpened():
        raise SystemExit(f"make_crop_clip: could not open {out} for writing")
    try:
        for frame in crops:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", nargs="?", help="a video with a visible face")
    parser.add_argument(
        "--synthetic",
        type=int,
        metavar="FRAMES",
        help="skip extraction and emit texture instead: plumbing only, no mouth",
    )
    parser.add_argument("--out", required=True, help="where to write the crop clip")
    args = parser.parse_args(argv)

    if bool(args.video) == bool(args.synthetic):
        parser.error("give either a video or --synthetic FRAMES, not both or neither")

    crops = synthetic(args.synthetic) if args.synthetic else extracted(Path(args.video))
    write(crops, Path(args.out))
    print(
        f"wrote {args.out}: {len(crops)} frames of {CROP}x{CROP} at {FPS} fps"
        + ("  (synthetic — Transcripts from it mean nothing)" if args.synthetic else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
