"""The voxlens command line.

This is the seam (ADR-0007): tests and the evaluation harness both go through
it, so what is measured is what a user gets. It orchestrates and renders; it
holds no recognition logic of its own.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from voxlens.devices import DEFAULT_DEVICE, resolve_device
from voxlens.extraction import MouthRegionError, extract_mouth_regions
from voxlens.occlusion import DEFAULT_MIN_FRAMES, find_occlusions
from voxlens.result import Result, Timing, checkpoint_identity
from voxlens.video import UnreadableClipError, decode_clip

__all__ = ["main"]

# A batch run over hundreds of Clips has to tell these apart without parsing
# prose: 1 means "this video is unusable", 2 means "this invocation is wrong".
EXIT_OK = 0
EXIT_UNREADABLE = 1  # the Clip decoded, but the Speaker's mouth could not be read
EXIT_BAD_INPUT = 2  # the invocation is wrong: missing/undecodable Clip, bad device
EXIT_MODEL = 3  # the checkpoint is missing, unreadable, or the wrong architecture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxlens",
        description="Read speech from the visible movement of a Speaker's mouth. "
        "No audio is used.",
        epilog=(
            "exit codes:\n"
            "  0  a Transcript was produced\n"
            "  1  the Clip decoded, but the Speaker's mouth could not be read\n"
            "  2  the invocation is wrong: missing or undecodable Clip, or an\n"
            "     unavailable device\n"
            "  3  the checkpoint is missing, unreadable, or the wrong architecture"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", help="path to a video file holding one Clip")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="path to the USR 2.0 Large checkpoint (never downloaded for you)",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["hybrid", "mps", "cpu"],
        help="hybrid runs the encoder on the GPU and the search on the CPU, "
        "which is the fastest measured configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit machine-readable output on stdout; the evaluation harness "
        "consumes this, so nothing else may reach stdout",
    )
    parser.add_argument(
        "--occlusion-min-frames",
        type=int,
        default=DEFAULT_MIN_FRAMES,
        metavar="N",
        help="consecutive Frames with no detected face before the gap counts "
        "as an Occlusion rather than detector noise; 3 frames is 120ms, about "
        "one viseme (default: %(default)s)",
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=1,
        help="beam width; 1 is greedy and roughly 16x faster than 40 "
        "(default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        plan = resolve_device(args.device)
    except RuntimeError as exc:
        print(f"voxlens: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    try:
        clip = decode_clip(args.video)
    except UnreadableClipError as exc:
        print(f"voxlens: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    started = time.perf_counter()
    try:
        regions = extract_mouth_regions(clip.frames)
    except MouthRegionError as exc:
        print(f"voxlens: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE
    extract_s = time.perf_counter() - started

    # Imported here so that the failures above do not pay for loading torch.
    from voxlens.recogniser import CheckpointError, load_recogniser

    try:
        recogniser = load_recogniser(args.checkpoint, plan, beam=args.beam)
    except CheckpointError as exc:
        # Only the known checkpoint failures map to an exit code; a genuine bug
        # must still surface as a traceback rather than a tidy error message.
        print(f"voxlens: {exc}", file=sys.stderr)
        return EXIT_MODEL

    started = time.perf_counter()
    transcript = recogniser.transcribe(regions.crops)
    infer_s = time.perf_counter() - started

    occlusions = find_occlusions(
        regions.undetected, min_frames=args.occlusion_min_frames
    )

    result = Result(
        video=args.video,
        frames=clip.frame_count,
        fps=clip.fps,
        duration_s=clip.duration_s,
        transcript=transcript,
        undetected_frames=regions.undetected_count,
        timing=Timing(extract_s=extract_s, infer_s=infer_s, duration_s=clip.duration_s),
        device=plan.name,
        beam=args.beam,
        occlusion_min_frames=args.occlusion_min_frames,
        occlusions=tuple(occlusions),
        checkpoint=checkpoint_identity(args.checkpoint),
    )

    if args.as_json:
        json.dump(result.as_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(result.transcript)

    # Diagnostics never touch stdout, in either mode: --json must stay pipeable.
    print(result.summary_line(), file=sys.stderr)
    for line in result.occlusion_lines():
        print(line, file=sys.stderr)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
