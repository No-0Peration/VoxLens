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
from voxlens.extraction import PreCroppedRegions
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
    parser.add_argument(
        "video",
        nargs="+",
        help="one or more video files, each holding one Clip. With several, the "
        "checkpoint is loaded once and --json emits one JSON object per line "
        "(JSON Lines) — which is what makes evaluating a corpus viable.",
    )
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
        "--pre-cropped",
        action="store_true",
        help="the input Frames are ALREADY Mouth Region crops, so skip face "
        "detection. Every obtainable evaluation corpus ships this way. No "
        "Occlusion can be reported in this mode — there is no face to lose.",
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

    from voxlens.recogniser import CheckpointError, load_recogniser

    # Loaded once, on the first Clip that actually needs it. Eagerly would make
    # a broken checkpoint preempt a missing file, so `voxlens missing.mp4` would
    # complain about the wrong thing; lazily would reload 4 GB per Clip.
    loaded: list = []

    def recogniser_for_use():
        if not loaded:
            loaded.append(load_recogniser(args.checkpoint, plan, beam=args.beam))
        return loaded[0]

    checkpoint = checkpoint_identity(args.checkpoint)
    worst = EXIT_OK
    many = len(args.video) > 1

    for path in args.video:
        try:
            clip = decode_clip(path)
        except UnreadableClipError as exc:
            print(f"voxlens: {exc}", file=sys.stderr)
            worst = max(worst, EXIT_BAD_INPUT)
            continue

        started = time.perf_counter()
        if args.pre_cropped:
            # The Frames are the Mouth Region already. Running a face detector
            # over a mouth crop finds nothing — or worse, finds something and
            # crops a "face" out of a mouth, silently producing garbage.
            regions = PreCroppedRegions(clip.frames)
        else:
            try:
                regions = extract_mouth_regions(clip.frames)
            except MouthRegionError as exc:
                print(f"voxlens: {path}: {exc}", file=sys.stderr)
                worst = max(worst, EXIT_UNREADABLE)
                continue
        extract_s = time.perf_counter() - started

        try:
            recogniser = recogniser_for_use()
        except CheckpointError as exc:
            # Only the known checkpoint failures map to an exit code; a genuine
            # bug must still surface as a traceback, not a tidy message.
            print(f"voxlens: {exc}", file=sys.stderr)
            return EXIT_MODEL

        started = time.perf_counter()
        transcript = recogniser.transcribe(regions.crops)
        infer_s = time.perf_counter() - started

        result = Result(
            video=path,
            frames=clip.frame_count,
            fps=clip.fps,
            duration_s=clip.duration_s,
            transcript=transcript,
            undetected_frames=regions.undetected_count,
            timing=Timing(extract_s=extract_s, infer_s=infer_s, duration_s=clip.duration_s),
            device=plan.name,
            beam=args.beam,
            occlusion_min_frames=args.occlusion_min_frames,
            occlusions=tuple(
                find_occlusions(regions.undetected, min_frames=args.occlusion_min_frames)
            ),
            checkpoint=checkpoint,
        )

        if args.as_json:
            # One object per line when there are several Clips, so a consumer
            # can stream rather than wait for the whole corpus.
            json.dump(result.as_dict(), sys.stdout, indent=None if many else 2)
            sys.stdout.write("\n")
        elif many:
            print(f"{path}\t{result.transcript}")
        else:
            print(result.transcript)

        # Diagnostics never touch stdout, in either mode: --json stays pipeable.
        print(result.summary_line(), file=sys.stderr)
        for line in result.occlusion_lines():
            print(line, file=sys.stderr)

    return worst


if __name__ == "__main__":
    raise SystemExit(main())
