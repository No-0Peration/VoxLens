"""Drive a running server from crops on disk, with no phone involved.

The camera side of live capture is an iOS app that does not exist yet (#20).
This is what stands in for it: it reads Mouth Region crops from a file, sends
them to ``voxlens-serve``, and prints what comes back. It is how the transport
is exercised end to end, and how a server is checked after a change without
picking up a phone.

Evaluation corpora ship pre-cropped 96x96 Mouth Regions (ADR-0005), so a
corpus Clip is already exactly what a phone would send. That is deliberate:
the same file can go through ``voxlens --pre-cropped`` and through this, and
the two must agree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from voxlens.cli import EXIT_BAD_INPUT, EXIT_MODEL, EXIT_OK
from voxlens.transport import (
    CROP_SHAPE,
    CROP_SIZE,
    DEFAULT_HOST,
    DEFAULT_PORT,
    FPS_TOLERANCE,
    MAX_CROPS_PER_MESSAGE,
    TARGET_FPS,
    CropClient,
    ProtocolError,
    ServerError,
)
from voxlens.video import UnreadableClipError, decode_clip

__all__ = ["load_crops", "main"]


class CropFileError(RuntimeError):
    """The file is not a sequence of Mouth Region crops."""


def load_crops(path: str | Path) -> tuple[np.ndarray, float | None]:
    """Read crops from a video or a .npy array.

    Returns the crops and the rate the file claims, or None where the file
    carries no rate of its own.
    """
    path = Path(path)
    if path.suffix == ".npy":
        try:
            crops = np.load(path)
        except (OSError, ValueError) as exc:
            raise CropFileError(f"{path}: could not read: {exc}") from exc
        fps = None
    else:
        try:
            clip = decode_clip(path)
        except UnreadableClipError as exc:
            raise CropFileError(str(exc)) from exc
        crops, fps = clip.frames, clip.fps

    crops = np.asarray(crops)
    if crops.ndim != 4 or crops.shape[1:] != CROP_SHAPE:
        raise CropFileError(
            f"{path}: this is {'x'.join(str(n) for n in crops.shape[1:])} per frame, "
            f"and crops must be {CROP_SIZE}x{CROP_SIZE}x3. A file of whole Frames "
            "needs extraction first — that is what the phone does, and what "
            "`voxlens` does with a video."
        )
    if crops.dtype != np.uint8:
        raise CropFileError(f"{path}: crops must be uint8 RGB; this is {crops.dtype}")
    return crops, fps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxlens-replay",
        description="Send Mouth Region crops from disk to a running "
        "voxlens-serve and print the Transcripts. Stands in for the camera.",
        epilog=(
            "exit codes:\n"
            "  0  every batch of crops came back as a Transcript\n"
            "  2  the invocation is wrong: a bad file, or nothing listening\n"
            "  3  the server reported a model failure"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "crops",
        nargs="+",
        help="pre-cropped 96x96 Mouth Region clips (video or .npy). Several are "
        "sent in one session, which is what proves the checkpoint is not "
        "reloaded per request.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="(default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="(default: %(default)s)")
    parser.add_argument(
        "--session",
        default="",
        help="session identity to declare; a random one is used when omitted",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=TARGET_FPS,
        help="the rate to declare for this session (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        metavar="N",
        help="split each file into batches of N frames, one Transcript each. "
        "The default sends a whole file as one batch, which is what matches "
        "the CLI. Chunks are decoded independently and NOT reconciled — "
        "overlapping windows with a revision boundary are #21 (ADR-0012).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the server's reply per batch as JSON Lines on stdout",
    )
    return parser


def _batches(crops: np.ndarray, chunk: int):
    if chunk <= 0:
        yield crops
        return
    for start in range(0, len(crops), chunk):
        yield crops[start : start + chunk]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.chunk and not 0 < args.chunk <= MAX_CROPS_PER_MESSAGE:
        print(
            f"voxlens-replay: --chunk must be 1..{MAX_CROPS_PER_MESSAGE}",
            file=sys.stderr,
        )
        return EXIT_BAD_INPUT

    # Loaded before connecting: a typo in a path should not open a session and
    # then abandon it.
    loaded = []
    for path in args.crops:
        try:
            crops, fps = load_crops(path)
        except CropFileError as exc:
            print(f"voxlens-replay: {exc}", file=sys.stderr)
            return EXIT_BAD_INPUT
        if fps is not None and abs(fps - args.fps) > FPS_TOLERANCE:
            print(
                f"voxlens-replay: {path} is {fps:g} fps but this session declares "
                f"{args.fps:g}. The checkpoint expects {TARGET_FPS:g}; resample the "
                "file rather than mislabelling it.",
                file=sys.stderr,
            )
            return EXIT_BAD_INPUT
        loaded.append((path, crops))

    client = CropClient(host=args.host, port=args.port, session=args.session, fps=args.fps)
    many = len(loaded) > 1 or args.chunk

    try:
        ready = client.open()
    except OSError as exc:
        print(
            f"voxlens-replay: nothing answered at {args.host}:{args.port} ({exc}). "
            "Start it with: voxlens-serve --checkpoint PATH",
            file=sys.stderr,
        )
        return EXIT_BAD_INPUT

    config = ready.get("config", {})
    print(
        f"voxlens-replay: session {client.session} on {args.host}:{args.port}, "
        f"device {config.get('device', '?')}, beam {config.get('beam', '?')}",
        file=sys.stderr,
    )

    try:
        for path, crops in loaded:
            for batch in _batches(crops, args.chunk):
                reply = client.send_crops(batch)
                if args.as_json:
                    json.dump({"crops": str(path), **reply}, sys.stdout, indent=None)
                    sys.stdout.write("\n")
                elif many:
                    print(f"{path}\t{reply['transcript']}")
                else:
                    print(reply["transcript"])
                sys.stdout.flush()
                timing = reply.get("timing", {})
                print(
                    f"voxlens-replay: {reply['frames']} frames, "
                    f"{reply['duration_s']}s, infer {timing.get('infer_s', '?')}s, "
                    f"RTF {timing.get('rtf', '?')}",
                    file=sys.stderr,
                )
    except ServerError as exc:
        print(f"voxlens-replay: the server refused: {exc}", file=sys.stderr)
        return EXIT_MODEL if exc.code == "model" else EXIT_BAD_INPUT
    except (ProtocolError, OSError) as exc:
        print(f"voxlens-replay: the session ended: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    finally:
        client.close()

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
