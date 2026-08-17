"""Scoring VoxLens against a corpus.

Drives the **CLI**, not internal APIs (ADR-0007). The WER reported here is
therefore what a user invoking `voxlens` would get: a regression in argument
handling or output shape cannot hide behind a path only tests exercise.

Measured baselines at beam 1, for regression detection:
    LRS3 test split   34.22% WER   (1,320 Clips, calibration)
    WildVSR           47.85% WER   (570-Clip stride sample, authoritative)

Per ADR-0005 these are valid against themselves, and are NOT claims of parity
with published figures — see docs/adr/0005 for the unresolved discrepancy.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ClipScore", "evaluate", "load_corpus", "normalise", "wer"]

# Reference normalisation, stated once because a WER is meaningless without it:
# lowercase, collapse runs of whitespace, drop surrounding space. Apostrophes
# are KEPT ("don't" stays one word). No other punctuation is stripped, because
# neither corpus's references carry any.
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text.strip().lower())


def wer(reference: str, hypothesis: str) -> tuple[int, int]:
    """Word-level edit distance. Returns (errors, reference_word_count)."""
    ref, hyp = normalise(reference).split(), normalise(hypothesis).split()
    if not ref:
        return (len(hyp), 0)

    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ref_word != hyp_word),  # substitution
                )
            )
        previous = current
    return (previous[-1], len(ref))


@dataclass(frozen=True)
class ClipScore:
    clip: str
    reference: str
    hypothesis: str
    errors: int
    words: int
    duration_s: float
    rtf: float


def load_corpus(root: Path, kind: str, stride: int = 1, limit: int | None = None):
    """Return (clip_path, reference) pairs for a corpus.

    Sampling is by **stride**, never a prefix: both corpora are ordered, so the
    first N Clips may share properties the corpus as a whole does not.
    """
    root = Path(root)
    if kind == "lrs3":
        # A fairseq-style manifest: the first line is a root path, not a Clip.
        manifest = (root / "muavic/en/test.tsv").read_text().splitlines()
        ids = [line.split("\t")[0] for line in manifest if line.strip()]
        refs = (root / "muavic/en/test.wrd").read_text().splitlines()
        if len(ids) == len(refs) + 1:
            ids = ids[1:]
        pairs = [
            (root / "muavic/en/video/test" / f"{clip_id}.mp4", ref)
            for clip_id, ref in zip(ids, refs)
        ]
    elif kind == "wildvsr":
        base = root / "wildvsr/WildVSR"
        labels = json.loads((base / "labels.json").read_text())
        pairs = [(base / "videos" / name, text) for name, text in sorted(labels.items())]
    else:
        raise ValueError(f"Unknown corpus {kind!r}. Choose from: lrs3, wildvsr.")

    pairs = [(path, ref) for path, ref in pairs if path.exists()][::stride]
    return pairs[:limit] if limit else pairs


def evaluate(pairs, checkpoint: Path, device: str, beam: int, voxlens: list[str]):
    """Run the CLI over every Clip and score the results."""
    command = [
        *voxlens,
        *[str(path) for path, _ in pairs],
        "--checkpoint", str(checkpoint),
        "--device", device,
        "--beam", str(beam),
        "--json",
        # Every obtainable corpus ships pre-cropped Mouth Regions (ADR-0005).
        # Running face detection over them would crop a face out of a mouth.
        "--pre-cropped",
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if not process.stdout.strip():
        raise RuntimeError(
            f"voxlens produced no output (exit {process.returncode}).\n"
            f"{process.stderr[-2000:]}"
        )

    by_clip = {}
    for line in process.stdout.splitlines():
        if line.strip():
            payload = json.loads(line)
            by_clip[payload["video"]] = payload

    scores = []
    for path, reference in pairs:
        payload = by_clip.get(str(path))
        if payload is None:
            continue  # the CLI reported this Clip unusable; excluded, not zeroed
        errors, words = wer(reference, payload["transcript"])
        scores.append(
            ClipScore(
                clip=str(path),
                reference=reference,
                hypothesis=payload["transcript"],
                errors=errors,
                words=words,
                duration_s=payload["duration_s"],
                rtf=payload["timing"]["rtf"],
            )
        )
    return scores, process.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="voxlens-eval",
        description="Score VoxLens against a corpus by driving the voxlens CLI.",
    )
    parser.add_argument("root", help="directory holding the corpus")
    parser.add_argument("--corpus", required=True, choices=["lrs3", "wildvsr"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="hybrid", choices=["hybrid", "mps", "cpu"])
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1, help="sample every Nth Clip")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", help="write per-Clip results as JSON here")
    args = parser.parse_args(argv)

    pairs = load_corpus(Path(args.root), args.corpus, stride=args.stride, limit=args.limit)
    if not pairs:
        print("voxlens-eval: no Clips found — check --corpus and the root path", file=sys.stderr)
        return 2
    print(f"scoring {len(pairs)} clips from {args.corpus} ...", file=sys.stderr)

    scores, exit_code = evaluate(
        pairs, Path(args.checkpoint), args.device, args.beam,
        voxlens=[sys.executable, "-m", "voxlens.cli"],
    )
    if not scores:
        print("voxlens-eval: no Clip produced a Transcript", file=sys.stderr)
        return 1

    errors = sum(score.errors for score in scores)
    words = sum(score.words for score in scores)
    audio_s = sum(score.duration_s for score in scores)
    weighted_rtf = (
        sum(score.rtf * score.duration_s for score in scores) / audio_s if audio_s else 0.0
    )

    summary = {
        "corpus": args.corpus,
        "clips_scored": len(scores),
        "clips_requested": len(pairs),
        "words": words,
        "errors": errors,
        "wer_pct": round(100 * errors / words, 2) if words else None,
        "audio_s": round(audio_s, 1),
        "rtf": round(weighted_rtf, 3),
        "config": {"device": args.device, "beam": args.beam, "stride": args.stride},
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")

    if args.out:
        # Per-Clip detail is retained: the worst failures teach more than the mean.
        Path(args.out).write_text(
            json.dumps(
                {"summary": summary, "clips": [vars(score) for score in scores]}, indent=2
            )
        )
        print(f"per-clip results written to {args.out}", file=sys.stderr)

    if len(scores) < len(pairs):
        print(
            f"note: {len(pairs) - len(scores)} clip(s) produced no Transcript and were "
            "excluded from the score, not counted as errors",
            file=sys.stderr,
        )
    return 0 if exit_code == 0 else 0  # per-Clip failures are reported, not fatal


if __name__ == "__main__":
    raise SystemExit(main())
