"""Regenerate docs/demo.html from real evaluation output.

The page reports measured behaviour, so it is built from measurements rather
than written by hand. Without this, the figures in it quietly rot: the pipeline
changes, the page keeps claiming last month's numbers, and nobody can tell.

    voxlens-eval CORPUS_ROOT --corpus lrs3 --checkpoint CKPT --out results.json
    voxlens FACE_CLIP --checkpoint CKPT --json > occlusion.json
    python scripts/build_demo.py --results results.json --occlusion occlusion.json

The occlusion input is optional; without it that section is dropped rather than
shown with stale numbers.
"""
from __future__ import annotations

import argparse
import difflib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "demo_template.html"
DEFAULT_OUT = ROOT / "docs" / "demo.html"

# How many clips to show, and where to take them from in the sorted-by-accuracy
# list. Showing only the good ones would be dishonest; showing only the bad ones
# would be theatre.
BEST = 2
TYPICAL = 3
WORST = 2
MIN_WORDS = 5  # a two-word clip's error rate is noise, not signal


def align(reference: str, hypothesis: str) -> list[tuple[str, str | None, str | None]]:
    """Word-level alignment, so each said word sits above what was read for it."""
    ref, hyp = reference.split(), hypothesis.split()
    ops: list[tuple[str, str | None, str | None]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ref, b=hyp).get_opcodes():
        if tag == "equal":
            ops.extend(("eq", word, word) for word in ref[i1:i2])
        elif tag == "replace":
            for offset in range(max(i2 - i1, j2 - j1)):
                ops.append((
                    "sub",
                    ref[i1 + offset] if i1 + offset < i2 else None,
                    hyp[j1 + offset] if j1 + offset < j2 else None,
                ))
        elif tag == "delete":
            ops.extend(("del", word, None) for word in ref[i1:i2])
        elif tag == "insert":
            ops.extend(("ins", None, word) for word in hyp[j1:j2])
    return ops


def render_clip(clip: dict, note: str) -> str:
    cells = []
    for tag, said, read in align(clip["reference"], clip["hypothesis"]):
        top = html.escape(said) if said else '<span class="nil">&mdash;</span>'
        bottom = html.escape(read) if read else '<span class="nil">&mdash;</span>'
        cells.append(
            f'<div class="pair {tag}"><span class="w said">{top}</span>'
            f'<span class="w read">{bottom}</span></div>'
        )
    rate = round(100 * clip["errors"] / clip["words"])
    return (
        '<article class="clip">\n'
        f'  <header class="clip-head">\n    <span class="note">{html.escape(note)}</span>\n'
        f'    <span class="meta"><span class="wer">{rate}%</span> word errors '
        f'&middot; {clip["duration_s"]:.1f}s</span>\n  </header>\n'
        f'  <div class="pairs">{"".join(cells)}</div>\n</article>'
    )


def pick_clips(clips: list[dict]) -> list[tuple[dict, str]]:
    scored = [c for c in clips if c["words"] >= MIN_WORDS]
    scored.sort(key=lambda c: c["errors"] / c["words"])
    if len(scored) < BEST + TYPICAL + WORST:
        raise SystemExit("not enough scored clips to build a representative page")
    middle = len(scored) // 2
    return [
        *[(c, "read exactly") for c in scored[:BEST]],
        *[(c, "close, but wrong") for c in scored[middle : middle + TYPICAL]],
        *[(c, "lost the thread") for c in scored[-WORST:]],
    ]


def render_distribution(clips: list[dict]) -> tuple[str, int, int]:
    buckets = [0] * 11
    for clip in clips:
        if clip["words"]:
            buckets[min(int((100 * clip["errors"] / clip["words"]) // 10), 10)] += 1
    peak = max(buckets) or 1
    labels = [f"{i * 10}&ndash;{i * 10 + 9}" for i in range(10)] + ["100+"]
    rows = "\n".join(
        f'<div class="bar-row"><span class="bl">{labels[i]}</span>'
        f'<span class="bt"><span class="bf" style="width:{100 * count / peak:.1f}%"></span></span>'
        f'<span class="bn">{count}</span></div>'
        for i, count in enumerate(buckets)
    )
    return rows, buckets[0], buckets[10]


def render_occlusion(payload: dict) -> dict:
    duration = payload["duration_s"]
    spans = payload["occlusions"]
    bands = "".join(
        f'<span class="band" style="left:{100 * s["start_s"] / duration:.2f}%;'
        f'width:{100 * (s["end_s"] - s["start_s"]) / duration:.2f}%" '
        f'title="{s["start_s"]:.2f}s &ndash; {s["end_s"]:.2f}s"></span>'
        for s in spans
    )
    step = 2 if duration <= 12 else 5
    ticks = "".join(
        f'<span class="tick" style="left:{100 * t / duration:.2f}%"><i></i>{t}s</span>'
        for t in range(0, int(duration) + 1, step)
    )
    return {
        "BANDS": bands,
        "TICKS": ticks,
        "OCC_COUNT": str(len(spans)),
        "OCC_FRAMES": str(payload["undetected_frames"]),
        "OCC_TOTAL": str(payload["frames"]),
        "OCC_LONGEST": f"{max((s['duration_s'] for s in spans), default=0):.1f}",
        "OCC_DURATION": f"{duration:.1f}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="voxlens-eval --out JSON")
    parser.add_argument("--occlusion", help="voxlens --json output for one face clip")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    data = json.loads(Path(args.results).read_text())
    clips, summary = data["clips"], data["summary"]

    bars, under_ten, over_hundred = render_distribution(clips)
    values = {
        "CLIPS": "\n  ".join(render_clip(c, note) for c, note in pick_clips(clips)),
        "BARS": bars,
        "WER": str(summary["wer_pct"]),
        "PERFECT": str(sum(1 for c in clips if c["words"] and not c["errors"])),
        "TOTAL": f"{len(clips):,}",
        "SPEEDUP": f"{1 / summary['rtf']:.0f}",
        "MINUTES": str(round(summary["audio_s"] / 60)),
        "WORDS": f"{summary['words']:,}",
        "UNDER10": str(under_ten),
        "OVER100": str(over_hundred),
    }

    page = TEMPLATE.read_text()
    if args.occlusion:
        values.update(render_occlusion(json.loads(Path(args.occlusion).read_text())))
    else:
        # Better absent than stale: drop the section rather than keep old numbers.
        start = page.index("<section>\n  <h2>When it cannot see the mouth</h2>")
        end = page.index("</section>", start) + len("</section>")
        page = page[:start] + page[end:]

    for token, value in values.items():
        page = page.replace("{{" + token + "}}", value)

    leftover = page.count("{{")
    if leftover:
        raise SystemExit(f"{leftover} placeholder(s) left unfilled — template drift")

    Path(args.out).write_text(page)
    print(f"wrote {args.out} from {len(clips)} clips ({summary['wer_pct']}% WER)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
