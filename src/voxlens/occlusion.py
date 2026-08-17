"""Turning a per-Frame detection mask into Occlusion spans.

Pure: no model, no video, no torch. This is the second test seam (ADR-0007),
because the edge cases here — runs at the exact threshold, runs touching either
end of a Clip, adjacent runs coalescing — are painful to provoke through a
fixture video and absurd to verify through a 4 GB model.

Per CONTEXT.md an Occlusion is defined by unreadability, not by cause: turned
away, obstructed, out of focus and out of frame are all the same thing here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = ["OcclusionSpan", "DEFAULT_MIN_FRAMES", "find_occlusions", "runs_of_unreadable"]

# Three Frames at 25 fps is 120 ms. A viseme lasts roughly 60-150 ms, so this is
# the point at which an entire viseme can vanish. Below it, interpolating across
# the gap approximates something that was nearly visible; at or above it, the
# recogniser is being fed invented mouth positions.
DEFAULT_MIN_FRAMES = 3


@dataclass(frozen=True)
class OcclusionSpan:
    """A stretch of Frames in which the Speaker's mouth could not be read."""

    start_frame: int
    end_frame: int  # inclusive

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    def as_dict(self, fps: float) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_s": round(self.start_frame / fps, 2),
            "end_s": round((self.end_frame + 1) / fps, 2),
            "duration_s": round(self.frame_count / fps, 2),
        }


def runs_of_unreadable(mask: Sequence[bool], min_frames: int) -> list[OcclusionSpan]:
    """Every run of at least ``min_frames`` consecutive unreadable Frames.

    ``mask[i]`` is True when Frame i had no detectable face.
    """
    if min_frames < 1:
        raise ValueError("min_frames must be at least 1")

    spans: list[OcclusionSpan] = []
    start: int | None = None
    # The sentinel closes a run that reaches the final Frame, so a Clip ending
    # mid-Occlusion is reported rather than silently dropped.
    for index, unreadable in enumerate(list(mask) + [False]):
        if unreadable and start is None:
            start = index
        elif not unreadable and start is not None:
            if index - start >= min_frames:
                spans.append(OcclusionSpan(start_frame=start, end_frame=index - 1))
            start = None
    return spans


def merge_near(spans: Iterable[OcclusionSpan], max_readable_gap: int) -> list[OcclusionSpan]:
    """Coalesce Occlusions separated by very few readable Frames.

    The rule, stated once: **two Occlusions separated by fewer readable Frames
    than the Occlusion threshold itself are one Occlusion.** Tying the merge
    distance to the threshold avoids inventing a second magic number, and the
    reasoning carries over — a readable stretch too short to hold a viseme is
    too short to be worth interrupting the report for.
    """
    merged: list[OcclusionSpan] = []
    for span in spans:
        if merged and span.start_frame - merged[-1].end_frame - 1 < max_readable_gap:
            merged[-1] = OcclusionSpan(
                start_frame=merged[-1].start_frame, end_frame=span.end_frame
            )
        else:
            merged.append(span)
    return merged


def find_occlusions(
    mask: Sequence[bool], min_frames: int = DEFAULT_MIN_FRAMES
) -> list[OcclusionSpan]:
    """The Occlusions in a Clip: detect runs, then coalesce near neighbours."""
    return merge_near(runs_of_unreadable(mask, min_frames), max_readable_gap=min_frames)
