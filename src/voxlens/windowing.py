"""Decoding a Stream in windows, and reconciling what they read.

ADR-0012: three-second windows advancing one second at a time, with cuts at
Occlusions, and a visible boundary between text that may still be revised and
text that has frozen. This module is that logic and nothing else — it never
touches a model. Windows in, readings back, one Transcript out.

Two things make it more than bookkeeping.

**Overlapping windows read the same second differently.** They must, or there
would be no reason to overlap them: a later window sees more context and often
reads an earlier moment better. So reconciliation cannot look for an exact
seam. It looks for the longest *near* match between what is already accumulated
and what the new window says, and prefers the newer reading where they overlap.

**The revision boundary can only be placed where the overlap ends.** Words have
no timestamps here — ADR-0008 records why alignment is unavailable — so "the
last two seconds" is not a quantity this code can locate. What it can locate is
the point beyond which a new window's reading no longer touches the accumulated
text, and that is one window's advance older than two seconds. See the
amendment on ADR-0012.

Pure, and a seam of its own on ADR-0007's terms: the edge cases are all in the
arithmetic of overlaps and cuts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from voxlens.evaluate import normalise, wer

__all__ = [
    "ADVANCE_S",
    "LiveTranscript",
    "PROVISIONAL_TOLERANCE",
    "Stitcher",
    "WINDOW_S",
    "Window",
    "Windower",
    "overlap",
]

# ADR-0012, in seconds. Frames are derived from the Stream's declared rate
# rather than assumed, so a 25 fps contract change is one place.
WINDOW_S = 3.0
ADVANCE_S = 1.0

# How much two readings of the same stretch may differ and still be treated as
# the same stretch. Roughly one word in three. Exact matching was tried first
# and finds no seam at all in practice — the two windows disagree about a word
# almost every time — which degrades into concatenating repeats, the failure
# #21 explicitly names.
PROVISIONAL_TOLERANCE = 0.34


@dataclass(frozen=True)
class Window:
    """A stretch of Frames to decode as one unit."""

    start_frame: int
    end_frame: int  # exclusive
    # "full" is the ordinary case; "occlusion" ends early because the signal
    # was lost; "final" is the tail of a Stream that has stopped.
    reason: str = "full"

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame

    def seconds(self, fps: float) -> tuple[float, float]:
        return (self.start_frame / fps, self.end_frame / fps)


class Windower:
    """Turns a growing Stream of Frames into windows to decode.

    Fed the number of Frames that have arrived so far, it returns the windows
    that are now complete. Nothing is decoded twice for the same reason it is
    overlapped: each window is a fresh reading of its whole stretch.
    """

    def __init__(self, fps: float = 25.0, window_s: float = WINDOW_S, advance_s: float = ADVANCE_S):
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not 0 < advance_s <= window_s:
            raise ValueError("advance must be positive and no longer than the window")
        self.fps = fps
        self.window = max(1, round(window_s * fps))
        self.advance = max(1, round(advance_s * fps))
        self.next_start = 0
        self._emitted_to = 0

    def feed(self, available: int) -> list[Window]:
        """Windows that have become complete now that `available` Frames exist."""
        windows = []
        while self.next_start + self.window <= available:
            windows.append(Window(self.next_start, self.next_start + self.window))
            self._emitted_to = self.next_start + self.window
            self.next_start += self.advance
        return windows

    def cut(self, at: int, resume: int) -> list[Window]:
        """Close the window early at an Occlusion and resume after it.

        The unreadable Frames are not decoded. Upstream interpolates across
        them and the encoder would read the interpolation as mouth movement,
        which is the invented text this project refuses to produce (ADR-0008).
        """
        if resume < at:
            raise ValueError("an Occlusion cannot end before it starts")
        windows = []
        if at > self.next_start:
            windows.append(Window(self.next_start, at, reason="occlusion"))
            self._emitted_to = at
        self.next_start = resume
        return windows

    def finish(self, available: int) -> list[Window]:
        """The tail of a Stream that has ended: whatever has not been read.

        A window's length, ending on the last Frame — not everything unread
        stretched into one long window. The model reads three seconds well
        because that is what it was given to read; a five-second tail would be
        a different and unmeasured thing. Whatever is unread is always inside
        that span, because a window only fails to be emitted when fewer than
        its length remain.
        """
        if available > self._emitted_to:
            start = max(0, available - self.window)
            self.next_start = available
            return [Window(start, available, reason="final")]
        self.next_start = available
        return []


def overlap(tail: list[str], head: list[str], tolerance: float = PROVISIONAL_TOLERANCE) -> int:
    """How many words at the end of `tail` are the same stretch as the start of `head`.

    Longest acceptable match wins, not best-scoring: a long overlap with one
    disagreement is a better description of two windows reading the same two
    seconds than a short flawless one, which is usually a coincidence of
    function words.
    """
    for size in range(min(len(tail), len(head)), 0, -1):
        distance, _ = wer(" ".join(tail[-size:]), " ".join(head[:size]))
        if distance <= tolerance * size:
            return size
    return 0


@dataclass(frozen=True)
class LiveTranscript:
    """One Transcript in two parts: what has settled, and what may still change."""

    frozen: str
    provisional: str

    @property
    def text(self) -> str:
        """The whole reading, unmarked — what a Clip would have produced."""
        return " ".join(part for part in (self.frozen, self.provisional) if part)

    def rendered(self, marker: str = "[") -> str:
        """Frozen text plain, provisional text bracketed.

        The boundary has to be visible (ADR-0012): text rewriting itself under
        a reader's eyes is exhausting, and a reader who cannot tell which part
        does that has to distrust all of it.
        """
        if not self.provisional:
            return self.frozen
        closing = {"[": "]", "(": ")", "<": ">", "{": "}"}.get(marker, marker)
        edge = f"{marker}{self.provisional}{closing}"
        return f"{self.frozen} {edge}".strip()

    def as_dict(self) -> dict:
        return {
            "frozen": self.frozen,
            "provisional": self.provisional,
            "text": self.text,
        }


@dataclass
class Stitcher:
    """Accumulates window readings into one Transcript with a frozen prefix.

    Frozen text never changes, structurally rather than by convention: words
    below the frozen mark are never rewritten, and the mark only advances.
    """

    tolerance: float = PROVISIONAL_TOLERANCE
    words: list[str] = field(default_factory=list)
    frozen_count: int = 0
    windows: int = 0

    def add(self, reading: str) -> LiveTranscript:
        """Reconcile one window's reading into the Transcript."""
        self.windows += 1
        incoming = normalise(reading).split()
        if not incoming:
            # A window that read nothing says nothing about what came before,
            # so it must not be allowed to freeze or replace any of it.
            return self.current()

        revisable = self.words[self.frozen_count :]
        shared = overlap(revisable, incoming, self.tolerance)

        # Whatever this window does not reach can never be revised again.
        keep = len(self.words) - shared
        self.frozen_count = max(self.frozen_count, keep)
        self.words = self.words[:keep] + incoming
        return self.current()

    def freeze(self) -> LiveTranscript:
        """Settle everything read so far.

        Called at an Occlusion and at the end of a Stream: in both cases no
        later window will cover this stretch, so nothing here is provisional
        any more.
        """
        self.frozen_count = len(self.words)
        return self.current()

    def current(self) -> LiveTranscript:
        return LiveTranscript(
            frozen=" ".join(self.words[: self.frozen_count]),
            provisional=" ".join(self.words[self.frozen_count :]),
        )
