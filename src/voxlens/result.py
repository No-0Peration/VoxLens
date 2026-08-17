"""What a run of VoxLens produced.

The JSON form is not a convenience: the evaluation harness consumes it, so it
is a load-bearing interface (ADR-0007). Adding fields is safe; renaming or
removing them breaks evaluation and is a breaking change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from voxlens.occlusion import OcclusionSpan

__all__ = ["Result", "Timing", "checkpoint_identity"]


@dataclass(frozen=True)
class Timing:
    """Seconds spent in each stage, and the resulting real-time factor."""

    extract_s: float
    infer_s: float
    duration_s: float

    @property
    def total_s(self) -> float:
        return self.extract_s + self.infer_s

    @property
    def rtf(self) -> float:
        """Below 1.0 means faster than real time."""
        return self.total_s / self.duration_s if self.duration_s else float("nan")

    def as_dict(self) -> dict:
        return {
            "extract_s": round(self.extract_s, 3),
            "infer_s": round(self.infer_s, 3),
            "total_s": round(self.total_s, 3),
            "rtf": round(self.rtf, 3),
        }


def checkpoint_identity(path: str | Path) -> dict:
    """Enough to tell which checkpoint produced a result, without hashing 4 GB."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        size = None
    return {"path": str(path.resolve()), "size_bytes": size}


@dataclass(frozen=True)
class Result:
    """A Transcript plus everything needed to reproduce and judge it."""

    video: str
    frames: int
    fps: float
    duration_s: float
    transcript: str
    undetected_frames: int
    timing: Timing
    device: str
    beam: int
    occlusion_min_frames: int
    occlusions: tuple[OcclusionSpan, ...] = ()
    checkpoint: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "video": self.video,
            "frames": self.frames,
            "fps": round(self.fps, 2),
            "duration_s": round(self.duration_s, 2),
            "transcript": self.transcript,
            "undetected_frames": self.undetected_frames,
            # Spans only, never per-word marking: the CTC head and beam search
            # transcribe differently, so word positions cannot be trusted
            # (ADR-0008).
            "occlusions": [span.as_dict(self.fps) for span in self.occlusions],
            "timing": self.timing.as_dict(),
            "config": {
                "device": self.device,
                "beam": self.beam,
                "occlusion_min_frames": self.occlusion_min_frames,
                "checkpoint": self.checkpoint,
            },
        }

    def summary_line(self) -> str:
        """The one-line diagnostic, for stderr in both output modes."""
        return (
            f"{self.frames} frames, {self.duration_s:.1f}s, RTF {self.timing.rtf:.2f}"
            f"  |  {len(self.occlusions)} occlusion(s), "
            f"{self.undetected_frames} frame(s) with no detected face"
        )

    def occlusion_lines(self) -> list[str]:
        """Human-readable Occlusion report, for stderr.

        Occlusions are NOT spliced into the Transcript on stdout. Placing them
        within the text would require knowing which words they cover, and
        ADR-0008 records why that cannot be done reliably. Reporting them
        beside the Transcript is honest; guessing a position would not be.
        """
        if not self.occlusions:
            return []
        lines = ["mouth unreadable:"]
        for span in self.occlusions:
            data = span.as_dict(self.fps)
            lines.append(
                f"  {data['start_s']:6.2f}s - {data['end_s']:6.2f}s"
                f"  ({data['duration_s']:.2f}s, frames {span.start_frame}-{span.end_frame})"
            )
        return lines
