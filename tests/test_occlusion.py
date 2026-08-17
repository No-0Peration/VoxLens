"""The pure seam: detection mask in, Occlusion spans out.

No model, no video, microseconds per case — which is the whole reason this
seam exists (ADR-0007).
"""
import pytest

from voxlens.occlusion import (
    DEFAULT_MIN_FRAMES,
    OcclusionSpan,
    find_occlusions,
    merge_near,
    runs_of_unreadable,
)


def mask(pattern: str) -> list[bool]:
    """'..XXX..' -> readable dots, unreadable Xs. Reads like a filmstrip."""
    return [char == "X" for char in pattern]


def spans(pattern: str, min_frames: int = DEFAULT_MIN_FRAMES):
    return [(s.start_frame, s.end_frame) for s in find_occlusions(mask(pattern), min_frames)]


def test_a_fully_readable_clip_has_no_occlusions():
    assert spans("..........") == []


def test_a_run_shorter_than_the_threshold_is_noise():
    assert spans("...XX.....") == []


def test_a_run_at_exactly_the_threshold_is_an_occlusion():
    """The boundary case the threshold is defined by."""
    assert spans("...XXX....") == [(3, 5)]


def test_a_run_longer_than_the_threshold_is_reported_whole():
    assert spans("..XXXXXX..") == [(2, 7)]


def test_an_occlusion_at_the_very_start_is_reported():
    assert spans("XXXX......") == [(0, 3)]


def test_an_occlusion_running_to_the_final_frame_is_reported():
    """Without a sentinel this run never closes and vanishes silently."""
    assert spans("......XXXX") == [(6, 9)]


def test_a_wholly_unreadable_clip_is_one_occlusion():
    assert spans("XXXXXXXXXX") == [(0, 9)]


def test_an_empty_clip_does_not_crash():
    assert spans("") == []


def test_occlusions_separated_by_a_long_readable_stretch_stay_separate():
    assert spans("XXX....XXX") == [(0, 2), (7, 9)]


def test_occlusions_separated_by_a_brief_flicker_become_one():
    """The prototype emitted four markers in one sentence; that is noise."""
    assert spans("XXX.XXX...") == [(0, 6)]


def test_the_threshold_is_configurable():
    assert spans("...XX.....", min_frames=2) == [(3, 4)]
    assert spans("...XX.....", min_frames=3) == []


def test_a_threshold_below_one_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        runs_of_unreadable(mask("XXX"), min_frames=0)


def test_merging_is_left_alone_when_nothing_is_near():
    original = [OcclusionSpan(0, 2), OcclusionSpan(20, 25)]
    assert merge_near(original, max_readable_gap=3) == original


def test_spans_convert_to_seconds_against_the_clip_rate():
    span = OcclusionSpan(start_frame=25, end_frame=49)
    payload = span.as_dict(fps=25.0)
    assert payload["start_s"] == 1.0
    assert payload["end_s"] == 2.0  # end_frame is inclusive
    assert payload["duration_s"] == 1.0
    assert span.frame_count == 25
