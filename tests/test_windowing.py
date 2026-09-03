"""Tests for windowed decoding (#21).

Pure, on ADR-0007's terms: scheduling windows and reconciling their readings
is arithmetic with real edge cases — a cut at the exact window boundary, two
windows that disagree about a word, a window that read nothing — and none of
them are worth provoking through a 4 GB model.
"""
from __future__ import annotations

import pytest

from voxlens.windowing import LiveTranscript, Stitcher, Window, Windower, overlap


# --- scheduling -----------------------------------------------------------

def test_nothing_is_decoded_until_a_whole_window_has_arrived():
    """Three seconds at 25 fps. A partial window would be read as a complete
    utterance, and the model would end it with something it did not hear."""
    windower = Windower(fps=25)
    assert windower.feed(74) == []
    assert windower.feed(75) == [Window(0, 75)]


def test_windows_advance_a_second_at_a_time_and_overlap():
    windower = Windower(fps=25)
    windows = windower.feed(150)
    assert [(w.start_frame, w.end_frame) for w in windows] == [
        (0, 75), (25, 100), (50, 125), (75, 150),
    ]


def test_frames_already_covered_are_not_scheduled_twice():
    windower = Windower(fps=25)
    first = windower.feed(100)
    again = windower.feed(100)
    assert first and again == []


def test_an_occlusion_closes_the_window_early_and_resumes_after_it():
    """The unreadable Frames are never decoded: upstream interpolates across
    them, and the encoder would read that interpolation as mouth movement."""
    windower = Windower(fps=25)
    windower.feed(90)  # one full window, 15 frames pending
    cut = windower.cut(at=90, resume=140)

    assert [(w.start_frame, w.end_frame, w.reason) for w in cut] == [(25, 90, "occlusion")]
    assert windower.feed(214) == []  # not yet three seconds past the resume point
    assert windower.feed(215) == [Window(140, 215)]


def test_an_occlusion_with_nothing_pending_cuts_without_a_stub_window():
    windower = Windower(fps=25)
    windower.feed(75)
    assert windower.cut(at=25, resume=50) == []


def test_the_tail_of_a_stream_is_read_when_it_ends():
    """A window's length ending on the last Frame, not everything unread
    stretched into one overlong one: three seconds is what the model reads
    well, and a longer tail would be a different, unmeasured thing."""
    windower = Windower(fps=25)
    windower.feed(100)
    tail = windower.finish(120)
    assert [(w.start_frame, w.end_frame, w.reason) for w in tail] == [(45, 120, "final")]
    assert tail[0].frames == 75


def test_a_stream_shorter_than_one_window_is_still_read():
    windower = Windower(fps=25)
    assert windower.feed(40) == []
    assert [(w.start_frame, w.end_frame) for w in windower.finish(40)] == [(0, 40)]


def test_a_stream_that_ends_on_a_window_boundary_has_no_tail():
    windower = Windower(fps=25)
    windower.feed(75)
    assert windower.finish(75) == []


def test_a_window_shorter_than_its_advance_is_refused():
    with pytest.raises(ValueError):
        Windower(fps=25, window_s=1.0, advance_s=3.0)
    with pytest.raises(ValueError):
        Windower(fps=0)


# --- reconciliation -------------------------------------------------------

def test_an_exact_seam_is_found():
    assert overlap("a b c d".split(), "b c d e".split()) == 3


def test_a_seam_survives_the_windows_disagreeing_about_a_word():
    """They must disagree sometimes, or overlapping them would buy nothing.
    Requiring an exact match finds no seam and concatenates repeats instead."""
    assert overlap("the wash was cold".split(), "the wish was cold today".split()) == 4


def test_readings_with_nothing_in_common_have_no_seam():
    assert overlap("alpha bravo".split(), "yankee zulu".split()) == 0


def test_the_longest_acceptable_seam_wins():
    """A long overlap with one disagreement describes two windows reading the
    same stretch; a short flawless one is usually a coincidence of small words."""
    assert overlap("one two three the".split(), "one two three the four".split()) == 4


def test_overlapping_windows_become_one_transcript_without_repeats():
    stitcher = Stitcher()
    stitcher.add("the choices dont make")
    stitcher.add("choices dont make sense because")
    live = stitcher.add("make sense because its the wrong")
    assert live.text == "the choices dont make sense because its the wrong"


def test_the_newer_reading_wins_where_two_windows_overlap():
    """Later windows see more context and often read an earlier moment better,
    which is the entire reason for overlapping them."""
    stitcher = Stitcher()
    stitcher.add("the wash was")
    live = stitcher.add("the wish was cold")
    assert "wish" in live.text and "wash" not in live.text


def test_frozen_text_never_changes_however_the_windows_disagree():
    stitcher = Stitcher()
    frozen_over_time = []
    for reading in (
        "one two three four",
        "three four five six",
        "five six seven eight",
        "seven eight nine ten",
    ):
        frozen_over_time.append(stitcher.add(reading).frozen)

    for earlier, later in zip(frozen_over_time, frozen_over_time[1:]):
        assert later.startswith(earlier), "frozen text was rewritten"


def test_the_provisional_edge_is_what_a_later_window_can_still_reach():
    stitcher = Stitcher()
    stitcher.add("one two three four")
    live = stitcher.add("three four five six")
    assert live.frozen == "one two"
    assert live.provisional == "three four five six"


def test_a_window_that_read_nothing_settles_nothing():
    """Silence from one window is not evidence about what came before it."""
    stitcher = Stitcher()
    stitcher.add("one two three")
    before = stitcher.current()
    after = stitcher.add("")
    assert after == before


def test_freezing_settles_everything_read_so_far():
    """At an Occlusion and at the end of a Stream, no later window covers this
    stretch, so nothing in it is provisional any more."""
    stitcher = Stitcher()
    stitcher.add("one two three")
    live = stitcher.freeze()
    assert live.provisional == ""
    assert live.frozen == "one two three"


def test_the_boundary_is_visible_in_the_rendered_form():
    live = LiveTranscript(frozen="settled words", provisional="still moving")
    assert live.rendered() == "settled words [still moving]"
    assert live.rendered("<") == "settled words <still moving>"
    assert LiveTranscript(frozen="all done", provisional="").rendered() == "all done"


def test_the_two_parts_reassemble_into_the_whole_reading():
    live = LiveTranscript(frozen="one two", provisional="three four")
    assert live.text == "one two three four"
    assert live.as_dict() == {
        "frozen": "one two",
        "provisional": "three four",
        "text": "one two three four",
    }
