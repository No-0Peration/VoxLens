"""Scoring footage shot through a lens (#18).

The corpus kind that is not pre-cropped, and the only one whose Clips are
grouped by the condition they were shot under. Neither needs a checkpoint: what
is asserted is which Clips are found, how they are grouped, and that extraction
is left switched on.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from voxlens.evaluate import evaluate, group_of, load_corpus


@pytest.fixture
def captured(tmp_path):
    """Two distances, two Clips each, as an afternoon with a tripod produces."""
    names = ["2m_00001.mp4", "2m_00002.mp4", "8m_00001.mp4", "8m_00002.mp4"]
    for name in names:
        (tmp_path / name).write_bytes(b"not really a video")
    (tmp_path / "references.tsv").write_text(
        "# distance_clip\treference\n"
        + "\n".join(f"{name}\tthe words that were said" for name in names)
        + "\n"
    )
    return tmp_path


def test_captured_footage_is_found_with_its_references(captured):
    pairs = load_corpus(captured, "captured")
    assert len(pairs) == 4
    assert all(reference == "the words that were said" for _, reference in pairs)
    assert all(path.exists() for path, _ in pairs)


def test_comments_and_blank_lines_are_skipped(captured):
    assert len(load_corpus(captured, "captured")) == 4  # the fixture has both


def test_a_manifest_line_without_a_reference_is_refused(tmp_path):
    """Silently dropping it would score fewer Clips than were filmed and never
    say so."""
    (tmp_path / "references.tsv").write_text("2m_00001.mp4\n")
    with pytest.raises(ValueError, match="filename"):
        load_corpus(tmp_path, "captured")


def test_missing_references_say_where_the_procedure_is(tmp_path):
    with pytest.raises(ValueError, match="camera-path-measurement"):
        load_corpus(tmp_path, "captured")


def test_clips_are_grouped_by_what_precedes_the_first_underscore():
    assert group_of("8m_00002.mp4") == "8m"
    assert group_of("/somewhere/12m_x_2.mp4") == "12m"
    assert group_of("ungrouped.mp4") == ""


def test_captured_footage_keeps_face_detection_switched_on(captured, tmp_path):
    """The one corpus with whole faces in it. Skipping extraction would flatter
    the result by measuring everything except the part most likely to fail at
    distance."""
    recorded = {}

    def fake_cli(command):
        recorded["command"] = command

        class Result:
            stdout = ""
            stderr = ""
            returncode = 0

        return Result()

    import voxlens.evaluate as module

    original = module.subprocess.run
    module.subprocess.run = lambda command, **kwargs: fake_cli(command)
    try:
        with pytest.raises(RuntimeError):  # no output, since nothing really ran
            evaluate(
                load_corpus(captured, "captured"),
                Path("/none.pth"),
                "cpu",
                1,
                voxlens=["voxlens"],
                pre_cropped=False,
            )
    finally:
        module.subprocess.run = original

    assert "--pre-cropped" not in recorded["command"]
