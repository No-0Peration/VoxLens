"""Tests at the CLI seam (ADR-0007).

These run the real command as a subprocess. Most of them need no checkpoint:
a Clip whose mouth cannot be read fails during extraction, before the model is
ever loaded, so the failure paths are cheap to cover.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from voxlens.upstream import is_vendored

CHECKPOINT = os.environ.get("VOXLENS_CHECKPOINT")
FACE_CLIP = os.environ.get("VOXLENS_TEST_CLIP")

needs_upstream = pytest.mark.skipif(
    not is_vendored(), reason="upstream not vendored — run: uv run python scripts/vendor.py"
)
needs_face_clip = pytest.mark.skipif(
    not (CHECKPOINT and FACE_CLIP and os.path.exists(CHECKPOINT) and os.path.exists(FACE_CLIP)),
    reason="set VOXLENS_CHECKPOINT and VOXLENS_TEST_CLIP to run this",
)


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "voxlens.cli", *args],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def faceless_clip(tmp_path):
    """A decodable Clip with no Speaker in it."""
    path = tmp_path / "faceless.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
    for _ in range(30):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    return path


def test_help_names_the_defaults_a_reader_needs():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "hybrid" in result.stdout
    assert "--beam" in result.stdout


def test_a_missing_clip_is_reported_as_bad_input(tmp_path):
    result = run_cli(str(tmp_path / "nope.mp4"), "--checkpoint", "/dev/null")
    assert result.returncode == 2
    assert "No such file" in result.stderr
    assert result.stdout == ""


def test_an_undecodable_clip_is_reported_as_bad_input(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video")
    result = run_cli(str(junk), "--checkpoint", "/dev/null")
    assert result.returncode == 2
    assert result.stdout == ""


def test_an_unknown_device_is_refused_before_any_work(tmp_path):
    result = run_cli(str(tmp_path / "nope.mp4"), "--checkpoint", "/dev/null", "--device", "tpu")
    assert result.returncode == 2
    assert "tpu" in result.stderr


@needs_upstream
def test_a_clip_with_no_speaker_is_distinguished_from_bad_input(faceless_clip):
    """Exit 1 means 'your video is unusable'; exit 2 means 'your path is wrong'."""
    result = run_cli(str(faceless_clip), "--checkpoint", "/dev/null")
    assert result.returncode == 1, result.stderr
    assert "mouth" in result.stderr.lower()
    assert result.stdout == ""


@needs_upstream
@needs_face_clip
def test_transcribing_a_clip_prints_a_transcript_and_succeeds():
    result = run_cli(FACE_CLIP, "--checkpoint", CHECKPOINT)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), "expected a Transcript on stdout"
    assert "RTF" in result.stderr, "diagnostics belong on stderr"


@needs_upstream
@needs_face_clip
def test_every_device_produces_the_same_transcript():
    """The hybrid split is a performance decision, never an accuracy one."""
    transcripts = {}
    for device in ("cpu", "hybrid"):
        result = run_cli(FACE_CLIP, "--checkpoint", CHECKPOINT, "--device", device)
        assert result.returncode == 0, result.stderr
        transcripts[device] = result.stdout.strip()
    assert transcripts["cpu"] == transcripts["hybrid"]


@needs_upstream
@needs_face_clip
def test_json_mode_puts_only_the_payload_on_stdout():
    result = run_cli(FACE_CLIP, "--checkpoint", CHECKPOINT, "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)  # would raise if anything else leaked
    assert payload["transcript"]
    assert payload["frames"] > 0


@needs_upstream
@needs_face_clip
def test_json_payload_records_what_produced_it():
    """A result has to be reproducible from its own output."""
    result = run_cli(FACE_CLIP, "--checkpoint", CHECKPOINT, "--json", "--beam", "1")
    payload = json.loads(result.stdout)
    config = payload["config"]
    assert config["beam"] == 1
    assert config["device"] in ("hybrid", "mps", "cpu")
    assert config["checkpoint"]["path"].endswith(".pth")
    assert config["checkpoint"]["size_bytes"] > 0


@needs_upstream
@needs_face_clip
def test_json_payload_separates_extraction_from_inference():
    """Extraction is the largest component; hiding it inside one total would
    conceal the thing most worth watching."""
    payload = json.loads(run_cli(FACE_CLIP, "--checkpoint", CHECKPOINT, "--json").stdout)
    timing = payload["timing"]
    assert timing["extract_s"] > 0
    assert timing["infer_s"] > 0
    assert timing["rtf"] > 0
    assert timing["total_s"] == pytest.approx(
        timing["extract_s"] + timing["infer_s"], abs=0.01
    )


@needs_upstream
@needs_face_clip
def test_diagnostics_stay_on_stderr_in_both_modes():
    for extra in ([], ["--json"]):
        result = run_cli(FACE_CLIP, "--checkpoint", CHECKPOINT, *extra)
        assert "RTF" in result.stderr
        assert "RTF" not in result.stdout


@needs_upstream
@needs_face_clip
def test_human_mode_still_prints_a_bare_transcript():
    """--json is additive; the default output is unchanged."""
    plain = run_cli(FACE_CLIP, "--checkpoint", CHECKPOINT)
    payload = json.loads(run_cli(FACE_CLIP, "--checkpoint", CHECKPOINT, "--json").stdout)
    assert plain.stdout.strip() == payload["transcript"]


@needs_upstream
@needs_face_clip
def test_json_reports_occlusion_spans_with_frames_and_seconds():
    payload = json.loads(run_cli(FACE_CLIP, "--checkpoint", CHECKPOINT, "--json").stdout)
    assert payload["occlusions"], "this fixture has undetected frames"
    span = payload["occlusions"][0]
    assert set(span) == {"start_frame", "end_frame", "start_s", "end_s", "duration_s"}
    assert span["end_frame"] >= span["start_frame"]
    assert span["duration_s"] > 0


@needs_upstream
@needs_face_clip
def test_no_word_is_labelled_read_or_unreadable():
    """ADR-0008: spans only. Per-word marking would need alignment we do not have."""
    payload = json.loads(run_cli(FACE_CLIP, "--checkpoint", CHECKPOINT, "--json").stdout)
    assert "words" not in payload
    assert "read" not in json.dumps(payload["occlusions"])


@needs_upstream
@needs_face_clip
def test_raising_the_threshold_reports_fewer_occlusions():
    def count(threshold: str) -> int:
        out = run_cli(FACE_CLIP, "--checkpoint", CHECKPOINT, "--json",
                      "--occlusion-min-frames", threshold).stdout
        return len(json.loads(out)["occlusions"])

    assert count("50") < count("3"), "a high threshold should suppress short gaps"


@needs_upstream
@needs_face_clip
def test_the_transcript_on_stdout_stays_free_of_occlusion_markers():
    """Occlusion is reported beside the Transcript, never spliced into it."""
    plain = run_cli(FACE_CLIP, "--checkpoint", CHECKPOINT)
    assert "unreadable" not in plain.stdout
    assert "unreadable" in plain.stderr
    payload = json.loads(run_cli(FACE_CLIP, "--checkpoint", CHECKPOINT, "--json").stdout)
    assert plain.stdout.strip() == payload["transcript"]


# --- exit codes (#16) ------------------------------------------------------
# 1 vs 2 is the distinction that matters: a harness running hundreds of Clips
# must tell "this video is unusable" from "this invocation is wrong".

@needs_upstream
def test_a_missing_checkpoint_is_a_model_failure_not_bad_input(faceless_clip, tmp_path):
    """Reached only once the Clip is fine, so it cannot be confused with exit 2."""
    result = run_cli(str(tmp_path / "absent.pth"), "--checkpoint", str(tmp_path / "absent.pth"))
    assert result.returncode == 2  # the video argument is what is missing here

    from voxlens.recogniser import CheckpointError, load_recogniser
    from voxlens.devices import resolve_device

    with pytest.raises(CheckpointError, match="never downloads"):
        load_recogniser(tmp_path / "absent.pth", resolve_device("cpu"))


@needs_upstream
def test_a_corrupt_checkpoint_is_reported_as_a_model_failure(tmp_path):
    from voxlens.devices import resolve_device
    from voxlens.recogniser import CheckpointError, load_recogniser

    junk = tmp_path / "corrupt.pth"
    junk.write_bytes(b"not a torch checkpoint")
    with pytest.raises(CheckpointError, match="Could not read"):
        load_recogniser(junk, resolve_device("cpu"))


@needs_upstream
def test_a_checkpoint_of_the_wrong_architecture_is_rejected(tmp_path):
    import torch

    from voxlens.devices import resolve_device
    from voxlens.recogniser import CheckpointError, load_recogniser

    wrong = tmp_path / "wrong.pth"
    torch.save({"not.a.real.layer": torch.zeros(2)}, wrong)
    with pytest.raises(CheckpointError, match="not a resnet_transformer_large"):
        load_recogniser(wrong, resolve_device("cpu"))


def test_every_failure_path_keeps_stdout_empty(tmp_path):
    """Anything on stdout would corrupt a --json consumer downstream."""
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"nope")
    for argv in (
        [str(tmp_path / "missing.mp4"), "--checkpoint", "/dev/null"],
        [str(junk), "--checkpoint", "/dev/null"],
    ):
        result = run_cli(*argv)
        assert result.returncode != 0
        assert result.stdout == ""
        # `in`, not `startswith`: OpenCV/FFmpeg writes its own diagnostics to
        # stderr first ("moov atom not found"). Noisy, but stderr is the right
        # stream for it, and the contract is that stdout stays clean.
        assert "voxlens:" in result.stderr


def test_help_documents_the_exit_codes():
    """A batch script author should not have to read the source."""
    out = run_cli("--help").stdout
    assert "exit codes:" in out
    for code in ("0", "1", "2", "3"):
        assert f"  {code}  " in out
