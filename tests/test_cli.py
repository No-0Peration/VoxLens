"""Tests at the CLI seam (ADR-0007).

These run the real command as a subprocess. Most of them need no checkpoint:
a Clip whose mouth cannot be read fails during extraction, before the model is
ever loaded, so the failure paths are cheap to cover.
"""
from __future__ import annotations

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
