"""Tests at the transport seam (#19).

The server is a process boundary like the CLI is (ADR-0007), so these drive it
over a real socket: real framing, real disconnects, real error replies. What
they do not drive is a real recogniser — the server takes one as a callable, so
a stub stands in and every message on the wire is covered without a 4 GB
checkpoint. The one test that needs the model says so and skips.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading

import cv2
import numpy as np
import pytest

from voxlens.transport import (
    CROP_SHAPE,
    CROP_SIZE,
    CropClient,
    Disconnected,
    ServerError,
    encode_crops,
    read_message,
    send_message,
)
from voxlens.serve import CropServer
from voxlens.upstream import is_vendored

CHECKPOINT = os.environ.get("VOXLENS_CHECKPOINT")
CROP_CLIP = os.environ.get("VOXLENS_CROP_CLIP")

needs_crop_clip = pytest.mark.skipif(
    not (
        is_vendored()
        and CHECKPOINT
        and CROP_CLIP
        and os.path.exists(CHECKPOINT)
        and os.path.exists(CROP_CLIP)
    ),
    reason="set VOXLENS_CHECKPOINT and VOXLENS_CROP_CLIP (a pre-cropped 96x96 "
    "Mouth Region clip, as evaluation corpora ship) to run this",
)

STUB_CHECKPOINT = {"path": "/stub/usr2_large.pth", "size_bytes": 1}


def crops(frames: int, value: int = 7) -> np.ndarray:
    return np.full((frames, *CROP_SHAPE), value, dtype=np.uint8)


@pytest.fixture
def serve():
    """Start stub servers on a free port, and stop them however a test ends."""
    running = []

    def start(transcribe=None, record_errors=False) -> CropServer:
        seen: list = []

        def default(batch):
            seen.append(batch)
            return f"{len(batch)} crops"

        server = CropServer(
            ("127.0.0.1", 0),
            transcribe or default,
            device="cpu",
            beam=1,
            checkpoint=STUB_CHECKPOINT,
            log=lambda message: None,
        )
        server.seen = seen
        if record_errors:
            # A test that provokes a model failure expects the traceback; it
            # just does not want it in the pytest report.
            server.errors = []
            server.handle_error = lambda request, address: server.errors.append(address)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        running.append((server, thread))
        return server

    yield start

    for server, thread in running:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def client_for(server: CropServer, **kwargs) -> CropClient:
    return CropClient(host="127.0.0.1", port=server.port, timeout=10, **kwargs)


def raw_connection(server: CropServer):
    """A socket with no client wrapped around it, for malformed messages."""
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=10)
    sock.settimeout(10)
    return sock, sock.makefile("rb"), sock.makefile("wb")


# --- the session ----------------------------------------------------------

def test_hello_answers_with_the_crop_contract(serve):
    """A client that gets crops wrong should find out at hello, not from a
    Transcript of noise."""
    with client_for(serve(), session="camera-1") as client:
        ready = client.ready

    assert ready["session"] == "camera-1"
    assert ready["crop"] == {
        "size": CROP_SIZE,
        "channels": 3,
        "dtype": "uint8",
        "layout": "RGB",
    }
    assert ready["fps"] == 25.0
    assert ready["config"]["checkpoint"] == STUB_CHECKPOINT


def test_crops_come_back_as_a_transcript(serve):
    server = serve()
    with client_for(server) as client:
        reply = client.send_crops(crops(25))

    assert reply["transcript"] == "25 crops"
    assert reply["frames"] == 25
    assert reply["duration_s"] == 1.0  # 25 frames at 25 fps
    assert server.seen[0].shape == (25, CROP_SIZE, CROP_SIZE, 3)
    assert server.seen[0].dtype == np.uint8


def test_a_session_identity_travels_with_every_reply(serve):
    """Two cameras on one Mac must be tellable apart in the replies."""
    server = serve()
    with client_for(server, session="left") as left, client_for(server, session="right") as right:
        assert left.send_crops(crops(5))["session"] == "left"
        assert right.send_crops(crops(5))["session"] == "right"


def test_replies_are_numbered_within_the_session(serve):
    with client_for(serve()) as client:
        assert [client.send_crops(crops(5))["sequence"] for _ in range(3)] == [1, 2, 3]


def test_one_recogniser_serves_every_request_and_every_session(serve):
    """The checkpoint is loaded before the socket opens, so nothing per request
    or per session can reload it. What the wire has to show is that the same
    recogniser answers all of them."""
    server = serve()
    for _ in range(2):
        with client_for(server) as client:
            client.send_crops(crops(4))
            client.send_crops(crops(4))

    assert len(server.seen) == 4, "all four requests reached the one stub"


def test_timing_reports_no_extraction_because_the_camera_did_it(serve):
    """extract_s is zero on purpose: sending crops rather than video is what
    moves extraction to the phone (ADR-0009)."""
    with client_for(serve()) as client:
        timing = client.send_crops(crops(10))["timing"]

    assert timing["extract_s"] == 0.0
    assert timing["infer_s"] >= 0.0
    assert timing["rtf"] >= 0.0


# --- what is refused ------------------------------------------------------

def test_crops_before_a_hello_are_refused(serve):
    sock, rfile, wfile = raw_connection(serve())
    try:
        send_message(wfile, {"type": "crops", "frames": 1}, encode_crops(crops(1)))
        header, _ = read_message(rfile)
    finally:
        sock.close()

    assert header["type"] == "error"
    assert header["code"] == "protocol"
    assert "hello" in header["message"]


def test_a_client_speaking_another_protocol_version_is_refused(serve):
    sock, rfile, wfile = raw_connection(serve())
    try:
        send_message(wfile, {"type": "hello", "protocol": 99, "session": "old", "fps": 25.0})
        header, _ = read_message(rfile)
    finally:
        sock.close()

    assert header["code"] == "protocol"
    assert "99" in header["message"]


def test_a_frame_rate_the_checkpoint_cannot_use_is_refused(serve):
    """Accepting 30 fps would cost accuracy with nothing in the output to say why."""
    with pytest.raises(ServerError) as caught:
        client_for(serve(), fps=30.0).open()

    assert caught.value.code == "protocol"
    assert "25" in caught.value.message


def test_a_nameless_session_is_refused(serve):
    sock, rfile, wfile = raw_connection(serve())
    try:
        send_message(wfile, {"type": "hello", "protocol": 1, "session": "  ", "fps": 25.0})
        header, _ = read_message(rfile)
    finally:
        sock.close()

    assert header["code"] == "protocol"
    assert "session" in header["message"]


def test_a_payload_that_contradicts_its_frame_count_is_refused(serve):
    """Desync is the failure that would otherwise transcribe garbage silently."""
    server = serve()
    sock, rfile, wfile = raw_connection(server)
    try:
        send_message(wfile, {"type": "hello", "protocol": 1, "session": "liar", "fps": 25.0})
        read_message(rfile)
        send_message(wfile, {"type": "crops", "frames": 10}, encode_crops(crops(3)))
        header, _ = read_message(rfile)
    finally:
        sock.close()

    assert header["code"] == "protocol"
    assert "bytes" in header["message"]
    assert server.seen == [], "nothing reached the recogniser"


def test_an_unknown_message_type_is_refused(serve):
    sock, rfile, wfile = raw_connection(serve())
    try:
        send_message(wfile, {"type": "hello", "protocol": 1, "session": "s", "fps": 25.0})
        read_message(rfile)
        send_message(wfile, {"type": "shrug"})
        header, _ = read_message(rfile)
    finally:
        sock.close()

    assert header["code"] == "protocol"
    assert "shrug" in header["message"]


def test_garbage_on_the_socket_is_not_read_as_a_message(serve):
    """Something other than a VoxLens client will connect eventually."""
    sock, rfile, wfile = raw_connection(serve())
    try:
        sock.sendall(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        header, _ = read_message(rfile)
    finally:
        sock.close()

    assert header["code"] == "protocol"


def test_off_contract_crops_are_refused_before_they_reach_the_wire(serve):
    """A phone shipping the wrong shape should hear about it where the bug is."""
    with client_for(serve()) as client:
        with pytest.raises(ValueError, match="96"):
            client.send_crops(np.zeros((5, 88, 88, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="uint8"):
            client.send_crops(np.zeros((5, *CROP_SHAPE), dtype=np.float32))
        # The session survives its own client's mistakes.
        assert client.send_crops(crops(5))["transcript"] == "5 crops"


# --- losing the camera ----------------------------------------------------

def test_a_disconnect_mid_session_leaves_the_next_session_working(serve):
    server = serve()
    sock, _, wfile = raw_connection(server)
    send_message(wfile, {"type": "hello", "protocol": 1, "session": "gone", "fps": 25.0})
    sock.close()  # the phone went into a pocket

    with client_for(server, session="next") as client:
        assert client.send_crops(crops(5))["transcript"] == "5 crops"


def test_a_disconnect_with_crops_in_flight_leaves_the_next_session_working(serve):
    """Half a message is the interesting case: the server is mid-read, holding
    a frame count it will never receive the frames for."""
    server = serve()
    sock, _, wfile = raw_connection(server)
    send_message(wfile, {"type": "hello", "protocol": 1, "session": "cut", "fps": 25.0})
    payload = encode_crops(crops(25))
    send_message(wfile, {"type": "crops", "frames": 25, "bytes": len(payload)}, payload[:1000])
    sock.close()

    with client_for(server, session="after") as client:
        assert client.send_crops(crops(5))["transcript"] == "5 crops"
    assert len(server.seen) == 1, "the truncated batch was never transcribed"


def test_a_model_failure_reaches_the_client_rather_than_an_empty_transcript(serve):
    def broken(_batch):
        raise RuntimeError("MPS backend out of memory")

    server = serve(transcribe=broken, record_errors=True)
    with client_for(server) as client:
        with pytest.raises(ServerError) as caught:
            client.send_crops(crops(5))

    assert caught.value.code == "model"
    assert "out of memory" in caught.value.message
    assert server.errors, "the traceback still surfaces on the server"


# --- the replay client ----------------------------------------------------

def run_replay(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "voxlens.replay", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_replay_drives_a_server_end_to_end(serve, tmp_path):
    """The acceptance criterion that is not a phone: crops off disk, text back."""
    server = serve()
    path = tmp_path / "mouth.npy"
    np.save(path, crops(50))

    result = run_replay(str(path), "--port", str(server.port))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "50 crops"
    assert "session" in result.stderr  # diagnostics never touch stdout


def test_replay_can_split_a_file_into_several_requests(serve, tmp_path):
    server = serve()
    path = tmp_path / "mouth.npy"
    np.save(path, crops(75))

    result = run_replay(str(path), "--port", str(server.port), "--chunk", "25", "--json")

    assert result.returncode == 0, result.stderr
    replies = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert [reply["sequence"] for reply in replies] == [1, 2, 3]
    assert all(reply["frames"] == 25 for reply in replies)


def test_replay_refuses_a_file_of_whole_frames(serve, tmp_path):
    """Whole Frames are what the phone extracts from; the wire carries crops."""
    path = tmp_path / "frames.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
    for _ in range(10):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()

    result = run_replay(str(path), "--port", str(serve().port))

    assert result.returncode == 2
    assert "96x96" in result.stderr
    assert result.stdout == ""


def test_replay_says_what_to_start_when_nothing_is_listening(tmp_path):
    path = tmp_path / "mouth.npy"
    np.save(path, crops(5))
    with socket.socket() as probe:  # a port that is bound but not serving
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    result = run_replay(str(path), "--port", str(port))

    assert result.returncode == 2
    assert "voxlens-serve" in result.stderr


def test_a_missing_crop_file_is_reported_before_a_session_opens(serve, tmp_path):
    result = run_replay(str(tmp_path / "absent.npy"), "--port", str(serve().port))
    assert result.returncode == 2
    assert result.stdout == ""


# --- with the real recogniser --------------------------------------------

@needs_crop_clip
def test_the_server_and_the_cli_read_the_same_crops_the_same_way(tmp_path):
    """ADR-0009's claim, tested: the Mac does what --pre-cropped already does.

    Same file, same checkpoint, two routes in. Any divergence means the network
    path is a second recognition path, which is the thing this must not be.
    """
    server = subprocess.Popen(
        [
            sys.executable, "-m", "voxlens.serve",
            "--checkpoint", CHECKPOINT,
            "--port", "0",
            "--device", "cpu",
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        port = None
        for line in server.stderr:  # the listening line names the bound port
            if "listening on" in line:
                port = int(line.split("listening on")[1].split(",")[0].strip().split(":")[1])
                break
        assert port, "the server never reported a port"

        replayed = run_replay(CROP_CLIP, "--port", str(port))
        assert replayed.returncode == 0, replayed.stderr
    finally:
        server.terminate()
        server.wait(timeout=30)

    direct = subprocess.run(
        [
            sys.executable, "-m", "voxlens.cli", CROP_CLIP,
            "--checkpoint", CHECKPOINT, "--device", "cpu", "--pre-cropped",
        ],
        capture_output=True,
        text=True,
    )
    assert direct.returncode == 0, direct.stderr
    assert replayed.stdout.strip() == direct.stdout.strip()


def test_a_silent_session_is_closed_rather_than_held_open(serve):
    """A phone that vanishes without closing its socket — airplane mode, a
    crash — must not hold a thread until TCP notices hours later."""
    server = serve()
    server.idle_timeout = 0.3
    sock, rfile, wfile = raw_connection(server)
    try:
        send_message(wfile, {"type": "hello", "protocol": 1, "session": "quiet", "fps": 25.0})
        read_message(rfile)
        with pytest.raises(Disconnected):
            read_message(rfile)  # the server let go of a session going nowhere
    finally:
        sock.close()

    server.idle_timeout = 10
    with client_for(server, session="after") as client:
        assert client.send_crops(crops(5))["transcript"] == "5 crops"
