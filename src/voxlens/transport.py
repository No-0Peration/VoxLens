"""The wire between a camera and the recogniser.

Live capture puts the camera on a phone and the model on a Mac (ADR-0009), so
Mouth Region crops have to cross a network. This module is that crossing and
nothing else: it knows the shape of a message and the crop contract a message
has to satisfy. It knows nothing about recognition.

What arrives on the far side is what ``voxlens --pre-cropped`` already takes —
crops in, text out — so the seam falls on an interface that exists and is
tested. The camera side of this protocol will be rewritten in Swift (#20); the
client here is what drives a server without one (``voxlens-replay``).

## The message

    4 bytes   big-endian unsigned   length of the header
    N bytes   UTF-8 JSON            the header, always carrying a "type"
    M bytes   raw crops             M is the header's "bytes"; absent means 0

One framing would have been simpler than this hybrid, and both alternatives
were worse. JSON alone, crops base64-encoded, costs a third more bandwidth and
an encode pass on the device holding the camera at 25 fps. Raw binary alone
makes every field a byte offset to be agreed twice, in Python and in Swift,
where a JSON header is read in an afternoon.
"""
from __future__ import annotations

import json
import socket
import struct
import uuid
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CLIP_MODE",
    "CROP_BYTES",
    "CROP_SHAPE",
    "CROP_SIZE",
    "CropClient",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "Disconnected",
    "MAX_CROPS_PER_MESSAGE",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "ServerError",
    "TARGET_FPS",
    "STREAM_MODE",
    "check_fps",
    "decode_crops",
    "encode_crops",
    "read_message",
    "send_message",
]

# Bumped when a message changes shape. The hello carries it so a phone built
# against an older protocol is refused with a sentence rather than dropping
# into a mismatched read.
PROTOCOL_VERSION = 1

# What a session is for. A Clip session answers one Transcript per batch of
# crops, which is what a corpus replay wants. A Stream session decodes in
# overlapping windows and revises its own recent text (ADR-0012), which is what
# a camera wants. Declared at hello because it changes what the replies mean.
CLIP_MODE = "clip"
STREAM_MODE = "stream"
MODES = (CLIP_MODE, STREAM_MODE)

# The crop contract, restated from extraction: 96x96 RGB uint8 at 25 fps. The
# recogniser centre-crops to 88 itself, so 96 is what crosses the wire.
CROP_SIZE = 96
CROP_SHAPE = (CROP_SIZE, CROP_SIZE, 3)
CROP_BYTES = CROP_SIZE * CROP_SIZE * 3
TARGET_FPS = 25.0

# A declared rate this far from 25 fps is refused rather than resampled: the
# checkpoint was trained at 25, and quietly accepting 30 would cost accuracy
# with nothing in the output to show why.
FPS_TOLERANCE = 0.5

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9601

# Bounds, so a corrupt length is a sentence rather than a 4 GB allocation.
MAX_HEADER_BYTES = 64 * 1024
MAX_CROPS_PER_MESSAGE = int(TARGET_FPS * 60)  # one minute of crops
MAX_PAYLOAD_BYTES = MAX_CROPS_PER_MESSAGE * CROP_BYTES

_LENGTH = struct.Struct("!I")


class ProtocolError(RuntimeError):
    """The peer sent something this protocol does not allow."""


class Disconnected(RuntimeError):
    """The peer closed the connection between messages.

    Kept distinct from ProtocolError because it is not a fault: a camera can
    leave at any time, and the only correct response is to end the session.
    """


class ServerError(RuntimeError):
    """The server answered with an error message.

    The code is kept separate from the prose so a caller can react to the kind
    of failure without matching on a sentence.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def send_message(stream, header: dict, payload: bytes = b"") -> None:
    """Write one message to a binary stream, header and payload together."""
    header = dict(header)
    if payload:
        header["bytes"] = len(payload)
    encoded = json.dumps(header).encode()
    if len(encoded) > MAX_HEADER_BYTES:
        raise ProtocolError(f"header of {len(encoded)} bytes exceeds the maximum")
    # One write: a header that reaches the peer ahead of its payload invites a
    # reader to act on a message it has not fully received.
    stream.write(_LENGTH.pack(len(encoded)) + encoded + payload)
    stream.flush()


def _read_exactly(stream, count: int, what: str) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ProtocolError(
                f"the connection ended inside a message {what}: "
                f"expected {count} bytes, got {count - remaining}"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message(stream) -> tuple[dict, bytes]:
    """Read one message from a binary stream.

    Raises Disconnected when the peer left cleanly between messages, and
    ProtocolError when what arrived is not a message.
    """
    raw = stream.read(_LENGTH.size)
    if not raw:
        raise Disconnected("the peer closed the connection")
    if len(raw) < _LENGTH.size:
        raw += _read_exactly(stream, _LENGTH.size - len(raw), "length")
    (length,) = _LENGTH.unpack(raw)
    if not 0 < length <= MAX_HEADER_BYTES:
        raise ProtocolError(
            f"header length {length} is outside 1..{MAX_HEADER_BYTES}. "
            "Is this a VoxLens client?"
        )

    try:
        header = json.loads(_read_exactly(stream, length, "header"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"the header is not JSON: {exc}") from exc
    if not isinstance(header, dict) or not isinstance(header.get("type"), str):
        raise ProtocolError("every message needs a JSON object header with a type")

    size = header.get("bytes", 0)
    if not isinstance(size, int) or isinstance(size, bool) or not 0 <= size <= MAX_PAYLOAD_BYTES:
        raise ProtocolError(
            f"payload size {size!r} is outside 0..{MAX_PAYLOAD_BYTES} bytes"
        )
    payload = _read_exactly(stream, size, "payload") if size else b""
    return header, payload


def encode_crops(crops: np.ndarray) -> bytes:
    """Mouth Region crops to wire bytes, refusing anything off-contract.

    Checked on the sending side as well as the receiving one: a phone that
    ships the wrong shape should hear about it where the bug is, not read a
    Transcript of noise.
    """
    array = np.asarray(crops)
    if array.ndim != 4 or array.shape[1:] != CROP_SHAPE:
        raise ValueError(
            f"crops must be (frames, {CROP_SIZE}, {CROP_SIZE}, 3); got {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(f"crops must be uint8 RGB; got {array.dtype}")
    if len(array) == 0:
        raise ValueError("no crops to send")
    if len(array) > MAX_CROPS_PER_MESSAGE:
        raise ValueError(
            f"{len(array)} crops exceeds {MAX_CROPS_PER_MESSAGE} in one message "
            f"({MAX_CROPS_PER_MESSAGE / TARGET_FPS:g}s). Send them in several."
        )
    return np.ascontiguousarray(array).tobytes()


def decode_crops(payload: bytes, frames: int) -> np.ndarray:
    """Wire bytes back to (frames, 96, 96, 3) uint8 RGB."""
    expected = frames * CROP_BYTES
    if len(payload) != expected:
        raise ProtocolError(
            f"{frames} crops need {expected} bytes; the message carried {len(payload)}"
        )
    # bytearray, not bytes: frombuffer over an immutable buffer yields a
    # read-only array, which torch.from_numpy then warns about downstream.
    return np.frombuffer(bytearray(payload), dtype=np.uint8).reshape(frames, *CROP_SHAPE)


def check_fps(fps: object) -> float:
    """Return a usable frame rate, or refuse it.

    The rate is declared once per session rather than inferred, because the
    only alternative is timing packet arrivals and calling the result a frame
    rate.
    """
    try:
        value = float(fps)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ProtocolError(f"fps must be a number; got {fps!r}") from None
    if not abs(value - TARGET_FPS) <= FPS_TOLERANCE:
        raise ProtocolError(
            f"crops must arrive at {TARGET_FPS:g} fps — the rate the checkpoint was "
            f"trained on — and this session declared {value:g}. Resample on the camera."
        )
    return value


@dataclass
class CropClient:
    """The camera side of the protocol, for anything that is not a phone.

    Strictly request and response: one Transcript comes back per batch of
    crops, in order. That is enough for a corpus replay and for #20's first
    version; overlapping windows and revision are #21 (ADR-0012).
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    session: str = ""
    fps: float = TARGET_FPS
    mode: str = CLIP_MODE
    timeout: float | None = 300.0
    ready: dict = field(default_factory=dict)

    _socket: socket.socket | None = field(default=None, repr=False)
    _rfile: object = field(default=None, repr=False)
    _wfile: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Session identity is the client's to choose, so a phone can keep one
        # across reconnects; a replay that does not care gets a random one.
        self.session = self.session or f"client-{uuid.uuid4().hex[:8]}"

    def open(self) -> dict:
        """Connect and say hello. Returns the server's ready message."""
        self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._rfile = self._socket.makefile("rb")
        self._wfile = self._socket.makefile("wb")
        send_message(
            self._wfile,
            {
                "type": "hello",
                "protocol": PROTOCOL_VERSION,
                "session": self.session,
                "fps": self.fps,
                "mode": self.mode,
            },
        )
        self.ready = self._expect("ready")
        return self.ready

    def send_crops(self, crops: np.ndarray) -> dict:
        """Send one batch of crops and wait for its Transcript."""
        if self._socket is None:
            raise RuntimeError("open() the client before sending crops")
        payload = encode_crops(crops)
        send_message(self._wfile, {"type": "crops", "frames": len(crops)}, payload)
        return self._expect("transcript")

    def send_occlusion(self, frames: int) -> dict:
        """Report Frames the camera could not read, rather than sending noise.

        The alternative is to send crops of whatever was in front of the lens,
        which the recogniser would read as mouth movement — the invented text
        this project exists not to produce. In a Stream session this also cuts
        the decoding window, since the signal is gone anyway (ADR-0012).
        """
        if self._socket is None:
            raise RuntimeError("open() the client before reporting an Occlusion")
        if frames <= 0:
            raise ValueError("an Occlusion covers at least one Frame")
        send_message(self._wfile, {"type": "occlusion", "frames": frames})
        return self._expect("transcript")

    def close(self) -> dict | None:
        """Leave the session cleanly. Safe to call on a connection already gone.

        A Stream session gets one last message back: the goodbye flushes the
        tail of the Stream, which no window had covered yet, and settles what
        was still provisional. That final reading is returned.
        """
        if self._socket is None:
            return None
        final = None
        try:
            send_message(self._wfile, {"type": "bye"})
            if self.mode == STREAM_MODE:
                final = self._expect("transcript")
        except (OSError, ValueError, ProtocolError, Disconnected, ServerError):
            pass  # the server left first; there is nothing to be polite about
        for handle in (self._rfile, self._wfile, self._socket):
            try:
                handle.close()  # type: ignore[union-attr]
            except OSError:
                pass
        self._socket = self._rfile = self._wfile = None
        return final

    def _expect(self, kind: str) -> dict:
        header, _ = read_message(self._rfile)
        if header["type"] == "error":
            raise ServerError(
                str(header.get("code", "error")), str(header.get("message", "no detail"))
            )
        if header["type"] != kind:
            raise ProtocolError(f"expected a {kind}, got {header['type']!r}")
        return header

    def __enter__(self) -> CropClient:
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
