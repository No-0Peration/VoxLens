"""The recogniser behind a socket.

Live capture keeps the model on the Mac and the camera on the phone
(ADR-0009). This is the Mac half: a process that loads the checkpoint once,
accepts Mouth Region crops over the network, and answers with Transcripts. It
does exactly what ``voxlens --pre-cropped`` does per batch of crops — the same
recogniser call, on the same crop contract — so the network buys a camera
somewhere else, not a second recognition path.

A third seam. ADR-0007 puts the tests at the CLI process boundary; this server
is a second process boundary, and its tests drive it over a real socket for
the same reason. The recogniser reaches it as a callable rather than being
loaded inside it, which is what lets those tests exercise every message on the
wire without a 4 GB checkpoint. The shipped path passes
``Recogniser.transcribe`` and nothing about the protocol differs.

Inference is serialised across sessions. There is one model on one GPU, so two
concurrent decodes would contend for it and gain nothing; a second camera
waits rather than halving the speed of the first.
"""
from __future__ import annotations

import argparse
import socketserver
import sys
import threading
import time
from dataclasses import dataclass, field

from voxlens.cli import EXIT_BAD_INPUT, EXIT_MODEL, EXIT_OK
from voxlens.devices import DEFAULT_DEVICE, resolve_device
from voxlens.result import Timing, checkpoint_identity
from voxlens.transport import (
    CROP_SIZE,
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAX_CROPS_PER_MESSAGE,
    PROTOCOL_VERSION,
    TARGET_FPS,
    Disconnected,
    ProtocolError,
    check_fps,
    decode_crops,
    read_message,
    send_message,
)

__all__ = ["CropServer", "Session", "main"]

# A session identity is echoed into every reply and every log line, so it is
# bounded and kept to printable characters rather than trusted as sent.
MAX_SESSION_ID = 64

# A phone that vanishes without closing its socket — airplane mode, a crash,
# a walk out of range — would otherwise hold a thread and a half-open
# connection until TCP noticed, which can be hours. Generous, because a
# session may legitimately go quiet between utterances.
IDLE_TIMEOUT = 300.0


@dataclass
class Session:
    """One camera's connection, for as long as it lasts.

    All of a session's state lives here, on the connection's own thread. That
    is what makes a disconnect uneventful: the crops, the counters and the
    declared frame rate go away with the socket, and the shared recogniser is
    left exactly as the session found it.
    """

    id: str
    fps: float
    frames: int = 0
    requests: int = 0
    started: float = field(default_factory=time.monotonic)


class CropServer(socketserver.ThreadingTCPServer):
    """Accepts crops, returns Transcripts, for as many sessions as connect."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        transcribe,
        *,
        device: str = "",
        beam: int = 0,
        checkpoint: dict | None = None,
        log=None,
        idle_timeout: float | None = IDLE_TIMEOUT,
    ) -> None:
        self.transcribe = transcribe
        self.device = device
        self.beam = beam
        self.checkpoint = checkpoint or {}
        self.lock = threading.Lock()
        self.idle_timeout = idle_timeout
        self.log = log if log is not None else _log_to_stderr
        super().__init__(address, _Handler)

    @property
    def port(self) -> int:
        """The bound port, which is what --port 0 makes worth asking for."""
        return int(self.server_address[1])


def _log_to_stderr(message: str) -> None:
    print(f"voxlens-serve: {message}", file=sys.stderr, flush=True)


class _Handler(socketserver.StreamRequestHandler):
    def setup(self) -> None:
        # Read before the base class applies it to the socket.
        self.timeout = self.server.idle_timeout
        super().setup()

    def handle(self) -> None:
        session: Session | None = None
        try:
            while True:
                header, payload = read_message(self.rfile)
                kind = header["type"]
                if session is None:
                    if kind != "hello":
                        raise ProtocolError(
                            f"the first message must be a hello, not {kind!r}"
                        )
                    session = self._begin(header)
                elif kind == "crops":
                    self._transcribe(session, header, payload)
                elif kind == "bye":
                    self._finish(session, "left")
                    return
                elif kind == "hello":
                    raise ProtocolError(
                        "this connection already has a session; one session per connection"
                    )
                else:
                    raise ProtocolError(f"unknown message type {kind!r}")
        except Disconnected:
            # The camera went away mid-session. Nothing to unwind: session
            # state belongs to this connection, and the recogniser holds no
            # state between calls, so the next session finds it intact.
            self._finish(session, "disconnected")
        except TimeoutError:
            self._finish(session, f"timed out after {self.timeout:g}s of silence")
        except OSError as exc:
            # The same event seen from the writing side: the socket died while
            # a reply was going out. A phone in a pocket is not a server fault,
            # so it ends the session rather than raising a traceback.
            self._finish(session, f"disconnected while replying ({exc})")
        except ProtocolError as exc:
            self._fail(session, "protocol", str(exc))

    def _begin(self, header: dict) -> Session:
        version = header.get("protocol")
        if version != PROTOCOL_VERSION:
            raise ProtocolError(
                f"this server speaks protocol {PROTOCOL_VERSION}; the client "
                f"announced {version!r}"
            )

        identity = header.get("session")
        if not isinstance(identity, str) or not identity.strip():
            raise ProtocolError("a hello must name a session")
        if len(identity) > MAX_SESSION_ID or not identity.isprintable():
            raise ProtocolError(
                f"a session identity must be printable and at most "
                f"{MAX_SESSION_ID} characters"
            )
        fps = check_fps(header.get("fps"))

        session = Session(id=identity, fps=fps)
        self.server.log(f"session {session.id} opened from {self.client_address[0]}")
        send_message(
            self.wfile,
            {
                "type": "ready",
                "protocol": PROTOCOL_VERSION,
                "session": session.id,
                # The contract, stated back rather than assumed shared: a
                # client that gets crops wrong should find out at hello.
                "crop": {"size": CROP_SIZE, "channels": 3, "dtype": "uint8", "layout": "RGB"},
                "fps": session.fps,
                "max_frames_per_message": MAX_CROPS_PER_MESSAGE,
                "config": {
                    "device": self.server.device,
                    "beam": self.server.beam,
                    "checkpoint": self.server.checkpoint,
                },
            },
        )
        return session

    def _transcribe(self, session: Session, header: dict, payload: bytes) -> None:
        frames = header.get("frames")
        if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
            raise ProtocolError(f"a crops message must count its frames; got {frames!r}")
        if frames > MAX_CROPS_PER_MESSAGE:
            raise ProtocolError(
                f"{frames} crops exceeds {MAX_CROPS_PER_MESSAGE} in one message"
            )
        crops = decode_crops(payload, frames)

        started = time.perf_counter()
        try:
            with self.server.lock:
                transcript = self.server.transcribe(crops)
        except Exception as exc:
            # The client is told, then the traceback is left to surface: a
            # broken model must not read as a quiet empty Transcript, and it
            # must not take the process down for every other session either.
            self._fail(session, "model", f"{type(exc).__name__}: {exc}")
            raise
        infer_s = time.perf_counter() - started

        session.requests += 1
        session.frames += frames
        duration_s = frames / session.fps
        # extract_s is zero because extraction happened on the camera, which is
        # the whole point of sending crops rather than video.
        timing = Timing(extract_s=0.0, infer_s=infer_s, duration_s=duration_s)
        send_message(
            self.wfile,
            {
                "type": "transcript",
                "session": session.id,
                # Numbered by the server, which is the side keeping the
                # session, and matched to requests by order: one Transcript per
                # crops message, in the order they arrived.
                "sequence": session.requests,
                "frames": frames,
                "duration_s": round(duration_s, 3),
                "transcript": transcript,
                "timing": timing.as_dict(),
            },
        )

    def _finish(self, session: Session | None, how: str) -> None:
        if session is None:
            self.server.log(f"a client {how} before opening a session")
            return
        self.server.log(
            f"session {session.id} {how} after {session.requests} request(s), "
            f"{session.frames} frame(s), {time.monotonic() - session.started:.1f}s"
        )

    def _fail(self, session: Session | None, code: str, message: str) -> None:
        where = f"session {session.id}" if session else "a client"
        self.server.log(f"{where}: {code}: {message}")
        try:
            send_message(self.wfile, {"type": "error", "code": code, "message": message})
        except (OSError, ProtocolError):
            pass  # the peer is already gone; the log line is the record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxlens-serve",
        description="Serve the recogniser over a socket: Mouth Region crops in, "
        "Transcripts out. The camera lives elsewhere.",
        epilog=(
            "exit codes:\n"
            "  0  the server ran and was stopped\n"
            "  2  the invocation is wrong: an unavailable device, or a taken port\n"
            "  3  the checkpoint is missing, unreadable, or the wrong architecture"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="path to the USR 2.0 Large checkpoint, loaded once at startup",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="interface to listen on; the default is loopback only, so nothing "
        "is exposed to a network until you ask for it (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="0 binds any free port and reports it (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["hybrid", "mps", "cpu"],
        help="as for the voxlens command (default: %(default)s)",
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=1,
        help="beam width, fixed for the life of the process (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        plan = resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        print(f"voxlens-serve: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    from voxlens.recogniser import CheckpointError, load_recogniser

    # Loaded before the socket opens, not on the first message: a client that
    # connects has a recogniser waiting for it, and a bad checkpoint fails at
    # startup where someone is watching rather than mid-session.
    try:
        recogniser = load_recogniser(args.checkpoint, plan, beam=args.beam)
    except CheckpointError as exc:
        print(f"voxlens-serve: {exc}", file=sys.stderr)
        return EXIT_MODEL

    try:
        server = CropServer(
            (args.host, args.port),
            recogniser.transcribe,
            device=plan.name,
            beam=args.beam,
            checkpoint=checkpoint_identity(args.checkpoint),
        )
    except OSError as exc:
        print(f"voxlens-serve: cannot listen on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT

    with server:
        _log_to_stderr(
            f"listening on {args.host}:{server.port}, device {plan.name}, "
            f"beam {args.beam}, {TARGET_FPS:g} fps"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            _log_to_stderr("stopped")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
