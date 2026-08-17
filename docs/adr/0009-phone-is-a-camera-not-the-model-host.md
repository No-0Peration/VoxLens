# The phone is a camera, not the model host

For live capture, the phone detects the face and produces 96×96 Mouth Region crops; a Mac runs the recogniser and returns text. The phone does not run the model.

Crops rather than video: roughly 0.7 MB/s raw and far less compressed, against megabytes per second of frames. The Mac then does exactly what the CLI already does with `--pre-cropped` — crops in, text out — so the seam falls on an interface that already exists and is tested.

## Consequences

- **The interesting problems land where they belong.** Zoom, face tracking and hand-shake are camera problems and stay on the camera. Nothing about them is easier for having first spent months on model conversion.
- **A network dependency is accepted.** No Mac, no transcript. For a proof of concept that is a fair trade; for anything else it would not be.
- **On-device inference stays possible but is not attempted.** The checkpoint is 4 GB and Core ML conversion is its own project. Keeping the encoder separable ([ADR-0002](0002-hybrid-encoder-gpu-search-cpu.md)) means that door stays open.

## Considered options

**Everything on device.** Rejected for now: it front-loads a conversion project of unknown difficulty ahead of every question that makes the idea interesting, and answers none of them.

**Send video, let the Mac find the face.** Rejected: an order of magnitude more bandwidth, and it moves tracking away from the device holding the lens.
