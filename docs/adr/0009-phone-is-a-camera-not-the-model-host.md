# The phone is a camera, not the model host

**Status:** proposed — **half built**. The Mac half ([#19](https://github.com/No-0Peration/VoxLens/issues/19)) exists; the camera half ([#20](https://github.com/No-0Peration/VoxLens/issues/20)) does not, so this stays `proposed`: a decision about where extraction belongs is not proven by a server with no camera in front of it. This ADR is the **definition of done** for both tickets, and becomes `accepted` when a phone feeds it. If building it shows the decision does not hold, amend this ADR rather than leaving it standing while the code says otherwise.

For live capture, the phone detects the face and produces 96×96 Mouth Region crops; a Mac runs the recogniser and returns text. The phone does not run the model.

Crops rather than video: roughly 0.7 MB/s raw and far less compressed, against megabytes per second of frames. The Mac then does exactly what the CLI already does with `--pre-cropped` — crops in, text out — so the seam falls on an interface that already exists and is tested.

**Built so far (#19).** `voxlens-serve` loads the checkpoint once, before opening its socket, and serves sessions of 96×96 crops through the same recogniser call `voxlens --pre-cropped` makes — asserted by running one Clip through both and requiring identical text. The protocol carries a version, a session identity and a declared frame rate, and refuses a rate the checkpoint was not trained on rather than resampling behind the caller's back. `voxlens-replay` stands in for the phone by replaying crops from disk.

What that does **not** yet show is anything the camera brings: whether a hand-held phone can hold a mouth in frame at 5× zoom, and what the crops look like when it tries. Until #18 and #20, this ADR's central claim is plumbing that works, not a design that is proven.

## Consequences

- **The interesting problems land where they belong.** Zoom, face tracking and hand-shake are camera problems and stay on the camera. Nothing about them is easier for having first spent months on model conversion.
- **A network dependency is accepted.** No Mac, no transcript. For a proof of concept that is a fair trade; for anything else it would not be.
- **On-device inference stays possible but is not attempted.** The checkpoint is 4 GB and Core ML conversion is its own project. Keeping the encoder separable ([ADR-0002](0002-hybrid-encoder-gpu-search-cpu.md)) means that door stays open.

## Considered options

**Everything on device.** Rejected for now: it front-loads a conversion project of unknown difficulty ahead of every question that makes the idea interesting, and answers none of them.

**Send video, let the Mac find the face.** Rejected: an order of magnitude more bandwidth, and it moves tracking away from the device holding the lens.
