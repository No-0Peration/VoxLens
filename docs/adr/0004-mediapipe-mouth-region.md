# MediaPipe for Mouth Region extraction, pinned to 0.10.35

The Mouth Region is extracted with MediaPipe FaceLandmarker, using upstream's bundled task file and its 478→68 landmark mapping. Measured at 4.2 ms per Frame (RTF 0.111), essentially resolution-independent because MediaPipe resizes internally, and producing the 96×96 RGB crop the checkpoint expects without adaptation.

RetinaFace + FAN — the higher-accuracy alternative — **requires a CUDA GPU** per upstream's own documentation, so it is unavailable on the target hardware. That fact alone settles the choice.

## Consequences

- **The version is pinned exactly, not as a range.** MediaPipe 1.0.1 aborts fatally on macOS 26 / Apple Silicon inside its Metal path — a process kill, not a catchable exception — and forcing the CPU delegate does not avoid it. A routine dependency bump reintroduces a crash that looks nothing like a version problem.
- Extraction roughly **doubles** pipeline cost and is the largest component after the decoder. End-to-end is ~0.29 RTF measured, against the 0.21 implied by summing components.
- MediaPipe returning no landmarks for a Frame is the **Occlusion signal** (ADR-0005 territory, see issue #6). Choosing a detector that could not report per-Frame failure would have made Occlusion undetectable.

Detail: VoxLens issue #9.
