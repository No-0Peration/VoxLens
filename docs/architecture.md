# Architecture

VoxLens turns a video Clip into a Transcript by reading the Speaker's lips. No audio is used at inference. This document describes the shape of the system and why the boundaries fall where they do; the decisions themselves are recorded in [`adr/`](adr/), and the evidence behind them in [`research/`](research/).

## The pipeline

```
  video file
      │
      ▼
┌─────────────────┐
│ decode          │  OpenCV → Frames (RGB)                          CPU
└────────┬────────┘
         ▼
┌─────────────────┐
│ extract         │  MediaPipe FaceLandmarker → 68 landmarks/Frame  CPU
│                 │  · no face found ⇒ None for that Frame  ────────────┐
│                 │  · warp to mean face, crop about the mouth          │
│                 │  → Mouth Region, 96×96 RGB @ 25 fps                 │
└────────┬────────┘                                                     │
         ▼                                                              │
┌─────────────────┐                                          detection mask
│ normalise       │  centre 88×88, greyscale, mean .421 std .165   CPU  │
└────────┬────────┘                                                     │
         ▼                                                              │
┌─────────────────┐                                                     │
│ encode          │  ResNet + transformer, 1 step per Frame        GPU  │
└────────┬────────┘                                                     │
         ▼                                                              │
┌─────────────────┐                                                     │
│ decode          │  beam search (default beam 1)                  CPU  │
└────────┬────────┘                                                     │
         ▼                                                              ▼
┌───────────────────────────────────────────────────────────────────────┐
│ assemble        │  Transcript + Occlusion spans                       │
└────────┬────────────────────────────────────────────────────────────┬─┘
         ▼                                                            ▼
    text on stdout                                          --json on stdout
                                                                      │
                                                     evaluation harness
```

Two things in that diagram carry most of the design.

**The detection mask bypasses the recogniser.** Whether a Frame was readable is decided during extraction and travels *around* the model to the output stage. It has to: the model is fed interpolated landmarks regardless, so by the time anything reaches the encoder the gap is invisible. See [ADR-0008](adr/0008-occlusion-as-spans.md).

**Encoding runs on the GPU and everything else on the CPU.** Not because the rest could not, but because it is measurably slower there. See [ADR-0002](adr/0002-hybrid-encoder-gpu-search-cpu.md).

## Modules

| Module | Owns | Deliberately does not own |
| --- | --- | --- |
| `voxlens.devices` | Which device each stage runs on | Anything about models |
| `voxlens.upstream` | The vendored recogniser: locating, loading, the pinned revision | Video, extraction, output |
| extraction | Frames → Mouth Region crops + detection mask | What "too short to matter" means |
| occlusion | Detection mask → spans (pure function) | Detecting anything itself |
| CLI | Argument surface, orchestration, rendering, exit codes | Any recognition logic |
| harness | Running the CLI over a corpus, scoring | Reaching into internals |

The recogniser itself is **not** a VoxLens module. It is vendored upstream code, reached only through `voxlens.upstream`, which is the single place aware that `vendor/` exists.

**The encoder is kept separable from the decoder.** It is the streaming-critical component and the natural unit for a later Core ML conversion; fusing them would forfeit both. This is a standing constraint, not an implementation detail.

## Where the time goes

Measured on an M4 Pro, USR 2.0 Large, beam 1, video-only:

| stage | RTF | share |
| --- | --- | --- |
| extraction | 0.111 | ~38% |
| encoding | 0.021 | ~7% |
| decoding (beam search) | 0.075 | ~26% |
| decode, assembly, overhead | ~0.08 | ~29% |
| **end to end** | **~0.29** | 3.4× faster than real time |

The counter-intuitive result is that **recognition is cheap and everything around it is not**. The encoder — the part that looks expensive — is 7% of the budget. Extraction is the largest single component, and it is essentially resolution-independent, so it will not shrink by feeding smaller video.

Beam width dominates when raised: at beam 40 the decoder alone is 3.45 RTF, making the pipeline 4× slower than real time. Beam 1 is the default for that reason, and beam 10 is strictly dominated — same accuracy as beam 1, seven times slower.

## Testing shape

Two seams: the **CLI process boundary**, and the **occlusion span function**. The evaluation harness goes through the CLI rather than around it, so measurements describe the shipped path. See [ADR-0007](adr/0007-the-cli-is-the-seam.md).

## What is deliberately absent

- **Streaming.** The destination ([ADR-0001](adr/0001-clips-first-streaming-target.md)), but Clips come first. Nothing here should assume a bounded input where a Stream would not have one.
- **Inferred Text.** Occlusion is detected and marked, never filled. The concept is defined in `CONTEXT.md` and unimplemented on purpose.
- **Fine-tuning.** No obtainable unconstrained-English training corpus has been identified.
- **A model abstraction layer.** There is one checkpoint. An interface to swap recognisers would be generality nothing has asked for.

## Decision index

| ADR | Decision |
| --- | --- |
| [0001](adr/0001-clips-first-streaming-target.md) | Clips first, streaming as the target |
| [0002](adr/0002-hybrid-encoder-gpu-search-cpu.md) | Encoder on the GPU, beam search on the CPU |
| [0003](adr/0003-usr2-large-non-commercial.md) | USR 2.0 Large, and non-commercial research |
| [0004](adr/0004-mediapipe-mouth-region.md) | MediaPipe for extraction, pinned to 0.10.35 |
| [0005](adr/0005-two-evaluation-bars.md) | Two evaluation bars, WildVSR authoritative |
| [0006](adr/0006-vendor-upstream-with-patches.md) | Vendor upstream at a pinned revision, with patches |
| [0007](adr/0007-the-cli-is-the-seam.md) | The CLI is the seam |
| [0008](adr/0008-occlusion-as-spans.md) | Occlusion as spans, not per-word |
