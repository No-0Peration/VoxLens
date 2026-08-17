# Confidence from decoder disagreement

**Status:** proposed — decided, **not built**. This ADR is the **definition of done** for [#22](https://github.com/No-0Peration/VoxLens/issues/22): that work is complete when the code does what is described here, and this status becomes `accepted`. If building it shows the decision does not hold, amend this ADR rather than leaving it standing while the code says otherwise.

Per-sentence confidence is derived from how much the **CTC** and **beam-search** decoders disagree about the same Clip. Both read the same encoder output; agreement is evidence, divergence is doubt.

This reuses something previously treated only as an obstacle. The two decoders routinely produce different transcriptions — that is why [ADR-0008](0008-occlusion-as-spans.md) exists and why per-word Occlusion marking was abandoned. The divergence does not stop being inconvenient; it turns out to also be informative.

## Consequences

- **It must be validated before it is trusted.** Running the benchmark with both decoders and correlating divergence against measured per-clip WER takes minutes on data already on disk. Without that, this is a hunch wearing a number.
- **Confidence is per sentence, not per word.** Word-level would need forced alignment, which ADR-0008 records as unavailable.
- **Both decoders must run**, roughly doubling decode cost. At RTF 0.075 for beam search that is affordable.

## Considered options

**The beam hypothesis score.** Free and already returned, but uncalibrated: a high score means the model was not hesitating, not that it was right. Kept as the fallback if the correlation does not hold.

**Forced alignment for per-word confidence.** The better answer eventually, and a project of its own.
