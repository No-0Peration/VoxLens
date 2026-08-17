# Windowed decoding, not true streaming

**Status:** proposed — decided, **not built**. This ADR is the **definition of done** for [#21](https://github.com/No-0Peration/VoxLens/issues/21): that work is complete when the code does what is described here, and this status becomes `accepted`. If building it shows the decision does not hold, amend this ADR rather than leaving it standing while the code says otherwise.

Live capture decodes three-second windows advancing one second at a time, running roughly two seconds behind the speaker, with additional cuts at Occlusions. It does not attempt frame-by-frame streaming.

[ADR-0001](0001-clips-first-streaming-target.md) names real-time streaming as the destination. This is how that destination is approached without first replacing the model: true streaming needs a causal encoder, and this one is a transformer that attends across the whole input. Within a three-second window it may attend freely — the window is complete before it is decoded.

## Consequences

- **Two seconds of lag is accepted as normal, not as a defect.** Live captioning has always run behind and nobody minds.
- **The last two seconds are provisional and may be revised; earlier text freezes.** Later windows see more context and often read an earlier moment better. Text rewriting itself under a reader's eyes is exhausting; freezing text that could immediately be improved is wasteful. The boundary between the two is visible.
- **Overlap triples the work and it does not matter.** Each second is processed three times; inference alone is RTF 0.096, so ×3 is 0.29 — comfortably real time with headroom for the network.
- **Whether the encoder is genuinely non-causal remains unmeasured.** This design does not depend on the answer, which is why it can proceed — but any future attempt at true low-latency streaming must start by measuring it.

## Considered options

**True streaming, sub-300 ms.** Requires a causal encoder and probably a different model. Not reachable with this checkpoint.

**Decode only when the speaker stops.** Simpler, and no longer live in any useful sense.
