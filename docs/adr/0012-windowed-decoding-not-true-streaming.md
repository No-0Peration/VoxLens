# Windowed decoding, not true streaming

**Status:** accepted — built, and **amended twice** where building measured something different from what this predicted (see the amendments). The decision itself held: three-second windows advancing one second, with a visible revision boundary and cuts at Occlusions, is what `voxlens-serve` now does in a Stream session.

Live capture decodes three-second windows advancing one second at a time, running roughly two seconds behind the speaker, with additional cuts at Occlusions. It does not attempt frame-by-frame streaming.

[ADR-0001](0001-clips-first-streaming-target.md) names real-time streaming as the destination. This is how that destination is approached without first replacing the model: true streaming needs a causal encoder, and this one is a transformer that attends across the whole input. Within a three-second window it may attend freely — the window is complete before it is decoded.

## Amendment: the boundary sits a second later than this said

The revision boundary cannot be placed at "the last two seconds", because words
have no timestamps. [ADR-0008](0008-occlusion-as-spans.md) records why alignment
is unavailable, and that absence resurfaces here: nothing in a window's text says
which part of it was the first second.

What the code *can* locate is where a new window's reading stops overlapping the
accumulated text — and that point is one advance older than the theoretical
limit. So the provisional edge is **the newest window, three seconds**, not two.
Text that will never change again therefore stays provisional for one extra
second before settling.

That is a real loss and a small one. It is also the honest boundary: placing it
at two seconds would mean guessing where inside a window's words the first second
ended, which is precisely the invented precision ADR-0008 rejected.

## Amendment: measured lag, which is better than this assumed

Measured on an M4 Pro, 20 seconds of Frames pushed at 25 fps through
`voxlens-serve` on the default hybrid device, 18 windows:

| | |
| --- | --- |
| decode lag, steady state | **0.25 – 0.37 s** per window |
| decode lag, mean | **0.38 s** |
| decode lag, worst | **1.44 s** — the first window, which pays for warm-up |
| aggregate cost | **≈ 0.3 RTF**, as predicted for ×3 overlap |

"Roughly two seconds behind" turns out to understate it. A word appears at the
provisional edge as soon as the window containing it completes, and windows
always end at the present moment — so text arrives about **0.3 to 1.3 seconds**
after it is spoken. It then takes three more seconds to freeze, per the
amendment above.

The prediction that overlap would not matter held exactly: three windows per
second of speech cost about 0.3 seconds of compute per second of speech.

*(Measured on synthetic Frames containing no mouth. Timing is timing — the
recogniser does the same work either way — but nothing here says anything about
accuracy, which needs [#18](https://github.com/No-0Peration/VoxLens/issues/18).)*

## Consequences

- **Two seconds of lag is accepted as normal, not as a defect.** Live captioning has always run behind and nobody minds.
- **The newest window is provisional and may be revised; earlier text freezes.** (Stated here as two seconds; measured as three — see the amendment.) Later windows see more context and often read an earlier moment better. Text rewriting itself under a reader's eyes is exhausting; freezing text that could immediately be improved is wasteful. The boundary between the two is visible.
- **Overlap triples the work and it does not matter.** Each second is processed three times; inference alone is RTF 0.096, so ×3 is 0.29 — comfortably real time with headroom for the network.
- **Whether the encoder is genuinely non-causal remains unmeasured.** This design does not depend on the answer, which is why it can proceed — but any future attempt at true low-latency streaming must start by measuring it.

## Considered options

**True streaming, sub-300 ms.** Requires a causal encoder and probably a different model. Not reachable with this checkpoint.

**Decode only when the speaker stops.** Simpler, and no longer live in any useful sense.
