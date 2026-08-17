# Occlusion is reported as spans, not marked per word

**Status:** accepted — decided and implemented, then amended once when building falsified part of it (see the amendment on [#6](https://github.com/No-0Peration/VoxLens/issues/6)).

When the Speaker's mouth cannot be read, VoxLens reports **time ranges** alongside the Transcript. It does not mark individual words as read or unreadable.

The original design did intend per-word marking, positioned using the CTC head's frame-level alignment — the encoder does not downsample, so alignment is exact at 40 ms. Building it falsified the approach: **the CTC head and beam search produce different transcriptions of the same Clip.** The alignment is precise; it simply aligns a transcription other than the one the CLI prints.

## Consequences

- **VoxLens cannot mislabel Read Text.** Marking the wrong words as invented is precisely the failure the Read Text / Inferred Text distinction exists to prevent, and reporting spans cannot commit it.
- The reader learns **when** the mouth was unreadable, not **which words** were affected. That is a genuine loss of precision, accepted knowingly.
- **The detection mask must be captured before interpolation.** Upstream interpolates across gaps and then asserts every Frame has landmarks, destroying the signal. Interpolation still runs — it keeps the tensor well-formed — but the mask survives beside it.
- Per-word marking remains reachable later via **forced alignment** of the beam transcript against encoder output. That is its own piece of work, not a refinement of this one.

## Considered options

**Per-word marking via CTC spikes.** Rejected on evidence, not principle — see above.

**Use the CTC transcription throughout**, making alignment consistent. Rejected: CTC greedy output is materially worse than beam search, so this trades accuracy for a labelling convenience.

**Report nothing about Occlusion.** Rejected: it makes VoxLens silently present invented mouth positions as though they were read, which contradicts what the project is for.
