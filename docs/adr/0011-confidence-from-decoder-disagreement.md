# Confidence from decoder disagreement

**Status:** proposed — **measurable, not adopted**, and amended once on what the unit of confidence can be (see the amendment). Divergence between the decoders is now computed and reported; whether it predicts being wrong is unmeasured, so nothing is presented as confidence. This ADR is the **definition of done** for [#22](https://github.com/No-0Peration/VoxLens/issues/22) and becomes `accepted` when the correlation has been run and a decision recorded against it. If building it shows the decision does not hold, amend this ADR rather than leaving it standing while the code says otherwise.

Confidence for one unit of speech — stated here as a sentence, amended below to the Clip — is derived from how much the **CTC** and **beam-search** decoders disagree about it. Both read the same encoder output; agreement is evidence, divergence is doubt.

This reuses something previously treated only as an obstacle. The two decoders routinely produce different transcriptions — that is why [ADR-0008](0008-occlusion-as-spans.md) exists and why per-word Occlusion marking was abandoned. The divergence does not stop being inconvenient; it turns out to also be informative.

## Amendment: per Clip, not per sentence (#22)

**This model emits no sentences.** The 1000-unigram vocabulary contains no punctuation, and neither corpus's references carry any, so a Transcript is an unpunctuated run of 3 to 20 words with no boundary anywhere in it. There is nothing to attach a per-sentence number to.

The unit is therefore the **Clip**, and for live capture the **decoding window** ([#21](https://github.com/No-0Peration/VoxLens/issues/21), ADR-0012) — which is the same thing seen live: a bounded stretch of speech decoded as one unit. That is a genuine loss against the original wording. It is also the finest grain available without forced alignment, which ADR-0008 already records as out of reach.

**What is exposed, and what is withheld.** `voxlens --divergence` reports the disagreement as a measured quantity, marked `calibrated: false`, alongside the CTC head's own reading so a reader can judge it. What it does not do is label anything "uncertain": a label needs a threshold, and the threshold is precisely what the validation below would establish. Reporting the number is honest; reporting a verdict before the measurement would be the hunch wearing a number that this ADR warns about.

## The measurement, when it is run

```bash
voxlens-eval "$VOXLENS_CORPUS" --corpus lrs3 --checkpoint "$VOXLENS_CHECKPOINT" \
  --divergence --out .confidence-results.json
```

Read `spearman` first — per-Clip WER is unbounded above, so a monotone relationship matters more than a linear one — then the buckets, which answer the question a decision actually rests on: when the decoders agree, how much better is the text, in points of WER? A coefficient that moves WER from 30% to 34% does not earn a user-facing number.

### A first data point, which is not the validation

Run on 60 Frames of synthetic texture containing no mouth at all — the fixture
`scripts/make_crop_clip.py --synthetic` produces for testing plumbing — the two
decoders behaved as this ADR hopes they would, at the extreme:

| decoder | reading |
| --- | --- |
| beam search | *"i am because i am biracial they try to figure out if i'm"* |
| CTC head | *(nothing)* |

Divergence 1.00. The beam search produced eleven fluent words from an input with
no speech in it, and the CTC head declined to read anything. In the one case
where the output is certainly worthless, the disagreement was total.

That is encouraging and it proves nothing about the correlation. It is a single
clip, the input contains no speech, and a signal that fires on pure noise may
say nothing about a real Clip read badly. It is recorded because it is also a
plain demonstration of what [ADR-0010](0010-read-on-screen-never-aloud.md) is
about: a confident, grammatical, entirely invented sentence.

Note also that the checkpoint alone does not unlock the validation — that needs a
corpus with references, and both are gated or no longer distributed (ADR-0005).
The blocker is the corpus, not the weights.

Record the outcome here, either way. A correlation that does not hold is a result, and the fallback it implies — the beam hypothesis score, uncalibrated — is already chosen below.

## Consequences

- **It must be validated before it is trusted.** Running the benchmark with both decoders and correlating divergence against measured per-clip WER takes minutes on data already on disk. Without that, this is a hunch wearing a number.
- **Confidence is per unit of speech, not per word** — a Clip, or a decoding window when live, per the amendment above. Word-level would need forced alignment, which ADR-0008 records as unavailable.
- **Both decoders must run**, roughly doubling decode cost. At RTF 0.075 for beam search that is affordable.

## Considered options

**The beam hypothesis score.** Free and already returned, but uncalibrated: a high score means the model was not hesitating, not that it was right. Kept as the fallback if the correlation does not hold.

**Forced alignment for per-word confidence.** The better answer eventually, and a project of its own.
