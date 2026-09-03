# Confidence from decoder disagreement

**Status:** accepted — **validated and adopted**, amended twice (on the unit, and on what it costs). The correlation this ADR demanded before trusting the idea has been run: Spearman **0.755** against per-Clip WER over 571 WildVSR Clips. Divergence is now presented as calibrated confidence, in measured bands.

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

### The validation, run

571 Clips of the WildVSR test split, stride 5, beam 1, hybrid device, both decoders
over the same encoder output:

| | |
| --- | --- |
| Spearman, divergence against per-Clip WER | **0.755** |
| Pearson | 0.748 |
| mean divergence | 0.376 |
| overall WER | 47.78% — against a recorded baseline of 47.85%, so nothing else moved |

Banded by divergence tercile:

| divergence | Clips | mean WER |
| --- | --- | --- |
| 0.00 – 0.25 | 190 | **27.6%** |
| 0.25 – 0.47 | 191 | 46.6% |
| 0.47 – 1.00 | 190 | **76.5%** |

**Decision: adopt.** Forty-nine points of WER separate the band where the decoders
agree from the band where they do not. A signal that moved WER from 30% to 34%
would not have earned a user-facing number; this one does.

**It is not clip length in disguise.** Divergence against reference length
correlates at −0.05, and restricted to Clips of eight words or more (n=528) the
relationship is unchanged at 0.756. That was the obvious confound — short Clips
being both harder and more divergent — and it is not what is happening.

**What "firm" does not mean.** Clips in the agreeing band still average 27.6% word
errors, and about one in thirty comes back word-perfect. The label says the two
decoders agreed, which is the best evidence available short of knowing. It does
not say the text is right.

The bands are the tercile boundaries above, used as thresholds because they were
measured rather than chosen.

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

*(An earlier note here said the corpus was unobtainable and the validation
therefore blocked. That was wrong: WildVSR is an 88 MB direct download with no
form and no agreement, as `docs/research/corpus-availability.md` had already
established. The measurement above is on WildVSR.)* A correlation that does not hold is a result, and the fallback it implies — the beam hypothesis score, uncalibrated — is already chosen below.

## Consequences

- **It must be validated before it is trusted.** Running the benchmark with both decoders and correlating divergence against measured per-clip WER takes minutes on data already on disk. Without that, this is a hunch wearing a number.
- **Confidence is per unit of speech, not per word** — a Clip, or a decoding window when live, per the amendment above. Word-level would need forced alignment, which ADR-0008 records as unavailable.
- **Both decoders must run**, and it is close to free — not the doubling this predicted. Measured RTF with both decoders is **0.097**, against 0.097–0.118 for beam search alone: the CTC head is one linear layer over encoder output and an argmax, while beam search is hundreds of sequential steps. Cost is therefore no longer the reason `--divergence` is opt-in, and making it the default is a decision left open rather than taken quietly.

## Considered options

**The beam hypothesis score.** Free and already returned, but uncalibrated: a high score means the model was not hesitating, not that it was right. Kept as the fallback if the correlation does not hold.

**Forced alignment for per-word confidence.** The better answer eventually, and a project of its own.
