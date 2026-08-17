# Research

The primary-source work behind VoxLens's decisions: what can actually be obtained,
what its licence actually says, and what the numbers actually are.

These documents exist because the answers turned out to be different from what the
literature implies. Papers cite corpora that are no longer distributed, and quote
accuracy figures measured on data most people cannot legally hold. Every claim here
was checked against a primary source — licence text read in full, HTTP endpoints
probed, archives enumerated — and each document separates **what was verified** from
**what was inferred**.

They are dated by their commits and are not maintained. Treat them as a record of
what was true when the decisions were made, not as current fact.

## The documents

### [`pretrained-checkpoints.md`](pretrained-checkpoints.md) — which model can we build on?

Surveys every downloadable visual speech recognition checkpoint: licence, whether the
weights are genuinely obtainable, parameter count, reported video-only accuracy, and
the exact input preprocessing each expects.

The finding that shaped the project: **the licence question has two axes that are
usually conflated.** Non-academic use is fine. Commercial use is not available at all —
every candidate trains on LRS3, which is TED-derived under CC BY-NC-ND, and LRS3 is no
longer distributed, so retraining clean is not an escape hatch either.

Also contains the preprocessing contract the Mouth Region extractor has to satisfy,
read out of the inference code rather than the papers, because the papers do not state
it precisely enough to reproduce.

→ [ADR-0003](../adr/0003-usr2-large-non-commercial.md), [ADR-0004](../adr/0004-mediapipe-mouth-region.md)

### [`corpus-availability.md`](corpus-availability.md) — what data can we hold?

The question was whether an unconstrained-English corpus fits in 78 GB. It turned out
that **disk was never the constraint**: the evaluation data VoxLens needs totals under
300 MB. Licensing and availability are the real walls.

LRS3 is withdrawn from every endpoint (verified: 200 through January 2023, 404 since
April 2024). LRS2 excludes VoxLens by name — the BBC permits neither companies *nor*
independent researchers, requires an academic address and an organisation as
contracting party. What remains obtainable is evaluation-only.

→ [ADR-0005](../adr/0005-two-evaluation-bars.md)

### [`corpus-alternatives.md`](corpus-alternatives.md) — a correction, and everything else

A follow-on sweep that **overturned the previous document's central conclusion.** It
had concluded there was no route to training data for an unaffiliated individual;
MultiVSR (MIT-licensed metadata, ungated, ~12,000 hours) and the VoxPopuli CC0 route
are both candidates. Neither is turnkey, but "impossible" was too strong.

Also covers the corpora the first pass treated only briefly — GRID, Lombard GRID,
CREMA-D, RAVDESS, MEAD, TCD-TIMIT, VoxCeleb2, AVSpeech — with verified sizes and
licence terms, and notes which third-party mirrors are redistributing data whose
original terms did not permit it.

## What they concluded, in one line each

| Question | Answer |
| --- | --- |
| Which checkpoint? | USR 2.0 Large — obtainable, CC BY-NC, no academic gate |
| Commercial use? | No cleanly-licensed VSR checkpoint exists |
| Which corpora? | LRS3 test split (calibration) and WildVSR (honest), under 300 MB together |
| Training data? | Licence-clear routes exist but none is turnkey — an open problem |
| Does the disk limit bind? | No. Licensing does |

## Two things to read carefully

**The unresolved discrepancy.** Published figures put USR 2.0 at 73.7% WER on WildVSR;
this project measured 47.85% — *better*, using a smaller configuration, where it should
be worse. Most likely a text-normalisation difference. Until reconciled, VoxLens's
baselines are valid against themselves and are not claims of parity with published
work.

**The verification sections.** Each document ends by listing what it could not confirm
and what would be needed to close it. Those lists are the honest part; the confident
prose above them rests on them.
