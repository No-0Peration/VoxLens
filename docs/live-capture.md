# Live capture — design

Tickets: [#18](https://github.com/No-0Peration/VoxLens/issues/18) measurement · [#19](https://github.com/No-0Peration/VoxLens/issues/19) transport · [#20](https://github.com/No-0Peration/VoxLens/issues/20) iOS app · [#21](https://github.com/No-0Peration/VoxLens/issues/21) windowing · [#22](https://github.com/No-0Peration/VoxLens/issues/22) confidence.
Decisions: [ADR-0009](adr/0009-phone-is-a-camera-not-the-model-host.md), [ADR-0010](adr/0010-read-on-screen-never-aloud.md), [ADR-0011](adr/0011-confidence-from-decoder-disagreement.md), [ADR-0012](adr/0012-windowed-decoding-not-true-streaming.md).

**Designed, not built.** Nothing described here exists yet. This records a design
worked out in full before implementation, so the reasoning survives even if the
work stalls.

The idea: point a phone at someone speaking, and read what they are saying on the
screen. The phone is the camera and does the cropping; a Mac runs the recogniser.

## What this is for, and what it is not

**Open question, deliberately recorded as open.** The clearest use — a deaf person
following a speaker at a seminar — is not one this beats. The audio exists at a
seminar; the user simply cannot hear it. A phone microphone with iOS Live Captions
transcribes that audio at roughly 5–10% word errors. VoxLens manages 48%.

**For that scenario this is five to ten times worse than what is already free on the
device.** Lip reading only wins where audio is absent or unusable: behind glass,
beyond microphone range but within sight, a muted video call, extreme noise.

Until a scenario like that is named, this is a technical exercise — a legitimate
thing to build, and not something to present to deaf users as an aid. Recorded here
so it is not quietly forgotten once something works.

## Physical limit: distance

The recogniser wants a 96×96 mouth crop. What the lens can actually resolve, for an
8 cm mouth region on a 12 MP sensor:

| distance | main 1× | tele 3× | tele 5× |
| --- | --- | --- | --- |
| 2 m | 133 px | 351 px | 542 px |
| 5 m | 53 px | 140 px | 217 px |
| 8 m | 33 px | 88 px | 135 px |
| 12 m | 22 px | 59 px | 90 px |
| 20 m | 13 px | 35 px | 54 px |

**Front rows and meeting rooms, not auditoriums.** Below roughly 60 px there is no
detail left to upscale — the information never reached the sensor. An auditorium seat
is out of reach on any current phone, and no amount of model work changes that.

## The design

**Consent.** The subject is someone addressing the user — a speaker at a seminar, a
conversation partner. Not a stranger across a room. The user taps a face to select
it, which is an explicit act aimed at one person rather than ambient capture.

**Output is text only, never spoken aloud.** A synthetic voice reading a 48%-accurate
transcript sounds confident and fluent while being wrong, which removes the very
signal that warns the user. Errors here are grammatical and plausible — *"of those
seven have nuclear weapons"* was read as *"if you use seven if you use a weapon"*.

**Everything is shown, with uncertainty marked per sentence.** Not per word: word-level
confidence would need forced alignment, which [ADR-0008](adr/0008-occlusion-as-spans.md)
records as unavailable.

**Confidence comes from CTC/beam disagreement.** The encoder feeds two decoders that
routinely produce different transcriptions of the same clip. That divergence has been
treated as a problem — it is why ADR-0008 exists — but it is also a signal: two
decoders agreeing is evidence, two decoders disagreeing is doubt. It must be validated
against measured WER before it is trusted.

**The phone extracts, the Mac recognises.** MediaPipe runs natively on iOS, so the
phone sends 96×96 mouth crops rather than video — roughly 0.7 MB/s raw, far less
compressed. This puts zoom, tracking and hand-shake on the phone where they belong,
and leaves the Mac doing exactly what it already does: crops in, text out, the same
interface as `--pre-cropped`.

On-device inference is an optimisation for later, not a precondition. The checkpoint
is 4 GB and Core ML conversion is a project of its own; doing it first would delay
every interesting question by months.

**Windowed decoding, about two seconds behind.** Three-second windows advancing one
second at a time, with cuts at Occlusions where signal is lost anyway. Overlapping
windows process each second three times: inference alone is RTF 0.096, so ×3 is 0.29 —
comfortably real time, with headroom for network latency.

Two seconds of lag is what live captioning already does and nobody minds.

**The last two seconds may be revised; everything before freezes.** Later windows see
more context and often read an earlier moment better. Rewriting text under a reader's
eyes is exhausting, and freezing text you could immediately improve is wasteful — so
there is a visible provisional edge, and a stable body behind it.

**Recording is opt-in, off by default, local, and visible in the UI.** Saved mouth
crops paired with what was said are biometric imagery plus speech content. For the
author's own test material this changes nothing; for anyone else it is the difference
between an experiment and an archive of other people's conversations.

## Measuring it

Live output cannot be scored — there is no reference. Two things preserve measurement:

**Play known clips on a monitor and point the phone at it.** An LRS3 test clip on a
screen, a tripod, and the existing CLI gives a real end-to-end number *including* the
lens, the distance, the shake and the compression, with ground truth for free.

**This is the number the whole idea rests on**, and nobody can guess it: how much WER
does the camera path add on top of the 48% the model already costs? If the answer is
"48% becomes 75%", it changes what is worth building. It needs a monitor, a tripod and
an afternoon — no Swift at all.

**Then record crops for later scoring**, opt-in, so real sessions can be reviewed
rather than judged by impression.

## Licensing

This stays a proof of concept and is not distributed. The recogniser is CC BY-NC 4.0
covering code and weights; shipping an app would require resolving a licence chain
that currently has no resolution. See [NOTICE](../NOTICE).
