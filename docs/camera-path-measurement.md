# Measuring what the camera path costs

**Not yet run.** This is the procedure and the tooling; the number it produces
does not exist yet. See [#18](https://github.com/No-0Peration/VoxLens/issues/18).

VoxLens reads a video file at **47.9% WER** on WildVSR. Live capture puts a lens,
a distance, a hand and a video codec between the speaker and that pipeline. Nobody
can estimate what those cost. If 48% becomes 75%, the design in
[live-capture.md](live-capture.md) is worth less than the effort of building it,
and that is worth knowing before the effort rather than after.

The trick that makes this measurable: **play Clips whose reference text is already
known on a monitor, and film the monitor.** Ground truth comes free, and the number
that comes out includes everything the camera path adds.

## What you need

A monitor, a tripod, a phone, a tape measure, and an afternoon. No Swift, no app,
nothing that does not already exist.

## The procedure

**Set up once, and record what you set.** Every one of these changes the answer,
so a run that does not record them cannot be repeated or compared.

| | |
| --- | --- |
| Monitor | model, size, and **brightness setting** — write down the number, not "bright" |
| Room light | overhead on or off, curtains, time of day. Aim for what a real room looks like, not a studio |
| Phone | model, and which lens is selected (1×, 3×, 5×) |
| Tripod | height set so the mouth is at the centre of frame, phone level, not tilted |
| Distance | tape-measured from lens to screen, not paced |
| Video | 1080p or 4K, 25 or 30 fps, and whether stabilisation was on |

**Play the Clips full-screen with nothing else on the display.** A face filling a
27-inch monitor is not the same target as one in a window, and the resolution table
in [live-capture.md](live-capture.md) is about the mouth's size in pixels.

**Film at three distances at least, and pick one you expect to fail.** The table
predicts the mouth falls below usable size somewhere between 8 and 12 m on a 5×
lens. Confirming where that boundary actually sits is more valuable than three
distances that all work. A reasonable set: **2 m, 5 m, 8 m, 12 m**.

**Lock focus and exposure** before each run, on the mouth. Autofocus hunting mid-Clip
is a variable nobody wants in the number.

**Twenty Clips per distance is enough** to see a difference of ten points, and few
enough to shoot in an afternoon. Use the same twenty at every distance, so a change
in the number is the distance and not the Clips.

## Naming, so the harness can group the results

One file per Clip, named `<distance>_<clip id>.mp4`:

```
2m_00001.mp4   5m_00001.mp4   8m_00001.mp4   12m_00001.mp4
```

Everything before the first underscore is the group. The harness reports WER per
group, because a single mean across every distance would hide the boundary this
measurement exists to find.

Alongside them, a `references.tsv` — one line per Clip, filename and the words
actually spoken, separated by a tab:

```
2m_00001.mp4	this is the only part that's battery backed up
5m_00001.mp4	this is the only part that's battery backed up
```

The references are the ones the corpus already ships; copy them rather than
transcribing by ear.

## Scoring it

```bash
voxlens-eval /path/to/captured --corpus captured --checkpoint "$VOXLENS_CHECKPOINT" \
  --out camera-path.json
```

`--corpus captured` is the one corpus that is **not** pre-cropped: this footage has
whole faces in it, so MediaPipe extraction runs, exactly as it would on a phone.
That is the point — extraction failing at distance is part of what is being measured,
and a run that skipped it would flatter the result.

Then score the same Clips fed directly, for the comparison that matters:

```bash
voxlens-eval /path/to/corpus --corpus lrs3 --checkpoint "$VOXLENS_CHECKPOINT" \
  --limit 20 --out direct.json
```

## What to write down

Into [live-capture.md](live-capture.md), because it is the number that design hangs
on:

| distance | lens | clips | WER through the camera | WER fed directly | difference |
| --- | --- | --- | --- | --- | --- |
| 2 m | | | | | |
| 5 m | | | | | |
| 8 m | | | | | |
| 12 m | | | | | |

And, separately, **how many Clips produced no Transcript at all** — the harness
reports them as excluded rather than scoring them as failures. At distance, "the
face was never found" is a different failure from "the words were wrong", and the
second number matters as much as the first.

**Record anything that contradicts the design rather than smoothing it over.** If
the mouth stays readable at 12 m, the resolution table is wrong and that is worth
more than a confirmation. If extraction fails at 5 m, the whole seminar premise
goes with it.
