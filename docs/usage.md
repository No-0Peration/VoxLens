# Using VoxLens

VoxLens reads speech off a speaker's lips. You give it a video; it gives you text.
No audio is used at any point — mute the file and nothing changes.

Install first: [`setup.md`](setup.md). It is not just `pip install`, because the
recogniser VoxLens builds on does not run on Apple Silicon unpatched.

## You need a checkpoint

VoxLens never downloads model weights for you — they are ~4 GB and carry their own
licence. Get **USR 2.0 Large** (the high-resource, fine-tuned one) from
[`ahaliassos/usr2`](https://github.com/ahaliassos/usr2) and keep the path handy.

```bash
export VOXLENS_CHECKPOINT=/path/to/usr2_large.pth
```

## Reading a video

```bash
voxlens interview.mp4 --checkpoint "$VOXLENS_CHECKPOINT"
```

The transcript goes to **stdout**. Everything else — timings, warnings, where the
mouth was unreadable — goes to **stderr**, so you can pipe the text somewhere
useful without cleaning it up first.

```
the choices don't make sense because it's the wrong question
208 frames, 8.3s, RTF 0.33  |  6 occlusion(s), 61 frame(s) with no detected face
mouth unreadable:
    1.32s -   1.44s  (0.12s, frames 33-35)
    3.44s -   4.04s  (0.60s, frames 86-100)
```

That second block is the part worth understanding.

## What "mouth unreadable" means

When the speaker turns away, blurs, or leaves frame, VoxLens cannot read anything —
and it says so, with timestamps. Those stretches are **Occlusions**.

Two honest caveats about them:

- **The transcript still covers that time.** The recogniser is handed interpolated
  mouth positions across a gap, so it produces words there anyway. VoxLens tells you
  *when* it was blind; it does not currently tell you *which words* were affected.
  Treat text near a reported Occlusion with suspicion.
- **It never fabricates deliberately.** Filling gaps with plausible text is a feature
  that was considered and deliberately not built. Gaps are reported, not invented.

A gap has to last **3 frames (120 ms)** to count — shorter dropouts are detector
noise. Tune with `--occlusion-min-frames N` if your footage is unusual.

## Machine-readable output

```bash
voxlens interview.mp4 --checkpoint "$VOXLENS_CHECKPOINT" --json
```

Emits one JSON object — transcript, occlusion spans with frames and seconds, timings,
and the exact configuration that produced it, so a result is reproducible from its own
output. Nothing but the payload reaches stdout, so this pipes cleanly:

```bash
voxlens clip.mp4 --checkpoint "$VOXLENS_CHECKPOINT" --json | jq -r .transcript
```

Pass several videos and the checkpoint loads once, with one JSON object per line:

```bash
voxlens clips/*.mp4 --checkpoint "$VOXLENS_CHECKPOINT" --json > transcripts.jsonl
```

## Options you may actually want

| Flag | Why |
| --- | --- |
| `--device hybrid\|mps\|cpu` | Defaults to `hybrid` — encoder on the GPU, search on the CPU, the fastest measured split. All three produce identical text. |
| `--beam N` | Defaults to `1` (greedy). Higher is more accurate and much slower: beam 40 costs about 16× the compute for ~3 points of accuracy. |
| `--pre-cropped` | Your video is *already* a mouth crop, so skip face detection. Benchmark corpora ship this way. Without it, VoxLens tries to find a face inside a mouth and produces nonsense. |
| `--occlusion-min-frames N` | How many consecutive unreadable frames count as an Occlusion. |

## Exit codes

Useful when scripting over many files:

| Code | Meaning |
| --- | --- |
| `0` | A transcript was produced |
| `1` | The video decoded, but the mouth could not be read — unusable footage |
| `2` | The invocation is wrong: missing or undecodable file, or an unavailable device |
| `3` | The checkpoint is missing, unreadable, or the wrong architecture |

`1` and `2` are deliberately distinct: one means *your video is bad*, the other means
*your command is bad*.

## Scoring against a corpus

```bash
voxlens-eval /path/to/corpora --corpus lrs3 \
  --checkpoint "$VOXLENS_CHECKPOINT" --out results.json
```

Reports word error rate and throughput, and writes per-clip references and hypotheses
so you can read the worst failures rather than only the average. `--stride N` samples
every Nth clip for a fast estimate; sampling is by stride rather than taking the first
N, because both corpora are ordered.

The harness drives the CLI rather than reaching into Python internals, so the number
it reports is what a user of the command actually gets.

**Current baselines**, greedy decoding: **34.3% WER** on the full LRS3 test split,
**47.9%** on a WildVSR sample. These are regression targets measured against
themselves — not claims of parity with published research
([ADR-0005](adr/0005-two-evaluation-bars.md)).

## What to expect from the output

Roughly a third of clips come back word-perfect. Roughly one in seven comes back worse
than useless. It is not uniformly mediocre — it tends to nail a clip or lose it
completely.

The errors are also not random. *Wash* becomes *wish*; *backed* becomes *banks*. Words
that look alike on the mouth are genuinely indistinguishable on camera, because the
difference between them happens in the throat. English has ~44 sounds and about a dozen
distinguishable mouth shapes, so whole groups of words collapse together. Human
lip-readers face the same wall, at 30–45% word accuracy on unconstrained speech.

See [`demo.html`](demo.html) for word-by-word comparisons across the full benchmark.

## What it does not do

- **Real-time streaming.** Clips only, for now ([ADR-0001](adr/0001-clips-first-streaming-target.md)).
- **Fill in what it could not see.** Occlusions are reported, never invented.
- **Multiple speakers.** Exactly one speaker per clip.
- **Languages other than English.**
