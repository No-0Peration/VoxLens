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
| `--divergence` | Read the clip with the CTC head too, and report how far the two decoders disagree. A measurement, not a confidence — see below. Roughly doubles decode cost. |

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

### Reading a Stream, not a Clip

By default a batch of crops is one clip and gets one transcript back. Pass
`--stream` and the session becomes live instead: the server decodes **three-second
windows advancing one second at a time**, so text appears while someone is still
speaking ([ADR-0012](adr/0012-windowed-decoding-not-true-streaming.md)).

```bash
voxlens-replay mouth.mp4 --stream --realtime
```

```
we're not doing that anymore i think it's a good thing to have [a very good]
we're not doing that anymore i think it's a good thing to have a very good [i'm going to tell you what]
```

**The brackets are the point.** Text inside them is provisional — the next window
sees more context and may read it differently. Text outside them has frozen and
will never change. Live captioning has always worked this way, because text that
rewrites itself under a reader's eyes is exhausting, and freezing text you could
immediately improve is wasteful.

Measured on an M4 Pro, 20 seconds pushed at 25 fps: text reaches the provisional
edge **0.3–1.3 s** after it is spoken, mean decode lag **0.38 s** per window, worst
**1.44 s** on the first window, which pays for warm-up. The three-times-over
decoding costs about **0.3 RTF** in total. `--realtime` paces the sending to the
frame rate, which is the only way the lag figure means anything.

When the camera loses the mouth it says so rather than sending whatever is in
front of the lens:

```bash
voxlens-replay mouth.mp4 --stream --occlude-every 25
```

An Occlusion cuts the window — there is no signal to read across it — and settles
everything before it, since no later window will cover that stretch.

## Two decoders, and how much to trust what they read

The model has two ways to turn the same encoding into words: a beam search, which is
what you normally get, and the CTC head reading greedily. They routinely disagree.
**How much they disagree predicts how wrong the text is** — measured, not assumed:
Spearman 0.752 against per-clip WER over the whole 2,854-clip WildVSR test set
([ADR-0011](adr/0011-confidence-from-decoder-disagreement.md)).

```bash
voxlens interview.mp4 --checkpoint "$VOXLENS_CHECKPOINT" --divergence
```

```
the choices don't make sense because it's the wrong question
208 frames, 8.3s, RTF 0.51  |  6 occlusion(s), 61 frame(s) with no detected face
reading is FIRM: decoders disagree by 0.18, and Clips in that band average 29% word errors  |  the CTC head read: 'the choices don't make sense because its the wrong question'
```

Three bands, and they are tercile boundaries from that measurement rather than
numbers anybody picked:

| band | divergence | what clips in it actually scored |
| --- | --- | --- |
| `firm` | below 0.27 | 28.8% WER |
| `uncertain` | 0.27 – 0.48 | 50.7% WER |
| `doubtful` | 0.48 and up | 79.3% WER |

**"Firm" does not mean right.** Clips in that band still average better than one
word in four wrong, and only one in twenty-eight comes back word-perfect. What it
does mean is that the two decoders agreed — and 34 of the 36 word-perfect clips in
the whole corpus are in this band. Where the model is exactly right at all, it is
right here. The
number and the band travel together for that reason — a reader told only
"uncertain" has been given a mood, not a measurement.

Under `--json` it arrives as a `divergence` object with the value, the band, that
band's measured WER, and the CTC head's own reading. The transcript on stdout is
untouched either way.

The second decoder is nearly free — RTF 0.097 with both, against 0.097–0.118 for
beam search alone, because the CTC head is one linear layer and an argmax while
beam search is hundreds of sequential steps. The flag stays opt-in, but not for
cost reasons any more.

To re-run the validation, or run it on your own corpus:

```bash
voxlens-eval "$VOXLENS_CORPUS" --corpus wildvsr --checkpoint "$VOXLENS_CHECKPOINT" --divergence
```

## Scoring against a corpus

```bash
voxlens-eval /path/to/corpora --corpus lrs3 \
  --checkpoint "$VOXLENS_CHECKPOINT" --out results.json
```

Three corpus kinds: `lrs3` and `wildvsr` are the benchmark splits, and `captured`
is footage you shot yourself through a lens — the only one that is not pre-cropped,
since it has whole faces in it. That path exists for
[measuring what the camera costs](camera-path-measurement.md), and it reports WER
per group so distances do not average into one meaningless number.

Reports word error rate and throughput, and writes per-clip references and hypotheses
so you can read the worst failures rather than only the average. `--stride N` samples
every Nth clip for a fast estimate; sampling is by stride rather than taking the first
N, because both corpora are ordered.

The harness drives the CLI rather than reaching into Python internals, so the number
it reports is what a user of the command actually gets.

**Current baselines**, greedy decoding: **34.3% WER** on the full LRS3 test split,
**50.0%** across the whole 2,854-clip WildVSR test set. These are regression targets measured against
themselves — not claims of parity with published research
([ADR-0005](adr/0005-two-evaluation-bars.md)).

## Live capture: crops over a socket

The recogniser also runs as a small server that takes **mouth crops** instead of video
files, so a camera somewhere else can feed it
([ADR-0009](adr/0009-phone-is-a-camera-not-the-model-host.md)). That camera is meant to
be a phone, and the phone app does not exist yet
([#20](https://github.com/No-0Peration/VoxLens/issues/20)) — so what ships alongside the
server is a client that replays crops from a file.

```bash
voxlens-serve --checkpoint "$VOXLENS_CHECKPOINT"
```

The checkpoint loads once, before the socket opens, so a client that connects has a
recogniser waiting rather than a 4 GB load on its first message. It listens on
`127.0.0.1:9601` — loopback only, so nothing is reachable from a network until you pass
`--host`. `--port 0` takes any free port and prints the one it bound.

Then, from anywhere that has crops:

```bash
voxlens-replay mouth.mp4
```

The transcript goes to stdout and diagnostics to stderr, exactly as with `voxlens`.
`--json` emits the server's whole reply per batch, one JSON object per line.

The input has to be **already-cropped** 96×96 mouth regions — what benchmark corpora
ship, and what a phone would send. A file of whole frames is refused rather than guessed
at: finding the face is the camera's job on this path. `.npy` arrays of shape
`(frames, 96, 96, 3)` uint8 work too, which is the easy way to test without video.

Pass several files and they go through one session, which is the point of a server: the
checkpoint is loaded once and reused for all of them. `--chunk N` splits a file into
N-frame batches, one transcript each — useful for exercising a session, but the batches
are decoded independently and **not** stitched together. Overlapping windows with a
visible revision boundary are [#21](https://github.com/No-0Peration/VoxLens/issues/21).

What the wire expects:

| | |
| --- | --- |
| Crops | 96×96×3 uint8 RGB — about 27 KB a frame, 0.7 MB/s at 25 fps |
| Frame rate | Declared once per session, and refused unless it is 25 fps, the rate the checkpoint was trained at. Nothing is resampled quietly. |
| Session | Named by the client and echoed in every reply, so two cameras are tellable apart in the output and the logs |
| Losing the camera | A disconnect ends that session and nothing else — the loaded model is untouched and the next session finds it intact. A session that goes silent for five minutes is let go, since a phone in airplane mode never sends a goodbye. |

**No authentication, and no encryption.** Crops of someone's mouth and the text of what
they said would both cross the wire in the clear, which is why the default is loopback.
Putting this on a network is a decision to make deliberately, and not on an untrusted
one.

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
