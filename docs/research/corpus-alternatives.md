# Audio-visual corpora beyond LRS2/LRS3 — follow-on sweep

Companion to [`corpus-availability.md`](./corpus-availability.md), produced by a parallel sweep over corpora that document treated only briefly. **It revises that document's central conclusion — see "Correction" below.**

Everything marked VERIFIED was read from the primary source (page text, HTTP headers, Zenodo/HuggingFace APIs, or repo files). Everything marked UNVERIFIED states what to check.

## Correction to `corpus-availability.md`

That document concluded there is **"no identified route to a licensed, obtainable, unconstrained-English AV training corpus for an unaffiliated individual."** This sweep identifies two candidate routes it missed:

- **MultiVSR** (VGG Oxford, ICASSP 2025) — HuggingFace `sindhuhegde/multivsr`, **VERIFIED `gated: false`, `license: mit`**, 42,984,746,968 B (42.98 GB) of metadata. Ships per-YouTube-ID transcripts *with word timings* plus face-track pickles; you fetch the video from YouTube yourself. ~12,000 h, ~1.6 M clips, English + 12 other languages, unconstrained continuous.
- **VoxPopuli / European Parliament CC0 route** — build mouth ROIs from EP video using the CC0 segment index. The only path found with no NC clause, no gate, and no institutional email. Cost is compute and bandwidth, not permission.

Both carry the same caveat: the permissive licence covers the **metadata**, not the underlying video. Neither is turnkey. But "no route exists" is too strong, and fine-tuning should be re-scoped as *hard and licence-encumbered* rather than *impossible*.

## Genuinely open and disk-feasible

| Corpus | Licence | Exact size | Content |
| --- | --- | --- | --- |
| **GRID** (Zenodo record 3625687) | **CC BY 4.0**, open, no signup | 16,211,527_582 B = 16.21 GB | Constrained 6-word grammar, 34 talkers × 1,000 sentences, English |
| **Lombard GRID** | **CC BY 4.0**, direct download | 2,485,003,295 B = 2.49 GB | GRID grammar, frontal **and profile** views, 54 talkers |
| **CREMA-D** | **ODbL 1.0 + DbCL 1.0** | ~7.55 GB | 12 fixed sentences, 91 actors — the only clearly commercial-permissive licence found |
| **NTCD-TIMIT** (Zenodo 260228) | CC BY-**NC** 4.0, open | 44,796,013,834 B = 44.80 GB | TCD-TIMIT derived: audio + **pre-extracted visual features, no raw video** |

Zenodo's GRID description claims the per-talker zips hold ".jpg videos" — this is wrong. Reading `s1.zip` local file headers via HTTP range request shows `s1/bbaf2n.mpg`. They are MPG. GRID's high-quality video tier is ~81 GB and will **not** fit 78 GB; the normal-quality Zenodo package will.

Prefer the Zenodo GRID copy over Sheffield's: Sheffield offers only "freely available for research use" with no formal licence instrument, while Zenodo carries an actual CC BY 4.0 grant.

## Closed by written policy

- **LRW / LRS2 (Oxford-BBC)** — payload live but behind HTTP Basic auth (`401`, realm confirmed). BBC R&D states verbatim that use "is not permitted by companies or independent researchers", requires a validated academic address, and grants permission for 12 months only. LRW is 70 GB, word-level (500 classes, 29 frames each) — constrained, not sentences.
- **LRW-1000 / CAS-VSR (CAS VIPL)** and **CMLR (Zhejiang)** — "public to universities and research institutes for research purpose only". CAS requires an agreement signed by a full-time staff member. Both Mandarin. CMLR's link is a Baidu Pan share; size UNVERIFIED.
- **TCD-TIMIT** — old download tree returns 404; the replacement site requires a university email and exposes no TCD-TIMIT licence document at all (only a RoomReader non-commercial agreement). Size extrapolated from 7 archived speaker folders at ~8.34 GB each × 62 speakers ≈ **~517 GB**. Infeasible regardless.
- **LIP-RTVE** — gated behind an NDA on the source RTVE database.

## Withdrawn, moved, or trap-laden

- **LRS3** — `mmai.io` now states "Downloads are no longer available from this website", pointing at LRS-VoxMM instead. This silently breaks MuAViC-English and Auto-AVSR's data preparation for everyone.
- **VoxCeleb2** — Oxford dropped it; **KAIST `mmai.io` still distributes it**, which is easy to miss. Metadata (YouTube URLs + timestamps) downloads anonymously today: `vox2_dev_txt.zip` = 1,577,119,007 B, VERIFIED 200. Full AV package is by request with an "(Use institutional email)" field. Licence is **CC BY 4.0** per `license.txt` — note Oxford's residual page says CC BY-**SA**, and the two disagree. **No transcripts exist anywhere official**; Auto-AVSR generates them with Whisper.
- **AVSpeech** — official CSVs now sit behind a Google sign-in (anonymous fetch returns 403). CC BY 4.0 covers the **annotation CSV only**, not the YouTube video. **No transcripts of any kind** — it was built for speech separation. Media size is multiple terabytes.
- **MuAViC** — CC BY-NC 4.0, ships **no media**. Its English path calls `prepare_lrs3()`, which demands a manual LRS3 download from a URL that no longer serves. Non-English is fully self-serve (mTEDx via OpenSLR-100), but mTEDx is CC BY-NC-**ND**, and ND arguably prohibits derived mouth crops.
- **Third-party HuggingFace mirrors** — several re-uploads exist (LRS3 raw at 149.99 GB, LRS2 pretrain at 61.86 GB with no licence, VoxCeleb2 clones tagged `mit` in contradiction of their own cards). The LRS3 mirrors have a partial defence, since LRS3 was published CC BY 4.0 — but the underlying TED material is CC BY-NC-ND, so the whole lineage has a chain-of-title problem predating any mirror. The LRS2, VoxCeleb2, AVSpeech and MuAViC dumps have no such defence.

## Recommended path

1. **GRID from Zenodo** (16.21 GB, CC BY 4.0, zero permission) — the only genuinely unencumbered sentence-level AV corpus. Use it to build and debug the pipeline, never to measure open-vocabulary performance.
2. **Add Lombard GRID** (2.49 GB, CC BY 4.0) — free second viewing angle plus a Lombard-speech robustness axis.
3. **CREMA-D** (7.55 GB, ODbL) if a commercially-permissive licence matters.
4. **For unconstrained English at scale**: MultiVSR metadata + your own YouTube fetch of an English subset, or the VoxPopuli CC0 route.
5. **Do not plan around** TCD-TIMIT, LRW/LRS2, LRW-1000/CAS-VSR/CMLR, or MuAViC-English.

## Unverified — how to close each

| Item | How to close |
| --- | --- |
| TCD-TIMIT exact total | Extrapolated from 7 archived folders. Email `sigmedia_database@tcd.ie`. |
| TCD-TIMIT licence | No document exists on the live site. Ask whether the RoomReader NC agreement is meant to cover it. |
| LRW-1000, CMLR sizes | Not published; obtainable only post-agreement (CMLR: read the Baidu Pan share listing). |
| VoxCeleb2 full AV size | No official figure. Request access and read the emailed manifest. |
| AVSpeech media size | Unknowable in advance. Run the downloader with `dryrun=1` on a sample. |
| WildVSR size and licence | Repo has **no LICENSE file**; the CC BY-NC-ND attribution circulating for it is unverified. The 91,755,398 B figure comes from the Drive page, not a labelled size — open it signed in. Email the authors for a licence statement. |
| LRS-VoxMM size | Not stated in the paper or on the project page. |
| MEAD size and licence | No licence published anywhere. Email the authors. |
| MultiVSR English hours | Not broken out. Derive from `ytids.txt` plus the language column in `train_major.csv`. |
