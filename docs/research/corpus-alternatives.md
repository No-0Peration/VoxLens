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
| **CREMA-D** | **ODbL 1.0 + DbCL 1.0** | 3,745,152,087 B = 3.75 GB working tree (7.55 GB to clone — git-lfs keeps a second copy) | 12 fixed sentences, 91 actors, `.flv` video — the only clearly commercial-permissive licence found |
| **RAVDESS** (Zenodo 1188976) | CC BY-**NC**-SA 4.0; commercial licence purchasable | 25,600,214,208 B = 25.60 GB (47 video zips = 25.17 GB) | **2 sentences only** — an eval/emotion set, not a training corpus |
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

## Acted emotional corpora, verified in detail

A follow-on pass closed the open questions on these three. **None contains unconstrained continuous speech** — all are acted, fixed-script read speech — so none substitutes for a real lip-reading corpus.

- **MEAD** — **442,952,785,920 B (442.95 GB)**, computed from the official Drive folders' embedded metadata. Terms of Use clause (1): *"used for non-commercial/non-profit research purposes only"*, clause (2) narrows further to "academic purposes". **Only Part0 (48 of 60 actors) ever shipped** — the Part1 release promised for June 2021 never happened, and actor `W021` has audio but no video. Best phonetic coverage of the three (159 TIMIT-derived sentences, 1080p, 7 camera views) and the worst terms. The repo's MIT badge covers the **baseline model code only**, not the data — an easy and expensive conflation.
- **CREMA-D** — **3,745,152,087 B** working tree, derived by summing all 22,326 git-lfs pointers. ODbL 1.0 + DbCL 1.0, **commercial use permitted**, no gate (the Google Form is a courtesy census). Video is `.flv` and needs transcoding; `VideoFlash` alone is 2.44 GB. Note GitHub classifies the licence as `NOASSERTION`, so automated scanners will flag it.
- **RAVDESS** — **25,600,214,208 B**, summed from all 49 Zenodo file sizes. CC BY-NC-SA 4.0. The original `smartlaboratory.org` host is **404**; it moved to `affectivedatascience.com` after the Ryerson → Toronto Metropolitan rename. Two sentences total ("Kids are talking by the door" / "Dogs are sitting by the door"), which makes it near-useless for training and fine as a held-out eval set. A commercial licence is genuinely purchasable (CAD $6k–$20k depending on term and company size).

### One finding that bears directly on #10

RAVDESS is the only rights-holder found that states explicitly whether its data restrictions **run with the model weights**:

> "This licence applies only to use of the RAVDESS recordings themselves. It does not apply to models, model weights, embeddings, extracted features, annotations, predictions, statistics, software, or other outputs derived from the recordings, provided that those outputs do not contain or reproduce the original RAVDESS recordings or any substantial portion of them."

The same page still requires an active commercial licence whenever the recordings are "used, copied, processed, or accessed for active commercial work… including retraining, fine-tuning, benchmarking, validation, feature extraction."

This is one dataset owner's position on one dataset, not a general legal rule, and it says nothing about what LRS2's or LRS3's terms mean. But #10's pivotal question is whether training-data restrictions travel with released weights, and this is a concrete instance of a rights-holder answering "no, provided the output does not reproduce the source" — while still gating the *act of training* itself. Both halves matter.

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
| ~~MEAD size and licence~~ | **Closed.** See below — 442.95 GB, non-commercial research only. |
| MultiVSR English hours | Not broken out. Derive from `ytids.txt` plus the language column in `train_major.csv`. |
