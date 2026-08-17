# Which unconstrained-English corpus can VoxLens actually obtain?

**Ticket:** [#2](https://github.com/No-0Peration/VoxLens/issues/2) · **Researched:** 2026-08-17

## The question

VoxLens needs an unconstrained-English audio-visual speech corpus it can actually obtain and hold on disk. For each realistic candidate, establish the access process, the licence terms for a **non-academic, experimental project run by an individual**, the on-disk size of the splits needed for evaluation now and fine-tuning later, and whether the dataset is still actively distributed.

**Hard constraints:** ~78 GB free disk (verified: `df -h` reports 78Gi available). No institutional or academic affiliation. Audio-visual training data is acceptable; inference is video-only.

**Scope note:** the immediate need is a spec for a clip-to-transcript CLI on a **pretrained** checkpoint. Fine-tuning is out of scope, so the binding requirement is an **evaluation** set, not a training corpus. That distinction turns out to decide the whole ticket.

---

## Comparison table

| Corpus | Still distributed? | Access | Licence for non-academic individual | Eval-split size | Unconstrained English? |
| --- | --- | --- | --- | --- | --- |
| **WildVSR** | **Yes** — direct link | None. Public Google Drive link from GitHub | No explicit licence stated (gap); built only from CC-licensed YouTube | **~87.5 MiB** (verified) | Yes — 4.8 h, 2,854 utts |
| **LRS3 test via MIT CSAIL** | **Yes** — direct link | None. Ungated `wget` | Unstated; conflicting upstream signals — treat non-commercial | **187 MiB** (verified) | Yes — the canonical 1,321-utt test split |
| **LRS-VoxMM** | **Yes** — request form | Web form; requests institutional email | CC BY 4.0, "for research purposes" | Unverified GB; 1.8 h test / 23.5 h dev | Yes — in-the-wild, 12 domains |
| **LRS2-BBC** | Yes, but gated | Signed BBC contract, `.edu` email, countersigned by BBC R&D | **Blocked** — "not permitted by companies or independent researchers" | 50 GB single tar (no split-level download) | Yes — 0.5 h test |
| **LRS3-TED** (original corpus) | **No — withdrawn** | All endpoints 404 | Moot | 407 h pretrain, gone | Yes |
| LRW | Yes, BBC-gated | Same BBC contract as LRS2 | Same blocker | — | **No** — 500-word classification |
| TCD-TIMIT | Yes, but **moved** to `sigmedia.tv` | Email request; portal required university email | Research use | — | **No** — 6,913 read TIMIT sentences |
| VoxCeleb2 | Yes | Oxford form | Research use | — | **No transcripts** — speaker corpus |
| AVSpeech | Yes | CSV of YouTube IDs only | You scrape YouTube yourself | Unbounded | **No transcripts** |
| CMLR | — | — | — | — | **No** — Mandarin |
| MuAViC | Scripts live, English data **not** | `get_data.py` demands a manual LRS3 download | Repo `NOASSERTION` | — | English blocked by LRS3 |

Every row is resolved in the per-candidate sections below.

---

## Per-candidate detail

### LRS3-TED — WITHDRAWN. Do not plan around it.

This is the single most important finding, because LRS3 is the default benchmark for essentially every published VSR checkpoint.

**Verified from primary sources:**

- The Oxford VGG page `https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html` returns **HTTP 404**, while its siblings `lrs2.html` and `lrw1.html` both return **200**. The [dataset index](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/) still *links* to `lrs3.html`, so the link is dangling — the index is not evidence of availability.
- Internet Archive CDX history for that URL shows `200` responses through **2023-01-08**, then `404` from **2024-04-03** onward, including a `404` captured **2026-02-22**. So the page went down between January 2023 and April 2024 and has stayed down.
- The successor site, KAIST Multimodal AI Lab, states plainly on [mmai.io/datasets/lip_reading](https://mmai.io/datasets/lip_reading/): *"Downloads are no longer available from this website."* It points instead to LRS-VoxMM as a new lip-reading benchmark.
- Every original download endpoint recovered from the archived 2023 page now returns **404** — both mirrors and every artefact:
  - `https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_test_v0.4.zip` → 404
  - `http://mm.kaist.ac.kr/lip_reading/data3/lrs3_test_v0.4.zip` → 404
  - `https://www.robots.ox.ac.uk/~vgg/data/lip_reading/files/lrs3_v0.4_txt.zip` (text annotations only) → 404
  - The access request form `https://goo.gl/forms/vGZmhJaZ9LAklozz2` → 404
- Corroborating structural evidence: on the VGG download host, `thor.robots.ox.ac.uk/lip_reading/data2/` (LRS2) returns **401 Unauthorized** — the directory exists and demands credentials — whereas `.../data3/` (LRS3) returns **404**. The LRS2 data is still there behind a password; the LRS3 data is gone.

**What LRS3 was**, from the archived page and the [LRS-VoxMM paper](https://arxiv.org/abs/2604.27866) Table 1:

| Split | Videos | Utterances | Hours | Download |
| --- | --- | --- | --- | --- |
| Pre-train | 5,090 | 118,516 | 407 | 7 parts, "approximately 10GB" each |
| Trainval | 4,004 | 31,982 | 30 | single zip |
| Test | 412 (451 in v0.4) | 1,321 (1,452 in v0.4) | ~1 | single zip |

The archived page stated the annotations were licensed **CC BY-NC-ND 4.0**, while mmai.io currently describes LRS3 as **CC BY 4.0**. The two statements conflict; since the data is unobtainable this is academic, but it is a caution against trusting any single restatement of these licences.

**Consequence:** published VSR WER numbers are overwhelmingly reported on the LRS3 test set, and that test set can no longer be obtained legitimately. VoxLens cannot reproduce a headline number against its original benchmark. It must evaluate on a substitute and accept that its numbers are not directly comparable to published LRS3 figures.

#### But the LRS3 *test set* is obtainable — preprocessed, ungated, 187 MB

The original corpus is gone; the **test split survives in a third-party preprocessed re-host**, and this is the most useful single finding in this document after WildVSR.

The MIT CSAIL Spoken Language Systems group publishes it as part of [Whisper-Flamingo](https://github.com/roudimit/whisper-flamingo), whose README states: *"LRS3 / MuAViC: We provide all data to reproduce the results on the test set."*

```
https://data.csail.mit.edu/public-release-sls/whisper-flamingo/muavic.tar.gz
```

**Verified by direct inspection** — I streamed the archive and enumerated it rather than trusting the description:

- HTTP **200**, no auth, no form. `Content-Length: 196215206` (**187 MiB**), `Last-Modified: 2024-05-03`.
- 3,521 entries, containing **1,321 `.mp4` files across 412 distinct video IDs** under `muavic/en/video/test/`, plus 1,321 matching `.wav` files and the transcripts `muavic/en/test.wrd`.
- **1,321 utterances across 412 videos is exactly the official LRS3 v0.4 test split**, as stated by both mmai.io and the archived Oxford page. This is the real benchmark, not an approximation.
- Provenance is documented in the repo's `preparation/README.md`: the LRS3 data was preprocessed with the MuAViC codebase, producing lip ROIs *"nearly the same as using the AV-HuBERT preparation code"*, with slightly more video compression.

**Why this matters:** VoxLens can, after all, compute WER on the canonical LRS3 test set and check it against published numbers. That is the single most valuable diagnostic available — if a checkpoint advertised at ~20% WER does not reproduce roughly 20% here, the fault is in VoxLens's preprocessing, not the model. The mouth-crop convention is where these pipelines silently break.

**Licence caution.** The tarball carries no licence file. Upstream signals conflict: mmai.io calls LRS3 CC BY 4.0, the archived Oxford page called the annotations CC BY-NC-ND 4.0, MuAViC is CC BY-NC 4.0, and the underlying TED content is CC BY-NC-ND. **Treat this as non-commercial, evaluation-only.** It is a sound basis for a local accuracy check and not a basis for redistribution or a shipped product.

#### One further artefact survives: the annotations

`https://mmai.io/datasets/lip_reading/files/lrs3_v0.4_txt.zip` still returns **HTTP 200**, `Content-Length: 417928862` (≈418 MB), `Last-Modified: 2023-02-15` — verified by direct request. It is **no longer linked from any page**; it is simply still on the server. Per the archived Oxford page, it contains for every sample the source **YouTube URL**, frame IDs, **per-frame face bounding boxes**, and word-boundary timestamps (pre-train only), at 25 fps.

So LRS3 is in principle *reconstructible*: 418 MB of annotations plus your own YouTube retrieval. **This is a fallback, not a recommendation.** The obvious problems: TED videos have been deleted in the years since, so coverage will be partial and unquantified; scraping YouTube engages its Terms of Service; the underlying TED content is CC BY-NC-ND; and an incompletely reconstructed test set produces WER figures that are *not* comparable to published LRS3 numbers, which defeats the only reason to want LRS3 in the first place. Reach for this only if a specific published comparison becomes essential.

#### Third-party mirrors exist. Their provenance is not clean.

HuggingFace hosts numerous LRS3 mirrors — `TheNHz/ellipsis-lrs3-raw` (declares `cc-by-4.0`, ~150 GB stated), `Ainncy/LRS3_Split`, `mattymchen/lrs3-test`, and a dozen others — and a parallel set of LRS2 mirrors (`MahmoodAnaam/LRS2-*`, `rishabhjain16/lrs2`, and more), none of which declare a licence.

Stated plainly:

- **The LRS2 mirrors are straightforward breaches.** The BBC agreement forbids redistribution outright, and no LRS2 mirror declares a licence because none can. Using one means holding BBC-copyright broadcast material with no licence at all, and the BBC's prohibition on training commercially-sold models attaches to the content regardless of how it was obtained. Avoid entirely.
- **The LRS3 mirrors sit on firmer but still unsettled ground.** The current mmai.io licence text says CC BY 4.0 and contemplates redistribution; the archived Oxford page said the annotations were **CC BY-NC-ND 4.0** — non-commercial, no derivatives — and a resharded, re-cropped mirror is squarely a derivative. Two primary sources genuinely conflict. Underneath both, TED's own content is CC BY-NC-ND, so commercial use is unavailable through any of these routes.
- **Mirror completeness is not guaranteed.** The largest LRS3 mirror's own card concedes its pretrain split holds roughly 87% of the official clips, with hundreds of videos missing and thousands of boundary duplicates. Trainval and test are described as complete, but that is the uploader's claim, unverified against the official MD5s.

None of this is needed if VoxLens evaluates on WildVSR or LRS-VoxMM, which is why the recommendation below avoids mirrors entirely.

### LRS2-BBC — still distributed, but contractually closed to an unaffiliated individual

**Verified from the [LRS2 page](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html):**

- Still live and still served: the download host returns `401` with `WWW-Authenticate: Basic realm="Lip Reading Sentences (LRS2) in the Wild Dataset"`. It is gated, not withdrawn.
- The page states the package *"is available for non-commercial, academic research"* and that you must sign a Data Sharing agreement with BBC Research & Development.
- **The dataset is one file**: `lrs2_v1.tar`, listed on the page as **50GB**. There is no per-split download — the filelists (`pretrain.txt`, `train.txt`, `val.txt`, `test.txt`) are only index files that partition the single archive. You cannot fetch just the 0.5-hour test split.

**Eligibility is settled explicitly on the [BBC R&D datasets page](https://www.bbc.co.uk/rd/projects/lip-reading-datasets), and it excludes VoxLens by name.** The page states: *"Use is not permitted by companies or independent researchers."* It adds that the BBC is *"only able to deal with universities or public service, non-commercial research organisations"*, that applicants must use an official academic email address (normally `.edu`), and that *"Gmail or non academic email accounts are not acceptable."* Completed forms go to `lrw.lrs2@bbc.co.uk`.

That single sentence resolves the ticket for LRS2: an unaffiliated individual is not an eligible applicant, and no amount of disk space or goodwill changes it.

**The actual contract** — I downloaded and read the real agreement, [LRS2 Permission Form (.docx)](https://downloads.bbc.co.uk/rd/datasets/content-analysis/LRS2-Lip%20Reading%20Sentences%20Permission%20Form-.docx), linked from the same page. Its terms confirm and extend the blocker:

- Permission is granted solely for **"Academic Research Purposes"**, for **12 months**, after which all content must be deleted.
- The form requires **"Name of organisation"** and the **"Registered/principal postal address of organisation"**, plus **"Name of your professor or supervisor (if you are not a member of staff of your organisation)"**. The agreement states it is *"a contract between your organisation (including you) and us"* and that *"You must be authorised by your organisation to enter into this contract."* An individual with no organisation has no way to complete it.
- **Storage is geographically and organisationally restricted**: content may only be stored *"on your organisation's servers in the United Kingdom or the European Economic Area"* or on UK/EEA cloud servers. A personal laptop is not an organisation's server. This alone rules out the intended on-device Apple Silicon workflow.
- Commercial use is prohibited outright, and the AI clause is explicit: the content *"must not be used for training any existing or new technology, algorithms or models that will be sold commercially"* and *"must not be used to train technology to improve or enhance your operational systems"* — permitted only *"for comparative or benchmarking purposes."*
- BBC can revoke at any time without reason, and the applicant must delete everything on request.
- Redistribution is flatly prohibited — the BBC page states the data is for your use only and must not be passed on; the form bars making copies or letting anyone else use the content, and forbids attaching Creative Commons terms to BBC content.
- Submission is by emailing a completed Word `.doc` to a named BBC contact, countersigned by Rob Cooper, Lead R&D Development Producer, BBC R&D; the page notes *"Forms must be sent as Word .doc attachments."* Approval time is not stated anywhere — **unverified**.

**Hours per split**, from Afouras et al., [Deep Audio-Visual Speech Recognition](https://arxiv.org/abs/1809.02108) Table I: pre-train **195 h** (96k utts), train-val **29 h** (47k utts), test **0.5 h** (1,243 utts). Per-split *byte* sizes are **unverified** and effectively unknowable from outside — LRS2 ships as one tar with one MD5, and on disk it divides into `pretrain/` and `main/` (train, val and test all drawn from `main/`), so a three-way size split does not exist.

**Assessment:** LRS2 is not a licensing grey area for VoxLens; it is a straightforward no, stated in as many words by the distributor. The contract needs an institutional counterparty VoxLens does not have, and it forbids the storage location and the eventual product direction. Even setting that aside, the 50 GB single tar against 78 GB free leaves no room to extract it (a naive `tar -xf` would need ~100 GB peak).

### LRW / LRW-1000 — wrong task, same blocker

LRW is distributed under the **same BBC permission-form regime** as LRS2 (the [BBC page](https://www.bbc.co.uk/rd/projects/lip-reading-datasets) offers a parallel `LRW-Lip Reading Words Permission Form-.docx`), so it inherits every constraint above. Independently, it is disqualified on task grounds: it is **word-level classification over a 500-word vocabulary**, not unconstrained sentence recognition. The [LRS-VoxMM paper](https://arxiv.org/abs/2604.27866) Table 1 lists LRW as 514k train-val utterances / 165 h and 25k test / 8 h, with vocabulary 500.

### WildVSR — obtainable today, 88 MB, no agreement

This is the find that unblocks the ticket.

**Verified from primary sources:**

- Repository: [github.com/YasserdahouML/VSR_test_set](https://github.com/YasserdahouML/VSR_test_set). Paper: [Do VSR Models Generalize Beyond LRS3?](https://arxiv.org/abs/2311.14063), WACV 2024.
- **Access: a direct public Google Drive link in the README.** No form, no data agreement, no institutional email, no approval wait.
- **Size: `WildVSR.zip` = 91,755,398 bytes (~87.5 MiB).** Confirmed two ways: Google Drive's interstitial reports `WildVSR.zip (88M)`, and the `content-length` on the confirmed download endpoint gives the exact byte count.
- Contents, verified from the ZIP central directory: **2,854 `.mp4` files** under `WildVSR/videos/` plus `labels.json` — matching the paper's 2,854 utterances exactly. The repo also ships the loader `wildvsr_test.py` and `transforms.py`.
- **Input convention matches the standard VSR pipelines**: clips are stored as mouth crops and `wildvsr_test.py` converts to grayscale with `CenterCrop((88,88))`, `mean=0.421`, `std=0.165` — the auto-AVSR / AV-HuBERT convention. They can be fed straight to a VSR model with no face detection at all.
- Statistics from the paper (Table 1): **618 speakers, 2,854 utterances, 45,182 word instances, 6,040 vocabulary, 4.8 hours**, drawn from 478 YouTube videos, clips 0.5–16 s. For comparison the same table gives the LRS3 test set as 412 speakers / 1,321 utterances / 1,997 vocabulary / 0.9 h — WildVSR is **5.3× the duration and 3× the vocabulary** of the benchmark it replaces.
- Construction: YouTube search filtered to **videos published under a Creative Commons licence**; scene detection, YOLOv5-Face detection, SyncNet active-speaker filtering, Whisper for language ID and pseudo-transcripts, then *"further verified manually to ensure high-quality clip-text pairs."*
- The authors state the collection approach *"frees from legal issues encountered with former public datasets"* by targeting only free-to-use content.

**Caveats, stated plainly:**

- **No explicit licence file.** The GitHub repo reports no SPDX licence, and the README specifies only an "Intended Use" of research and development. The underlying videos were CC-filtered and the authors assert freedom from the legal issues of LRW/LRS2, but there is **no licence grant text** covering redistribution or commercial use. For private, local evaluation this is low risk; it is **not** a clean basis for redistributing the data or for a commercial release. Treat "CC-sourced" as the authors' claim about provenance, not as a licence you have been granted.
- **It ships only pre-cropped mouth regions**, deliberately, as a privacy measure — the paper says they *"make available the cropped sequence to reduce the potential for individual identification."* This means WildVSR exercises the **recogniser**, not VoxLens's full Clip-to-Transcript path: face detection, Mouth Region extraction, and Occlusion handling are all bypassed. Those stages need separate evaluation on other footage.
- Transcripts originate as Whisper pseudo-labels, manually verified.
- Last repository activity was 2023-12-13; the Google Drive link resolved on 2026-08-17. There is single-point-of-failure risk in a personal Drive link — **mirror it locally on first download.**

### LRS-VoxMM — the successor benchmark, permissively licensed, but the form asks for an institutional email

**Verified from primary sources:**

- Project page: [mm.kaist.ac.kr/projects/voxmm](https://mm.kaist.ac.kr/projects/voxmm/). Paper: [LRS-VoxMM: A benchmark for in-the-wild audio-visual speech recognition](https://arxiv.org/abs/2604.27866) (arXiv 2604.27866, 30 Apr 2026). This is what mmai.io now recommends in place of LRS3.
- **Licence: Creative Commons Attribution 4.0.** The [licence text](https://mm.kaist.ac.kr/projects/voxmm/files/license.txt) reads: *"available to download for research purposes under a Creative Commons BY 4.0 license,"* with copyright remaining with the original video owners. CC BY 4.0 is permissive and does not itself bar commercial use — but the "for research purposes" framing sits awkwardly beside it, so the effective permission for a commercial product is **ambiguous**. For private evaluation it is fine.
- **It is explicitly evaluation-only.** Paper Table 1 caption: *"LRS-VoxMM is an evaluation-only benchmark derived from the original VoxMM dataset."* Splits are **Dev: ~698 speakers, 27k utterances, 23.5 h** and **Test: ~113 speakers, 2,146 utterances, 1.8 h**, plus four synthetically distorted variants (noise easy/hard, 3-dist easy/hard). There is no training split — so it cannot serve the later fine-tuning need.
- **It includes video files**, preprocessed in LRS-style format (25 fps, 224×224, LRS2/3 face alignment and directory structure) for drop-in use with existing AVSR pipelines. Unlike WildVSR's 96×96 crops, the 224×224 face tracks leave room for VoxLens's own Mouth Region extraction — so this set can exercise more of the pipeline.
- **Access: a web form** at `cn01.mmai.io/keyreq/voxmm` requesting first name, last name, affiliation, and email, with a licence-agreement checkbox. Download links are emailed.

**The constraint:** the email field is annotated **"(Use institutional email)"** and the form requires an "Affiliation". There is no visible client-side enforcement — the field is a plain text input — but approval is manual and a personal address may simply be declined. **Approval time is not stated: unverified.** I did not submit the form. Do not misrepresent an affiliation; if the honest application is refused, WildVSR remains available with no gate at all.

**Disk size is unverified.** Neither the project page nor the paper states a GB figure. Inferring from duration: at 25 fps and 224×224, the 1.8 h test split is plausibly single-digit GB and the 23.5 h dev split a few tens of GB, but these are **estimates, not verified figures**. What would need checking: the sizes quoted in the download email after an approved request.

### Older and constrained corpora

These were considered and rejected on task grounds, independent of licensing.

- **TCD-TIMIT** — **62 speakers reading 6,913 phonetically rich sentences** (verified from the [dataset page](https://sigmedia.tv/datasets/tcd_timit/)). This is scripted read speech, not unconstrained conversational English, so it cannot benchmark a checkpoint trained for in-the-wild sentences. It has also **moved**: the long-cited `sigmedia.tcd.ie/TCDTIMIT/` now returns 404, and the live home is `sigmedia.tv`, where access is requested by email with a required subject tag. The prior portal required a university email validated against the stated institution — the same affiliation blocker. Disqualified on task grounds regardless.
- **GRID** — a fixed six-word command grammar over a tiny vocabulary. Useful only for toy pipelines.
- **CMLR** — Mandarin, not English. Disqualified by the ticket's English requirement.
- **MEAD / CREMA-D / RAVDESS** — emotional-speech corpora built on small fixed sentence sets, not continuous unconstrained speech.

### Corpora without usable transcripts

- **VoxCeleb2** — [still live](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) (HTTP 200), 1,092,009 utterances from 5,994 speakers, metadata licensed **CC BY-SA 4.0**. But its stated objective is *"speaker recognition under noisy and unconstrained conditions"* — it **ships no speech transcripts**. It is used in VSR work as *unlabelled* pretraining data (AV-HuBERT pretrains on it; Auto-AVSR uses it with machine-generated pseudo-labels, per the [LRS-VoxMM paper](https://arxiv.org/abs/2604.27866) §3.1). It cannot serve as an evaluation set because there is no ground truth to score against.
- **AVSpeech** — Google's corpus distributes **CSV files of YouTube IDs and timestamps, not video**. You must scrape YouTube yourself, which means link rot, an unbounded download, and YouTube ToS exposure. It also carries **no transcripts**, and is used in VSR only as pseudo-labelled pretraining data. Not an evaluation option.
- **MuAViC** — Meta's multilingual AV corpus (9 languages, 1,200 h; English portion 436 h from TED/TEDx) **distributes scripts, not video**. Its English path is hard-blocked by LRS3. Verified by reading [`get_data.py`](https://github.com/facebookresearch/muavic/blob/main/get_data.py) directly: `prepare_lrs3()` aborts with *"You have to download LRS3 dataset manually from this link"* and then checks for `pretrain`, `trainval` and `test` directories. Since LRS3 is withdrawn, **MuAViC English is unobtainable by construction.** Note the asymmetry: the non-English languages come from mTEDx, which `get_data.py` downloads directly — so MuAViC is live for other languages, just not the one VoxLens needs. Repo licence is `NOASSERTION`; last pushed 2023-09-11.

---

## What the pretrained checkpoints are actually scored on

This matters because it determines what number VoxLens can quote. From [LRS-VoxMM](https://arxiv.org/abs/2604.27866) Table 2, **visual-only (video-only) WER %** using each project's official public checkpoints:

| Model | LRS3 test | WildVSR | LRS-VoxMM dev / test |
| --- | --- | --- | --- |
| Auto-AVSR | 20.61 | 38.36 | 47.36 / 55.15 |
| Llama-AVSR | 24.31 | 49.22 | 62.88 / 70.71 |
| AV-HuBERT | 27.20 | 51.67 | 59.69 / 65.80 |
| Auto-AVSR* (retrained without LRS3) | 27.64 | 38.26 | 49.39 / 56.55 |

Two things follow. First, **WildVSR and LRS-VoxMM both have published video-only baselines for the exact checkpoints VoxLens would use**, so evaluating on them yields directly comparable reference numbers despite LRS3 being gone. Second, the generalisation gap is large — Auto-AVSR nearly doubles its WER from 20.6% on LRS3 to 38.4% on WildVSR. **A pretrained checkpoint pointed at real footage should be expected to perform closer to the 38–55% band than to the ~20% headline.** VoxLens's spec should set expectations accordingly.

### Checkpoints are downloadable without any corpus access

Confirmed: **obtaining a pretrained VSR checkpoint does not require LRS2 or LRS3 access.** The model zoos are plain download links.

| Checkpoint | LRS3 WER | Size | Licence |
| --- | --- | --- | --- |
| [`auto_avsr`](https://github.com/mpc001/auto_avsr) `vsr_trlrs2lrs3vox2avsp_base.pth` | 20.3 | 1,001,892,616 B (~956 MiB) | Code Apache-2.0; **weights unresolved** |
| [`Visual_Speech_Recognition_for_Multiple_Languages`](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) visual-only | 19.1 | 891 MB (repo-stated) | **Non-commercial, benchmarking only** |
| [AV-HuBERT](https://github.com/facebookresearch/av_hubert) `large_vox_433h.pt` | — | ~5.3 GiB | **Meta licence — see below** |
| BRAVEn Large (+ST +LM) | 20.1 | unverified | MIT (repo field) |
| VALLR (ICCV 2025) | 18.7 | ~696 MiB | CC BY-NC 4.0 (README badge) |

Also relevant: **preprocessing does not require the training corpus.** The `auto_avsr` preparation README makes the landmark archives optional — *"If the `landmarks-dir` is specified, face detector will not be used"* — so inference on arbitrary mp4s runs standalone via RetinaFace or MediaPipe. The 18 GB LRS3 landmark archive is a convenience for reproducing corpus preprocessing, not a dependency.

### The licence problem VoxLens inherits through its checkpoint

Even avoiding these corpora entirely, VoxLens inherits their terms through the weights it builds on.

Auto-AVSR's released checkpoints are trained on **LRS2, LRS3, VoxCeleb2 and AVSpeech** ([LRS-VoxMM](https://arxiv.org/abs/2604.27866) §3.1). The BBC agreement states LRS2 content *"must not be used for training any existing or new technology, algorithms or models that will be sold commercially."* The auto_avsr authors do not resolve whether that runs with their weights — they explicitly decline to, and the hedge is the whole point:

> "Code is Apache 2.0 licensed. The pre-trained models provided in this repository may have their own licenses or terms and conditions derived from the dataset used for training." — [auto_avsr README](https://github.com/mpc001/auto_avsr)

So the Apache-2.0 badge covers the code only, and the weights' status is **openly unsettled by the people who trained them**. The sibling repo is blunter: its code *"can only be used for comparative or benchmarking purposes"* and *"for non-commercial purposes."*

**AV-HuBERT carries a restriction VoxLens should read carefully.** Meta's licence grants rights solely for non-commercial research and explicitly prohibits use for *"purposes of surveillance"* and *"biometric processing."* For a lip-reading tool that reads speech from video of identifiable people, those clauses are not incidental boilerplate — they go to the heart of what the product does. Note also that at least one HuggingFace mirror of AV-HuBERT tags itself `apache-2.0`, which **contradicts Meta's actual licence**; do not rely on mirror metadata.

**Practical effect:** local evaluation is unaffected — do it freely. But the eventual on-device product ambition is constrained less by which corpus VoxLens downloads than by which corpus its checkpoint was trained on, and no mainstream English VSR checkpoint currently offers a clean commercial licence. **This is worth resolving before commercial ambition hardens**, and it is a better-scoped question than the corpus one.

---

## RECOMMENDATION

**Take two small, ungated evaluation sets now — together under 300 MB — and apply for a third.** The disk constraint turns out not to bind at all; the licensing and availability constraints are the real story.

1. **Download the LRS3 test set from MIT CSAIL** (187 MB, ungated): `https://data.csail.mit.edu/public-release-sls/whisper-flamingo/muavic.tar.gz`. Verified to contain exactly the official 1,321 utterances across 412 videos, with transcripts. This is the **calibration set**: it is the benchmark every published WER is quoted against, so it tells you whether your preprocessing is correct. Treat as non-commercial, evaluation-only.

2. **Download WildVSR** (91,755,398 B, ~87.5 MiB, direct link, no agreement, no affiliation). This is the **honesty set**: unconstrained English, 4.8 h, 5.3× the duration and 3× the vocabulary of the LRS3 test set, with published video-only baselines for Auto-AVSR, AV-HuBERT and Llama-AVSR. **Mirror it locally on first download** — it lives on a personal Google Drive link in a repo last touched December 2023.

3. **Apply honestly for LRS-VoxMM** as a third set. CC BY 4.0, evaluation-only, and its 224×224 face tracks exercise VoxLens's own Mouth Region extraction rather than bypassing it. Its form asks for an institutional email, so it may be declined; nothing is lost if so.

4. **Write off LRS2 and LRS3 as corpora.** LRS2 excludes independent researchers by name, needs an organisation as contracting party, restricts storage to UK/EEA organisational servers, forbids commercial model training, and ships as a single 50 GB tar that will not comfortably extract inside 78 GB. LRS3 is withdrawn from every endpoint. Neither should appear in any VoxLens plan — the CSAIL tarball in step 1 gives you the only piece of LRS3 you actually need.

**Suggested starting checkpoint:** `auto_avsr` `vsr_trlrs2lrs3vox2avsp_base.pth` — ~956 MiB, ungated, best public LRS3 VSR WER at 20.3%, Apache-2.0 code, standalone mp4 inference, and a model size that is plausible for the eventual Apple Silicon target. AV-HuBERT is a poor first choice here: 5.3 GiB, a heavier fairseq stack, and a licence prohibiting surveillance and biometric processing.

**Total disk for a checkpoint plus all three evaluation sets: under 2 GB of the 78 GB available.**

### What this means for scope

The ticket asked whether this forces a scope change. **For the current milestone, no.** The immediate goal is a clip-to-transcript CLI on a pretrained checkpoint, and the evaluation need is fully met at under 300 MB. The spec in #5 is not blocked, and the 78 GB disk constraint — the thing the ticket was most worried about — turns out to be irrelevant to it.

**For fine-tuning later, yes.** Every obtainable corpus identified here is **evaluation-only**: WildVSR is a test set by construction, LRS-VoxMM states it has no training split, and the CSAIL tarball is the LRS3 *test* split only. The large labelled training corpora are all closed — LRS2 contractually, LRS3 by withdrawal, LRW on both counts, MuAViC's English path because its own script demands a manual LRS3 download. There is at present **no identified route to a licensed, obtainable, unconstrained-English AV training corpus for an unaffiliated individual**, and 78 GB would not hold LRS3's 407-hour pretrain set (~70 GB of parts alone, before extraction) even if it were available. Any future fine-tuning plan should be treated as an open problem in its own right, not an assumed next step.

**A third risk deserves naming:** the licence encumbrance on pretrained weights (see above) means the eventual commercial-product question is not settled by picking a corpus. That is a separate ticket, and a more urgent one than fine-tuning.

### Caveats to carry forward

- **Evaluating on WildVSR or the CSAIL tarball tests the recogniser, not the whole pipeline.** Both ship pre-cropped mouth regions, so face detection, Mouth Region extraction, and Occlusion handling are all bypassed. Those stages need separate evaluation — LRS-VoxMM's 224×224 face tracks would cover them if access is granted.
- **Use the LRS3 test set as a preprocessing check, not a headline.** If a checkpoint advertised at ~20% WER does not reproduce roughly that figure on the CSAIL data, the fault is almost certainly VoxLens's mouth-crop convention (88×88 grayscale, mean 0.421, std 0.165, aligned to mean face) rather than the model.
- **Neither corpus gives a clean commercial licence.** WildVSR publishes no licence text at all (only a CC-filtered provenance claim), and LRS-VoxMM pairs CC BY 4.0 with "for research purposes". Both are fine for private local evaluation. Neither is a safe basis for redistributing data or shipping a commercial product trained on it.

---

## Verification status

**Verified directly from primary sources:** LRS3's page and every original download endpoint returning 404 (live HTTP checks against both the Oxford and KAIST hosts, plus Internet Archive CDX history showing 200 → 404 between Jan 2023 and Apr 2024); mmai.io's withdrawal notice; the LRS3 annotations zip still live at 417,928,862 bytes; LRS2's 50 GB single-tar packaging and its live 401-gated host (contrasted with LRS3's 404); the BBC's explicit exclusion of companies and independent researchers, and its `.edu` email rule; the full text of the BBC LRS2 permission form, downloaded and read in its entirety; LRS2 hours from arXiv:1809.02108 Table I; WildVSR's 88 MB size (Google Drive interstitial), repo contents, absent SPDX licence, and paper statistics; LRS-VoxMM's CC BY 4.0 licence text, form fields including the institutional-email annotation, evaluation-only status and split statistics; all WER figures above (LRS-VoxMM Table 2, re-extracted from the PDF); MuAViC's hard LRS3 dependency, read from `get_data.py`; TCD-TIMIT's read-speech design and its site move; the CSAIL tarball's 196,215,206-byte size and its contents — enumerated by streaming the archive and counting 1,321 mp4s across 412 video IDs, matching the official LRS3 test split exactly; the auto_avsr weights-licence hedge, quoted verbatim from its README.

**Unverified — would need checking:** LRS-VoxMM's on-disk size in GB (not stated in any primary source; check the sizes quoted in the download email after an approved request). Approval turnaround for both the BBC agreement and the LRS-VoxMM form — stated nowhere. Whether an LRS-VoxMM application from a non-institutional email is accepted in practice — untested, and not to be worked around by misrepresenting affiliation. Whether the non-commercial restriction on LRS2/LRS3-trained checkpoints runs with the released weights — the plain reading says yes, but no explicit weight-licence statement was found. LRS2 per-split byte sizes — unknowable from outside, since it ships as one tar and divides on disk only into `pretrain/` and `main/`. The reason LRS3 was withdrawn — no primary source states one; claims circulating in third-party dataset cards are hearsay and were not relied on here.

**Note on method:** an automated summary of the LRS-VoxMM PDF returned split sizes and model names that did not match the paper. All figures in this document were re-extracted from the PDF text directly. Treat single-source restatements of these datasets' statistics with suspicion — the LRS3 licence discrepancy between the Oxford and KAIST pages is a second instance of the same hazard.
