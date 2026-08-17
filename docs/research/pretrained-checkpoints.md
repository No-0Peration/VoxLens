# Which pretrained visual-speech checkpoint is usable?

Research note for [issue #3](https://github.com/No-0Peration/VoxLens/issues/3).

## The question

VoxLens needs a published visual speech recognition checkpoint to build its first
Clip-to-Transcript pipeline on. A candidate has to clear four bars:

1. **Obtainable without academic affiliation.** No institutional request form, no
   proof-of-dataset-access email, no university address.
2. **Non-academic use permitted.** A research-only or non-commercial licence is
   disqualifying, not inconvenient.
3. **Video-only at inference.** Audio during training is fine and expected. A model
   that needs audio at inference is not a lip reader.
4. **Runs in 24 GB unified memory** on an Apple M4 Pro.

Accuracy is reported as **video-only WER on the LRS3 test set** throughout. The
LRS3 test set is 1,321 clips / 0.9 hours
([Auto-AVSR paper](https://arxiv.org/abs/2303.14307), §3.1).

> **The number that is easiest to get wrong.** Nearly every paper in this area
> headlines its *audio-visual* WER, because that is the number that wins the
> benchmark. Auto-AVSR's abstract leads with **0.9% WER on LRS3** — that is
> audio-visual, using clean audio. The same system's video-only WER is **19.1%**,
> twenty times higher. The `Visual_Speech_Recognition_for_Multiple_Languages`
> README goes further and mislabels its own results in prose: it reads
> *"19.1%, 1.0% and 0.9% WER for automatic, visual, and audio-visual"*, which
> pairs 19.1% with "automatic". Its own model-zoo table shows the opposite —
> Visual-only 19.1, Audio-only 1.0, Audio-visual 0.9. **Trust the tables, not
> the prose.** Every WER in this document is video-only unless explicitly marked.

## Comparison

| Candidate | Licence | Non-academic use | Weights gated? | Params | Video-only LRS3 WER | Framework |
|---|---|---|---|---|---|---|
| **Auto-AVSR** (`mpc001/auto_avsr`) | Code **Apache-2.0**; weights *unstated* | Code yes; weights unresolved | **No** — public Google Drive | 250.4 M | **20.3** (3,291 h ckpt) / 24.6 (1,759 h) | PyTorch + Lightning, bundled ESPnet |
| **BRAVEn / RAVEn** (`ahaliassos/raven`) | Code **MIT**; weights *unstated* | Code yes; weights unresolved | **No** — public Google Drive | Base 41 M / Base+ 93 M / **Large 328 M** | **20.1** (BRAVEn Large + self-training + LM) | PyTorch Lightning, bundled ESPnet |
| **VSRML** (`mpc001/Visual_Speech_..._Languages`) | Custom BSD **+ BBC non-commercial clauses** | **NO — disqualifying** | No | ~250 M (891 MB file) | 19.1 (with LM) | PyTorch, bundled ESPnet |
| **AV-HuBERT** (`facebookresearch/av_hubert`) | Meta licence, **non-commercial research only**, plus anti-surveillance/biometric clauses | **NO — disqualifying** | No (weights serve directly; repo archived Dec 2023) | Base 103 M / Large 325 M | 26.9 (Large, 433 h + self-training) | fairseq (archived, pinned to a 2021 commit) |
| **USR** (`ahaliassos/usr`) | **No licence file at all** | **NO — no rights granted** | No — public Google Drive | UNVERIFIED | 22.3 (Large) — superseded by USR 2.0 | PyTorch Lightning |
| **USR 2.0** (`ahaliassos/usr2`) | **CC BY-NC 4.0** | **NO — explicitly non-commercial** | **No** — public Google Drive | UNVERIFIED | **17.6** (Huge) — best verified open number | PyTorch Lightning |
| **Whisper-Flamingo** (`roudimit/whisper-flamingo`) | BSD-3 + MIT + **AV-HuBERT agreement pasted in** | **NO** | Ungated, but **no video-only checkpoint exists** | 2,497 M (Large-V2) | **None published** — see below | PyTorch |
| **VatLM** (`microsoft/SpeechT5/VATLM`) | MIT tag, but **AV-HuBERT-derived** — see below | **Contaminated — treat as no** | No, but **checkpoints fail to load** (2 open issues, 2+ yrs) | Base 107 M / Large 332 M | 28.4 (Large, best *downloadable*) | fairseq (same 2021 commit) |

Rows marked UNVERIFIED were not confirmed against a primary source in this pass and
must not be quoted as fact.

### Before anyone gets attached to these numbers

**LRS3 WER does not survive contact with real video.** The USR 2.0 README reports
its own model at **17.6% on LRS3** and **73.7% on WildVSR** — the same weights, a
harder test set. That is not a small degradation; it is the difference between a
working transcript and noise.

This matters more for VoxLens than for most projects, because VoxLens's stated aim
is **speaker-independent, unconstrained English**. LRS3 is TED talks: well-lit,
frontal, professionally shot, spoken by trained presenters. Every number in this
document was measured on that. Treat LRS3 WER as a **relative** yardstick for
comparing checkpoints, never as a prediction of what VoxLens will do on a webcam.
Whichever checkpoint is chosen, an early evaluation on harder footage should be a
priority — otherwise the first real-world test will be a shock.

## Two different licence questions — do not conflate them

The ticket asks whether **non-academic** use is permitted. That is a different
question from whether **commercial** use is permitted, and the two have different
answers. Keeping them apart is what makes this decision tractable.

**Axis 1 — can an unaffiliated individual use these? Yes.** Every checkpoint here
downloads publicly, and nothing requires a university address. Auto-AVSR's code is
Apache-2.0 and RAVEn's is MIT, both of which permit any use by anyone. VoxLens
clears this bar comfortably. **This is the bar the ticket set, and it is met.**

**Axis 2 — could VoxLens ship a commercial product on these weights? No, or
unresolved, for every candidate found — without exception.** This is not a property
of any one repository; it comes from the training data, and it therefore
contaminates the entire field:

- **LRS3 is built from TED talks**, and TED's
  [usage policy](https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy)
  is CC BY-NC-ND: **no commercial context, and no derivative works**.
  **There is a live contradiction in the first-party sources** about the dataset's
  own licence: the archived Oxford page said **CC BY-NC-ND 4.0**, while the current
  mmai.io / KAIST page says **CC BY 4.0** (permissive) — and, confusingly, also says
  "available to download for research purposes". Even taking the permissive reading
  at face value, it would cover the *annotations*; the underlying **video is TED's**
  and NC-ND regardless of how Oxford licensed the labels. So the conservative
  reading governs, and it is the one to plan against.
- **LRS2 and LRW are BBC-sourced** and stricter still (see below).

Every published VSR checkpoint of consequence is trained on LRS3, usually plus
LRS2/VoxCeleb2. So the honest conclusion is: **there is no cleanly
commercially-licensed lip-reading checkpoint available today.** That is a fact
about the field in August 2026, not a gap in this search.

**What this means for VoxLens right now:** the project is described in its README as
experimental, and the immediate goal is a spec for a clip-to-transcript CLI on a
pretrained checkpoint. For that — a technical baseline, private study, individual
research — the recommendation below is sound and the licences permit it. **Before
anything is sold or shipped commercially, this has to be revisited deliberately.**
It is a strategic constraint on the project, and it deserves its own ADR rather
than a line in a research note.

The rest of this section covers Axis 2 in detail, because getting it wrong is
expensive.

### The trap: two repos everyone treats as one

The two Pingchuan Ma repositories that everyone treats as "the Auto-AVSR models"
**carry different licences**, and the more popular one is the restrictive one.

### `Visual_Speech_Recognition_for_Multiple_Languages` — non-commercial

Its [LICENSE](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/blob/master/LICENSE)
is a BSD-3-Clause body with two extra clauses appended. Clause 5 reads, verbatim:

> Using our content to build a research prototype to compare with your
> organisation's own existing technology. This work must be done separately from
> your organisation's commercial product development. The BBC's content must not
> be used for training any existing or new technology, algorithms or models that
> will be sold commercially. And it must not be used to train technology to
> improve or enhance your operational systems. It can only be used for
> comparative or benchmarking purposes.

GitHub classifies this repo as `NOASSERTION` / "Other" — it is not a recognised
open-source licence. **Anything VoxLens ships commercially cannot be built on this
repository's weights.** This matters more than it looks, because this is the repo
that hosts the widely-cited **19.1%** visual-only checkpoint, and it is the one
that downstream tools reach for by default.

**Concretely:** [Chaplin](https://github.com/amanvirparhar/chaplin) (MIT, 752 stars,
a real-time Mac lip-reading tool) is itself MIT-licensed but its
[`setup.sh`](https://github.com/amanvirparhar/chaplin/blob/main/setup.sh) downloads
`LRS3_V_WER19.1` — the checkpoint from *this* restrictively-licensed repo, re-hosted
on a third-party Hugging Face account. An MIT wrapper around a non-commercial
checkpoint does not make the checkpoint MIT. Copying Chaplin's setup would import
the restriction silently.

### AV-HuBERT — non-commercial research only

The [AV-HuBERT LICENSE](https://github.com/facebookresearch/av_hubert/blob/main/LICENSE)
is a bespoke Meta agreement. Its grant clause reads:

> Meta grants you a non-exclusive, worldwide, non-transferable, non-sublicensable,
> revocable, royalty free and limited license under Meta's copyright interests to
> reproduce, distribute, and create derivative works of the Software **solely for
> your non-commercial research purposes**.

Note also `revocable` and `non-transferable`. Worse for this project specifically,
§2(a) forbids using the Software **"or any data produced by the Software"** for

> (i) any commercial or production purposes, (ii) military purposes or in the
> service of nuclear technology, (iii) purposes of surveillance, including any
> research or development relating to surveillance, (iv) biometric processing

Clauses (iii) and (iv) point directly at a lip-reading system, and **"any data
produced by the Software" reaches the model's outputs**, not just the weights. So
even a strictly non-commercial VoxLens would have to argue it is neither
surveillance research nor biometric processing. **Disqualifying twice over.**
Separately, the repository was **archived on 2023-12-07**, and its `fairseq`
submodule is pinned to a **June 2021** commit of a project that is itself now
archived — a dependency situation that is painful before licence even enters into
it.

For completeness, since the numbers are otherwise hard to find stated correctly:
AV-HuBERT is **103 M (Base) / 325 M (Large)** parameters, and its **video-only**
LRS3 WER is **28.6%** (Large, 433 h labelled, 1,759 h pretrain) or **26.9%** with
self-training. Do not confuse these with the Interspeech-2022 robustness paper,
whose headline 5.8% / 14.1% figures are **audio-visual under noise**.

**Note on access:** AV-HuBERT's weights are *not* actually gated. The
`facebookresearch.github.io/av_hubert` page shows a client-side "I Accept" dialog,
but the underlying `dl.fbaipublicfiles.com` URLs serve directly and unauthenticated
(verified `HTTP/2 200` on `base_lrs3_433h.pt`, ~1.93 GB, without accepting
anything). This is worth stating precisely because it is the opposite of the usual
assumption: **the gate on AV-HuBERT is the licence, not the download.** Being able
to fetch a file is not permission to use it.

### The datasets themselves — LRS2/LRW exclude you by name

The live [BBC R&D lip-reading datasets page](https://www.bbc.co.uk/rd/projects/lip-reading-datasets)
governs LRS2 and LRW:

> The datasets are available to researchers from universities and other reputable
> academic institutions and relevant public organisations, **strictly for
> non-commercial research**. Use is not permitted by commercial organisations.

> **You must use your official academic email address** (normally a .edu address)
> so we can validate you. Gmail or non academic email accounts are not acceptable.

> **Use is not permitted by companies or independent researchers.**

An unaffiliated individual is *categorically* excluded — not gated, excluded. This
does not block using a checkpoint someone else trained, but it does mean VoxLens can
never retrain on, or verify against, LRS2/LRW.

**LRS3 is no longer distributed at all.** The official Oxford page
(`robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html`) **404s**, as do the data files
and even the annotation/URL lists that would let you rebuild it from YouTube. The
[mmai.io mirror](https://mmai.io/datasets/lip_reading/) states: *"Downloads are no
longer available from this website."* The old password-request form 404s too. LRW
and LRS2 remain served (HTTP 401, password-gated); LRS3's directory is simply gone.

The practical consequence is significant: **retraining a clean-provenance
replacement is not currently possible through official channels.** Unauthorised
re-uploads of LRS3 and LRS2 exist on Hugging Face; using them would breach the BBC
terms outright and is not a route this note recommends.

### USR — no licence is worse than a bad licence

[`ahaliassos/usr`](https://github.com/ahaliassos/usr) has **no licence file** (the
GitHub API returns `license: null`). Absent a licence, no rights are granted at all
— default copyright applies. Unusable regardless of its benchmark numbers, which
were competitive (**22.3%** video-only on LRS3, Large, high-resource).

It is in any case **superseded by USR 2.0**, which does carry a licence — CC BY-NC
4.0 — and is both more accurate and more clearly governed. See the USR 2.0 entry
below; if the licence permits your use, there is no reason to reach for v1.

### Third-party Hugging Face mirrors do not launder licences

Mirrors such as `nguyenvulebinh/auto_avsr_visual_trlrwlrs2lrs3vox2avsp_base` and
`Amanvir/LRS3_V_WER19.1` re-host these weights. The
[nguyenvulebinh model card](https://huggingface.co/nguyenvulebinh/auto_avsr_visual_trlrwlrs2lrs3vox2avsp_base)
states its licence as `[More Information Needed]` — i.e. **no licence at all**. A
mirror cannot grant rights the uploader never held. Treat every mirror as carrying
the licence of the original checkpoint, and prefer the upstream source.

## Candidates in detail

### 1. Auto-AVSR — `mpc001/auto_avsr` (recommended)

- **Repo:** https://github.com/mpc001/auto_avsr
- **Paper:** Ma et al., *Auto-AVSR: Audio-Visual Speech Recognition with Automatic
  Labels*, ICASSP 2023 — https://arxiv.org/abs/2303.14307
- **Licence:** Code is **Apache-2.0** (GitHub-detected; per-file headers read
  `Apache 2.0`). The README's licence section says:
  > Code is Apache 2.0 licensed. The pre-trained models provided in this repository
  > may have their own licenses or terms and conditions derived from the dataset
  > used for training.

  Read this carefully: it is a **disclaimer, not a grant**. The code is
  unambiguously Apache-2.0. The weights are left explicitly unresolved. This is
  better than VSRML's active prohibition and better than AV-HuBERT's explicit
  non-commercial clause, but it is *not* an affirmative permission. See
  "Residual licence risk" below.
- **Weights gated?** **No.** The [model zoo](https://github.com/mpc001/auto_avsr#model-zoo)
  links straight to public Google Drive files. No form, no email, no proof of LRS3
  access. Verified: the links return Google Drive's standard large-file "Virus scan
  warning" interstitial rather than a sign-in wall, meaning they are publicly
  readable (fetch with `gdown`, which handles the confirm token).
- **Caution — the mirror in the tutorial is dead.** The `inference.ipynb` tutorial
  fetches weights from `http://www.doc.ic.ac.uk/~pm4115/autoAVSR/...`. Every URL on
  that host now **404s** (verified). Only the Google Drive links work. The tutorial
  will fail as written.

**Released video-only checkpoints** (from the repo's model zoo table):

| Checkpoint | Training data | Video-only LRS3 WER | Params | Training-data provenance |
|---|---|---|---|---|
| `vsr_trlrs3_23h_base.pth` | 23 h subset | 93.0 | 250 M | LRS3 |
| `vsr_trlrs3_base.pth` | 438 h | 36.0 | 250 M | LRS3 |
| `vsr_trlrs3vox2_base.pth` | 1,759 h | **24.6** | 250 M | LRS3 + VoxCeleb2 |
| `vsr_trlrs2lrs3vox2avsp_base.pth` | 3,291 h | **20.3** | 250 M | **+ LRS2 (BBC)** + AVSpeech |

The paper's own best video-only figure is **19.1%** at 3,448 h
([Table 2](https://arxiv.org/abs/2303.14307), 100% column). The best *downloadable*
checkpoint reports **20.3%** at 3,291 h — slightly worse than the paper, because it
is trained on slightly less data. Quote 20.3%, not 19.1%, when describing what you
can actually run from this repo.

The paper's Table 2 also gives the data-scaling curve for video-only WER, which is
useful for judging what more data would buy:

| Extra unlabelled data | 0 h | 526 h | 1,052 h | 1,578 h | 2,104 h | 2,630 h |
|---|---|---|---|---|---|---|
| Total training data | 818 h | 1,344 h | 1,870 h | 2,396 h | 2,922 h | 3,448 h |
| **Video-only WER** | 33.0 | 26.6 | 23.6 | 21.9 | 20.0 | **19.1** |

- **Parameters:** **250.4 M** for the visual-only model (paper §3.3, stated
  exactly). Breakdown given there: VSR front-end 11.2 M, Conformer back-end
  170.9 M, Transformer decoder 64.5 M, CTC projection layer 3.9 M. This is
  corroborated by architecture arithmetic over the released code
  (`espnet/nets/pytorch_backend/e2e_asr_conformer.py`) and by the 891 MB on-disk
  size of the equivalent VSRML checkpoint.
- **Architecture:** 3D-conv stem (5×7×7, stride 1×2×2) + ResNet-18 visual front-end
  → `Linear(512, 768)` → 12-layer Conformer (768 dim, 3072 FF, kernel 31) →
  6-layer Transformer decoder, joint CTC/attention, SentencePiece unigram-5000
  vocabulary.
  *Minor discrepancy:* the paper says 16 attention heads, the released code says 12
  (`attention_heads=12`). Head count does not change parameter count, so both are
  compatible with the same `state_dict`; the code is authoritative for inference.
- **Framework:** PyTorch + `pytorch-lightning`, with a **vendored copy of ESPnet**
  inside the repo (`espnet/`). Dependencies are light and current:
  `torch torchvision torchaudio pytorch-lightning sentencepiece av`. There is **no
  fairseq dependency**, which is a meaningful practical advantage over AV-HuBERT —
  no pinned-old-PyTorch dependency hell.
- **Streaming:** not applicable. The Conformer is **non-causal** — symmetric
  depthwise-conv padding (`padding=(kernel_size-1)//2`), relative positional
  encoding, full-sequence attention — and decoding is full-sequence beam search.
  This is a Clip model. Per [ADR-0001](../adr/0001-clips-first-streaming-target.md)
  that makes it a legitimate first baseline and explicitly a feasibility probe, not
  a step toward the Stream product.

### 2. BRAVEn / RAVEn — `ahaliassos/raven` (fallback)

- **Repo:** https://github.com/ahaliassos/raven
- **Papers:** RAVEn, ICLR 2023 — https://arxiv.org/abs/2212.06246 ;
  BRAVEn — https://arxiv.org/abs/2404.02098
- **Licence:** **MIT** (GitHub-detected `MIT License`). This is a cleaner *code*
  licence than Auto-AVSR's Apache-2.0, and much cleaner than VSRML. As with
  Auto-AVSR, the repo does not attach an explicit separate licence to the weights.
- **Weights gated?** **No** — public Google Drive links in the README, no form.
- **Video-only LRS3 WER.** The README splits results by labelled-data setting, and
  the distinction matters: **high-resource** = fine-tuned on the full LRS3 433 h;
  **low-resource** = fine-tuned on a 30 h "trainval" subset. The high-resource
  video-only numbers are the ones to compare against Auto-AVSR:

  | Model | Pre-training data | Video-only WER |
  |---|---|---|
  | RAVEn Large + self-training + LM | LRS3 + Vox2-en | 23.1 |
  | BRAVEn Large | LRS3 + Vox2-en + AVS | 23.6 |
  | BRAVEn Large + self-training | LRS3 + Vox2-en + AVS | 20.9 |
  | **BRAVEn Large + self-training + LM** | LRS3 + Vox2-en + AVS | **20.1** |

  The README warns that some models were retrained, so figures may differ slightly
  from the papers. The 20.1% figure **requires the external language model** (also
  linked from the README); without it the same checkpoint is 20.9%.
- **Parameters:** **Base 41 M**, **Base+ 93 M**, **Large 328 M** (RAVEn paper's
  setup section; matches `conf/model/visual_backbone/*.yaml`). **Caveat: these count
  the encoder blocks only** and exclude the ~11 M ResNet-18 front-end and the
  decoder, so the deployable total is higher. Large is still comfortably inside
  budget. Note RAVEn **Base (41 M) is not comparable to AV-HuBERT Base (103 M)**;
  only the Large variants are like-for-like.
- **Framework:** PyTorch Lightning with a vendored ESPnet and Hydra configs — and
  crucially **no archived-fairseq submodule**, which makes it materially easier to
  stand up than AV-HuBERT or VatLM. The pinned environment is still 2022-era
  (`python=3.8.13`, `pytorch=1.11.0`).
- **Significant limitation: inference code only.** The README states *"Code for
  pre-training and fine-tuning coming soon..."* — still true as of the last push
  (2025-02-27); `scripts/` contains only `testing/`. Fine for running a checkpoint,
  not for adapting one.
- **Licence caveats worth knowing before leaning on the MIT tag:** the copyright is
  **personal** (`Copyright (c) 2023 ahaliassos`), not an institutional grant, and
  several co-authors are Meta-affiliated — whether Imperial or Meta has any claim is
  **UNVERIFIED**. No separate weights licence exists, so whether MIT extends to the
  checkpoints is **UNVERIFIED**. The repo also vendors Apache-2.0 ESPnet with intact
  file headers but **ships no LICENSE/NOTICE for it** — a hygiene defect to fix
  before redistributing anything built on it.
- **Practical gotchas:** the README's script paths are wrong — the real ones are
  `scripts/testing/vsr/...`. On the plus side the repo **ships the LRS3 test
  manifest and subword vocabularies**, so reproducing its numbers needs only the
  LRS3 *test* videos, not the full corpus. Mouth crops are written as **lossless
  FFV1 `.avi`**, which is a better archival choice than AV-HuBERT's lossy MP4.

### 3. USR 2.0 — the accuracy ceiling, and explicitly non-commercial

- **Repo:** https://github.com/ahaliassos/usr2 (ICLR 2026), superseding
  [`ahaliassos/usr`](https://github.com/ahaliassos/usr) (NeurIPS 2024)
- **Licence: CC BY-NC 4.0.** The LICENSE file reads *"Creative Commons
  Attribution-NonCommercial 4.0 International / Copyright (c) 2026 Alexandros
  Haliassos"*, covering code and release alike, with no separate weights grant.
  **Explicitly non-commercial — disqualifying for a commercial product**, though it
  does permit non-academic, non-commercial use.
- **Weights:** genuinely ungated public Google Drive. No form, no proof of dataset
  access.
- **Video-only LRS3 WER** (README "VSR (%)" column, high-resource): Base+ 24.8,
  Large 21.5, **Huge 17.6** — the best verified open figure found anywhere in this
  sweep. **And 73.7% on WildVSR**, per the same README.
- **No AV-HuBERT dependency** — PyTorch Lightning + Hydra + vendored ESPnet, plain
  `.pth`. Ships `demo.py` with `modality=v` for lip-reading-only, so it is unusually
  easy to try.
- **Preprocessing: the same family contract** — `crop_width=96, crop_height=96,
  start_idx=48, stop_idx=68, window_margin=12`, warped to `20words_mean_face.npy`;
  `random_crop_dim: 88`; `mean 0.421, std 0.165`; 25 fps. MediaPipe is the default
  detector (a `face_landmarker.task` is bundled), RetinaFace + FAN optional.
- **Parameters: UNVERIFIED** — not published. The Huge config is `adim 1280`, 32
  encoder layers, 9 decoder layers, `eunits 5120`, which is substantially larger
  than Auto-AVSR; budget accordingly.

**How to think about it:** if VoxLens ever commits to being non-commercial, USR 2.0
is the obvious choice — best accuracy, ungated, no AV-HuBERT contamination, and the
same Mouth Region format as everything else here. It is excluded from the primary
recommendation only because its licence forecloses a commercial path *explicitly*,
where Auto-AVSR's merely leaves it unresolved.

### 4. Whisper-Flamingo — cannot do video-only

- **Repo:** https://github.com/roudimit/whisper-flamingo

Whisper-Flamingo is an **audio-visual** model: it fuses visual features into a
Whisper backbone to make *audio* recognition robust to noise. That is the opposite
of VoxLens's problem. Three independent reasons it is out:

1. **No video-only checkpoint ships.** The code does contain a `vsr` branch
   (`whisper_decode_video.py`, with `test_v=True` skipping the mel front-end), but
   that path is only coherent under an `av_fusion="lip-reader"` config, and **no
   such config is released** — `config/audio-visual/` contains only `separate`
   fusion variants, and every released checkpoint is labelled audio or audio-visual.
2. **No video-only WER is published**, in either the original paper
   ([arXiv:2406.10082](https://arxiv.org/abs/2406.10082), which reports AVSR 0.76% /
   ASR 0.68% on LRS3) or the successor **mWhisper-Flamingo** (IEEE SPL 2025). The
   successor does train with 50% video-only inputs via decoder modality dropout, yet
   still publishes no video-only number. You would be flying blind.
3. **Licence.** The README claims BSD-3, which is misleading: the actual LICENSE
   file is BSD-3 **plus MIT (Whisper) plus the full AV-HuBERT LICENSE AGREEMENT
   pasted in**, and the AV decoding path requires downloading AV-HuBERT weights.
   Fine-tuned on MuAViC (CC BY-NC 4.0). Non-commercial, unambiguously.

Also impractical: Large-V2 is **2,497 M** parameters (~5 GB fp16).

**Excluded on capability grounds, before licence even matters.**

### 5. VatLM — an MIT tag that does not mean what it looks like

- **Repo:** https://github.com/microsoft/SpeechT5/tree/main/VATLM (there is no
  `X-LANCE/VatLM` — that URL 404s).
- **Licence:** the SpeechT5 root LICENSE is **MIT, Copyright (c) Microsoft
  Corporation**, with no per-subfolder licence and no non-commercial language.
  **Do not read that as a clean green light.** VATLM's own README says *"Portions of
  the source code are based on the FAIRSEQ and av_hubert"*; its source headers carry
  `Code based on … av_hubert` above the Microsoft MIT notice, and some files still
  carry the verbatim `Copyright (c) Facebook, Inc. and its affiliates.` header. It
  pins **the same fairseq commit as AV-HuBERT**, and its pre-training targets are
  k-means units produced by *"the AV-HuBERT model pre-trained in the fourth
  iteration"* — squarely "data produced by the Software" under Meta's §2(a).
  MIT covers Microsoft's own contribution; it does not on its face dispose of the
  upstream Meta terms. Whether Microsoft obtained a separate grant is **UNVERIFIED**.
  **Treat as contaminated.**
- **Weights:** 13 ungated Google Drive links — but **the fine-tuned checkpoints do
  not load.** Two open, unanswered issues report this
  ([#54](https://github.com/microsoft/SpeechT5/issues/54), open since June 2023 with
  zero comments; [#88](https://github.com/microsoft/SpeechT5/issues/88)). The cause
  is visible in `vathubert_asr.py`: the released checkpoints have `w2v_args=None`
  and a `w2v_path` pointing at a dead Microsoft-internal mount. A `model.w2v_path`
  override is the plausible workaround but is undocumented and **UNVERIFIED**.
  The repo also does not run as checked in — a required dataset module was never
  committed, and several packages are missing `__init__.py`.
- **Parameters:** 107 M (Base) / 332 M (Large).
- **Video-only LRS3 WER:** **28.4%** (Large, 433 h labelled) is the best figure with
  a *released* checkpoint; the headline **26.2%** uses self-training weights that
  were never published. Traps in the paper's Table I: the adjacent column is
  **audio-visual** (1.2–3.6%) and is easily misquoted as lip-reading, and Table II's
  24.3% is **LRS2, not LRS3**.

**Conclusion: not worth pursuing.** It is licence-contaminated by AV-HuBERT, its
best downloadable video-only number (28.4%) is worse than Auto-AVSR's and RAVEn's,
and its checkpoints have had a known loading bug unanswered for over two years.

### 6. The 2024–2026 wave — surveyed, none usable

A sweep of newer work found better *numbers* but no better *licences*. Recorded so
nobody re-runs this search:

| Model | Licence | Video-only LRS3 WER | Why it's out |
|---|---|---|---|
| **VALLR** (ICCV 2025) | CC BY-NC 4.0 (badge only, **no LICENSE file**) | **18.7** (from 30 h labelled) | Non-commercial; the stage-2 Llama LoRA that turns phonemes into words **was never released**, so the published checkpoint cannot reproduce 18.7%. Also 224×224 **colour**, 16-frame clips — incompatible with the 96→88 greyscale mainstream. |
| **DLLM-VSR** (May 2026) | Code MIT; adapters MIT on HF | **19.5** (USR 2.0 backbone) | The MIT stamp covers only ~110 MB of LoRA adapters. The deployable system needs Dream-7B **plus** USR 2.0 (CC BY-NC) or AV-HuBERT (Meta NC). Non-commercial either way; ~16 GB fp16. |
| **SyncVSR** (Interspeech 2024) | **MIT** | 23.4 | **Weights are not obtainable.** Release tags exist but the releases list is empty and both tag lookups 404. A permissive licence on files nobody can download. |
| **Llama-AVSR** (ICASSP 2025) | **None** (no LICENSE, API 404) | 23.68 | No licence = no rights. Built on AV-HuBERT Large + gated Llama 3.1-8B (~17 GB fp16). |
| **Omni-AVSR** (ICASSP 2026) | **None** (no LICENSE) | UNVERIFIED (README results are images) | Same — no licence, AV-HuBERT-based. |
| **VSP-LLM** | **AV-HuBERT agreement verbatim** | 25.4–29.8 | Strictest licence here: non-commercial plus explicit surveillance and biometric bars. Needs gated LLaMA2-7B. |
| **ViSpeR** (TII) | CC BY-NC | **No LRS3-only English number exists** | Non-commercial. Its widely-quoted 49.1 combines LRS3 *and* WildVSR — do not cite it as LRS3. |
| **CoGenAV** | **None** (no LICENSE, code or weights) | **No LRS3 visual-only number exists** | No licence. Its VSR figures are **LRS2**; the paper says LRS3 evaluation awaits the dataset becoming available again. |

Two patterns worth internalising. First, **the frontier moved to LLM-backed
systems**, which trade a small WER gain for a 7B-parameter dependency and a second
licence to satisfy — a bad trade for an on-device Apple Silicon target. Second,
**almost every 2024–2026 system builds on AV-HuBERT**, so Meta's non-commercial and
anti-biometric clauses propagate through most of the field. The two repositories
recommended here are notable precisely for *not* depending on it.

## Input preprocessing — the part that constrains the Mouth Region extractor

This is the highest-value finding in this note.

**Auto-AVSR, BRAVEn/RAVEn, and the VSRML models all expect the *same* Mouth Region
format.** One extractor serves the recommendation and the fallback. Verified by
reading the source of both repositories, not from documentation — the papers state
only "a bounding box of 96 × 96".

**But be precise about how far that sameness goes.** The *tensor* contract —
25 fps, 96 → 88 px, greyscale, ÷255, `mean 0.421 / std 0.165`, mouth centre from
landmarks 48–68 — is identical across **all four** families, AV-HuBERT and VatLM
included. The *alignment* is not:

| | Landmark source | Stable points for the warp |
|---|---|---|
| **Auto-AVSR, RAVEn/BRAVEn, VSRML** | RetinaFace + iBUG FAN (68) | **8**: (28, 33, 36, 39, 42, 45, 48, 54) |
| **AV-HuBERT, VatLM** | **dlib** (68) | **5**: (33, 36, 39, 42, 45) |

So Auto-AVSR and RAVEn crops are genuinely interchangeable — which is what makes the
first-choice/fallback pairing cheap. AV-HuBERT-style crops are *not* a drop-in for
them, despite the identical normalisation constants. If you ever compare against
those models, regenerate the crops rather than reusing them.

**A trap common to every one of these repos:** the crop size, the 88 px model input
and the `0.421 / 0.165` constants appear **only in code and config**, never in the
papers, which say just "96 × 96" and "converted to greyscale". Implementing any of
these from the paper alone gets both the resolution and the normalisation wrong.

### The contract

| Property | Value |
|---|---|
| Frame rate | **25 fps** |
| Stored crop | **96 × 96** px, **RGB** (see the colour note below — this surprises people) |
| Model input | **88 × 88** px, **greyscale, 1 channel** (centre crop at inference) |
| Scaling | divide by 255 → [0, 1] |
| Normalisation | `Normalize(mean=0.421, std=0.165)` — a single scalar pair, applied after greyscale |
| Alignment | similarity warp of the whole frame to a 68-point mean face, then crop |

**The colour ordering is the easy mistake.** The Mouth Region extractor should emit
**96 × 96 RGB**, *not* greyscale. Auto-AVSR warps and crops in colour and converts
to greyscale only inside the model's input transform. Verified in source:
`preparation/preprocess_lrs2lrs3.py` line 97 constructs the loader with
`convert_gray=False`, and the inference tutorial does the same. The
`convert_gray=True` default on `VideoProcess.__init__` is never exercised by
Auto-AVSR's own entry points. Greyscaling at crop time instead would warp
already-flattened pixels and does not reproduce the published numbers.

### Where each value comes from

- **25 fps.** Stated in the Auto-AVSR paper (Fig. 1 caption: *"The frame rate of
  audio and visual features from the ASR and VSR encoders is 25 frames per second
  (fps)"*), and pinned in code by `rate_ratio=640` in
  [`datamodule/av_dataset.py`](https://github.com/mpc001/auto_avsr/blob/main/datamodule/av_dataset.py)
  — 16000 Hz ÷ 640 = 25. Chaplin's config states it explicitly as `v_fps=25`.
  **Gotcha:** the loading code (`torchvision.io.read_video`) does **not** resample.
  A Clip at 30 fps will be silently fed at the wrong rate and degrade. VoxLens must
  resample to 25 fps itself.
- **96 → 88, greyscale, 0.421/0.165.** From
  [`preparation/detectors/retinaface/video_process.py`](https://github.com/mpc001/auto_avsr/blob/main/preparation/detectors/retinaface/video_process.py)
  (`crop_width=96, crop_height=96`) and
  [`preparation/transforms.py`](https://github.com/mpc001/auto_avsr/blob/main/preparation/transforms.py),
  whose test/val pipeline is exactly:
  ```python
  x / 255.0  →  CenterCrop(88)  →  Grayscale()  →  Normalize(0.421, 0.165)
  ```
  RAVEn independently confirms the same constants in Hydra config:
  `conf/data/channel/grayscale.yaml` gives `mean: [0.421]`, `std: [0.165]`,
  `in_video_channels: 1`, and `conf/data/crop_type/mouth.yaml` gives
  `random_crop_dim: 88`.

### The alignment procedure (68-landmark path)

From `video_process.py`, in order:

1. **Detect** face + **68 landmarks** per Frame. Note the division of labour, which
   the name obscures: **RetinaFace (ResNet-50, threshold 0.8) is only the face
   detector**; the 68 landmarks come from **iBUG FAN (`2dfan2`, `input_size=256`)**
   (`preparation/detectors/retinaface/detector.py`). When several faces are present
   it keeps the **largest bounding box** — VoxLens's one-Speaker-per-Clip rule
   matches this, but it is a silent heuristic, not a guarantee.
2. **Interpolate** Frames where detection failed, linearly between the nearest
   valid neighbours; Frames before the first / after the last detection are filled
   by copying the nearest valid landmark set. *(This is where VoxLens's Occlusion
   handling will need its own policy — the reference code silently papers over
   missing landmarks rather than reporting them.)*
3. **Temporally smooth** the landmarks over a **12-Frame window** (`window_margin=12`,
   so ±6), then re-centre the smoothed set on the current Frame's centroid.
4. **Greyscale** the Frame (`cv2.COLOR_RGB2GRAY`) — in the offline path this happens
   *before* the warp.
5. **Similarity warp** the whole Frame to a **256 × 256** canvas, fitting
   `cv2.estimateAffinePartial2D(..., method=cv2.LMEDS)` from **8 stable landmarks**
   — indices **(28, 33, 36, 39, 42, 45, 48, 54)** in the 68-point iBUG convention
   (nose bridge, nose tip, eye corners, mouth corners) — onto the corresponding
   points of the reference mean face.
6. **Crop** a 96 × 96 patch centred on the **mean of landmarks 48–68** (the 20 mouth
   points).

There is **no pre-resize step** — the warp maps the source-resolution Frame directly
onto the 256 × 256 canvas in one operation.

The reference face is `20words_mean_face.npy`, a **68 × 2 float64** array shipped in
the repo (1168 bytes = 80-byte npy header + 68×2×8), with coordinates spanning
x ∈ [70.92, 194.44], y ∈ [72.64, 200.53] inside that 256 × 256 frame. It is
**byte-identical** (`sha b659613a…`) across Auto-AVSR's two detector folders, the
VSRML equivalents, and RAVEn — so the alignment target is genuinely shared across
the whole family.

### Porting notes for whoever writes the extractor

- **Do not take a scikit-image dependency.** `video_process.py` imports
  `skimage.transform` and defines `warp_img()` / `apply_transform()`, but **neither
  is ever called**. The live path is `cv2.estimateAffinePartial2D` +
  `cv2.warpAffine` only.
- **Take the code from `auto_avsr` (Apache-2.0)**, not from the VSRML sibling
  (`NOASSERTION`, BBC-restricted). The preprocessing behaviour is identical and the
  mean-face asset is literally the same git blob — so there is no reason to copy
  from the restrictively-licensed repo.
- **There is a live bug on the Apple Silicon path.**
  `preparation/preprocess_lrs2lrs3.py` line 94 reads:
  ```python
  if args.gpu_type != "cuda" or "mps":
      raise ValueError(...)
  ```
  `or "mps"` is a truthy constant, so the guard **raises unconditionally**
  regardless of the flag passed. It should be `if args.gpu_type not in ("cuda", "mps"):`.
  This is exactly the code path a Mac build touches first.
- Clips shorter than **12 Frames** are dropped outright by the RetinaFace variant
  (`window_margin` guard) — a floor on usable Clip length.
- Auto-AVSR applies **no horizontal flip**; its training augmentation is random crop
  + adaptive time masking only. (The paper mentions flipping; the released code does
  not do it. Training-only, so it does not affect inference, but do not "restore" it
  when porting.)

### The MediaPipe path is a *different* geometry

Auto-AVSR ships a second detector path, and it is **not** a drop-in equivalent.
`preparation/detectors/mediapipe/detector.py` returns only **4 keypoints** (right
eye, left eye, nose tip, mouth centre), not 68. Accordingly its `video_process.py`
uses `stable_points=(0,1,2,3)` and crops centred on **keypoint 3 alone**
(`start_idx=3, stop_idx=4`) rather than the mouth-landmark centroid. Its 4-point
reference is derived from the same 68-point mean face by averaging groups:
`mean(ref[36:42])`, `mean(ref[42:48])`, `mean(ref[31:36])`, `mean(ref[48:68])`.

This matters for VoxLens: **MediaPipe is the practical Apple Silicon route** (it
avoids the CUDA-oriented `ibug` RetinaFace/FAN stack — Chaplin uses exactly this),
but it produces a measurably different crop from the one the checkpoints were
trained on. Expect some accuracy loss versus the published numbers, and treat
"which detector" as a variable to evaluate rather than an implementation detail.

### The two repos disagree on where greyscale happens

Worth knowing before debugging a WER gap, because both are "correct" for their own
checkpoints:

- **Auto-AVSR** — `convert_gray=False` in *both* the offline dataset prep and the
  inference tutorial. Warp and crop in **colour**, store RGB, then `Grayscale()`
  inside `VideoTransform`. **This is the path the published Auto-AVSR WERs
  correspond to, and the one VoxLens should implement.**
- **VSRML** — `pipelines/data/data_module.py` defaults to `convert_gray=True`:
  greyscale *before* the warp, and correspondingly its `VideoTransform` has **no**
  `Grayscale()` step at all.

Both end at a single-channel 88×88 tensor, but the pixel values differ slightly.
Feeding a VSRML-style greyscale-first crop into the Auto-AVSR transform would
double-handle the conversion; feeding an Auto-AVSR RGB crop into the VSRML
transform would leave three channels where one is expected. Tensor layout differs
too: Auto-AVSR feeds `(T, 1, 88, 88)`; VSRML feeds `(1, T, 88, 88)`.

## Memory and Apple Silicon feasibility

**Memory is not the constraint — the toolchain is.** These figures were measured on
an actual M4 Pro with 24 GiB (`hw.memsize = 25,769,803,776`), macOS 26.2.

**The real budget is 17.76 GiB, not 24.** Metal reports
`recommendedMaxWorkingSetSize = 19,069,665,280 B` (17.76 GiB, ~74% of RAM) — Apple
has already discounted for the OS and other GPU clients. There is also a
single-tensor hard cap, `maxBufferLength = 13.32 GiB`.

Against that budget, at an exact **250,411,442** parameters (read from the
safetensors index of a checkpoint mirror, corroborating the paper's 250.4 M):

| | fp32 | fp16 |
|---|---|---|
| Weights | **0.933 GiB** | 0.466 GiB |
| Activations, 100 Frames (4 s) | ~151 MiB | ~75 MiB |
| Activations, 250 Frames (10 s) | ~381 MiB | ~190 MiB |
| Activations, 500 Frames (20 s) | ~773 MiB | ~386 MiB |

**A 20-second Clip needs roughly 1.8 GiB total — about 10% of the budget.** Huge
margin. Two results here are counterintuitive and worth carrying into design work:
the **front-end dominates** (~95% of activation memory, from the ResNet stage at
22² × 64 per Frame), and **O(T²) attention is negligible** — 12 heads × 500² × 2 B
is 5.7 MiB per block. Attention only becomes a memory factor around T ≈ 10,000, so
long Clips are cheap. (Weights are exact; activation figures are estimates.)

### The actual risks are two PyTorch-version gotchas

Both land squarely on this model, so they are worth stating precisely.

1. **The Conv3d front-end hits a known slow path.** PyTorch issue
   [#192213](https://github.com/pytorch/pytorch/issues/192213) (Aug 2026) measured
   `F.conv3d` on Apple Silicon at **7% of conv2d throughput** (4.12 vs 62.26
   TFLOP/s) while conv2d and matmul both reach hardware peak. The cause, named in
   the fix ([#192229](https://github.com/pytorch/pytorch/issues/192229)): the fast
   Metal kernels were only instantiated for convolutions **with a bias term**, and
   bias-free convolutions fall to a slow SIMD path. **The VSR front-end is exactly
   that case** — both repos declare it bias-free:
   `auto_avsr/espnet/nets/pytorch_backend/frontend/resnet.py` uses
   `nn.Conv3d(1, …, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False)`.
   On PyTorch ≤ 2.13 the stem therefore runs at roughly a tenth of achievable
   throughput. The fix restores 82–85%.
2. **CTC loss on MPS is 2.14-only.** `ctc_loss_mps` appears **zero** times in
   `native_functions.yaml` at tag `v2.13.0` and **once** at `v2.14.0-rc3`. On stable
   2.13 it either errors or silently round-trips to CPU. Auto-AVSR decodes with
   hybrid CTC/attention (`ctc_weight 0.1`), so this is on the inference path.

**Conclusion: build on PyTorch 2.14 or nightly, not stable 2.13.** That single
choice removes both problems.

Two further configuration notes: PyTorch's MPS allocator defaults to
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.7`, which permits allocating **more than
physical RAM** before it OOMs — so it will swap rather than fail; set it to ~0.9.
And **float64 is a hard type error on MPS, not a fallback** — `PYTORCH_ENABLE_MPS_FALLBACK`
will not rescue it, which matters because the landmark/mean-face code is
float64-based and must be kept on CPU/NumPy.

Nothing conformer-relevant appears on PyTorch's most-requested-missing-MPS-ops
list; the remaining gaps there are linear algebra. SDPA is native, though it has no
dedicated MPS backward — fine for inference, weak if fine-tuning is ever attempted.

Supporting real-world evidence: Auto-AVSR's preprocessing entry point takes
`--gpu_type mps`, and [Chaplin](https://github.com/amanvirparhar/chaplin) runs this
model class **interactively on a Mac on CPU alone** (its `main.py` selects `cuda` or
falls back to `cpu`, with no MPS path at all) — so even the unaccelerated path is
usable, and MPS is upside.

Whether the full inference graph runs cleanly on the PyTorch **MPS** backend
(rather than CPU) was not verified in this pass — **UNVERIFIED**, and worth a
spike. The operator to check first is **`Conv3d`**, used by the 5×7×7
spatio-temporal stem: it is the load-bearing op in the visual front-end and the
most likely source of a gap. MPS operator gaps typically surface as silent CPU
fallbacks rather than errors, so measure, don't assume.

### There is no existing Apple-Silicon-native port

Checked and confirmed negative, so nobody repeats the search:

- **MLX: nothing exists.** No lip-reading, visual-speech, AVSR, or AV-HuBERT model
  in `mlx-community`, `ml-explore/mlx-examples`, or `mlx-vlm`.
- **Core ML: three ports, all compromised.** `ebowwa/silentvsr-models` is an
  Auto-AVSR → `.mlpackage` conversion (88×88 greyscale, 25 fps, ~250 M) but is
  **gated** and stamps MIT over an upstream that never granted it — that MIT is
  void. `rkmtlab/LipLearner` (MIT) ships a genuine `LipEncoder.mlpackage` but does
  few-shot *command matching*, not open-vocabulary transcription. The third is a
  25-word Japanese classifier with no licence.
- **An ONNX export exists** (`HereLiesAz/liperty-syncvsr-onnx`,
  `syncvsr_lrs3_visual_ctc_fp16.onnx`) — the most directly Core ML-convertible
  video-only artefact found — but it has **no model card and no declared licence**.
- **Apple has published VSR research but shipped no lip-reading model, API, or Core
  ML conversion.** (Careful: `machinelearning.apple.com/research/lip-articulation`
  is about *perception* of animated lips, not recognition.) Apple's **AV-CPL**
  (ECCV 2024) does target visual-only but released **no code or weights**.
- **WhisperKit and similar Swift speech packages are audio-only** — no visual path.

**So the cleanest legal route to the Neural Engine is to convert Auto-AVSR
(Apache-2.0 code) or BRAVEn (MIT) yourself with `coremltools`**, accepting that the
weights still carry dataset terms. Both recommendations here are the right shape for
that: modest size, no LLM dependency, no AV-HuBERT.

## Residual licence risk — read before committing

Neither recommended repository grants an affirmative licence *to the weights*.
Auto-AVSR disclaims ("may have their own licenses … derived from the dataset used
for training"); RAVEn is silent. The unsettled legal question of whether model
weights inherit training-data restrictions is not one this note can resolve.

**No checkpoint here is commercially clean.** Every one is trained on LRS3, and
LRS3 is TED-derived under **CC BY-NC-ND** — non-commercial *and* no-derivatives,
which a fine-tuned model arguably is. Choosing between checkpoints therefore ranks
*degrees of exposure*; it does not produce a clean option.

That said, the degrees differ in a way worth acting on:

- `vsr_trlrs2lrs3vox2avsp_base.pth` (**20.3%**) adds **LRS2 — BBC data**. The BBC
  terms are the sharpest language in the field, because they restrict the *trained
  model* explicitly, not just the data: content "must not be used for training any
  existing or new technology, algorithms or models that will be sold commercially."
  Highest accuracy, highest exposure.
- `vsr_trlrs3vox2_base.pth` (**24.6%**) is trained on **LRS3 + VoxCeleb2 only** —
  no BBC corpus in the fine-tuning data. Still NC-ND-encumbered via LRS3, but it
  avoids the one licence that names commercial models directly.
- **Caveat, UNVERIFIED:** the paper states its VSR models are initialised from a
  front-end pre-trained on **LRW** (also BBC). The repo does not say per checkpoint
  whether the released `trlrs3vox2` model used that initialisation. So the "no BBC
  data" claim covers the *fine-tuning* corpora only and cannot yet be extended to
  the initialisation. Worth confirming with the author (contact address is in the
  Auto-AVSR README) before the distinction is leaned on.

**And the escape hatch is closed.** The usual answer — "retrain on
permissively-licensed data" — is not currently available: LRS3 is no longer
distributed by anyone official, LRS2/LRW categorically exclude unaffiliated
individuals, and the annotation/URL lists needed to rebuild LRS3 from YouTube are
404 as well. Building a clean-provenance VSR model today would mean collecting a
new corpus, which is a project in its own right.

None of this blocks the immediate work. It does mean **VoxLens should not assume a
commercial path exists on pretrained weights**, and that assumption deserves to be
written down as an ADR rather than discovered later.

## Recommendation

**First choice: `vsr_trlrs3vox2_base.pth` from `mpc001/auto_avsr`.**

- Apache-2.0 code, ungated public download, no fairseq, no academic gate.
- **24.6% video-only WER on the LRS3 test set** (1,759 h training data).
- 250.4 M parameters, ~1 GB fp32 — trivially inside 24 GB.
- Trained on LRS3 + VoxCeleb2 only, so it avoids the BBC-sourced LRS2 corpus whose
  terms restrict commercially-sold models by name.

Start here rather than at the headline 20.3% checkpoint. The 4-point WER gap is not
what determines whether the clip-to-transcript CLI works, and the lower-exposure
weights avoid building the pipeline on the one corpus with explicit
anti-commercial-model language. **Both files are the same architecture and the same
preprocessing**, so switching to `vsr_trlrs2lrs3vox2avsp_base.pth` (20.3%) is a
one-line change — do that deliberately for benchmarking once the pipeline works.

To be clear about what this recommendation does and does not claim: it is the best
choice for **building and evaluating the CLI now**. It is **not** a
commercially-clear checkpoint, and no such checkpoint was found to exist.

**Fallback: BRAVEn Large from `ahaliassos/raven`.**

- **MIT** code — the cleanest code licence of any candidate here.
- **20.1% video-only WER** (BRAVEn Large, self-training + LM, high-resource) — the
  best *downloadable* video-only figure found in this pass, marginally better than
  Auto-AVSR's best.
- **328 M** parameters (Large) ≈ 1.3 GB fp32 — larger than Auto-AVSR but still
  comfortably inside 24 GB.
- Identical Mouth Region preprocessing to Auto-AVSR, so it costs nothing to keep as
  a second option: the same extractor feeds both, byte-for-byte.
- No fairseq dependency — PyTorch Lightning, which is a genuine advantage over
  AV-HuBERT and VatLM.

Held as fallback rather than first choice for three concrete reasons, not
squeamishness: its best number **depends on an external language model** (20.9%
without it); the repo ships **inference code only** ("pre-training and fine-tuning
coming soon", still true), so adapting the model later is not supported; and its
MIT grant is a **personal** copyright with no weights licence attached, which is a
thinner assurance than it first appears. Auto-AVSR's single-file checkpoint and
working tutorial is also simply a shorter path to a running CLI.

**If VoxLens decides it is non-commercial: use USR 2.0 instead.**
CC BY-NC 4.0, ungated, **17.6% video-only** — seven points better than the first
choice — no AV-HuBERT dependency, the same Mouth Region format, and a `demo.py`
with `modality=v` that runs lip-reading out of the box. The only thing keeping it
out of the top slot is that its licence forecloses a commercial path *explicitly*,
where Auto-AVSR's leaves it unresolved. If that path is formally abandoned, this
becomes the obvious pick and the switch is cheap, because the preprocessing is
identical.

**Explicitly rejected:**

- **AV-HuBERT** — Meta licence is **non-commercial research only**, and the repo is
  archived. Disqualified on licence alone.
- **`Visual_Speech_Recognition_for_Multiple_Languages`** (the 19.1% checkpoint) —
  BBC clauses restrict it to **benchmarking and comparison**, explicitly not
  commercial products. Disqualified. Do not copy Chaplin's setup script, which
  pulls exactly this checkpoint.
- **USR** — no licence file, so no rights granted.
- **Whisper-Flamingo** — an audio-visual model that needs audio at inference; not a
  lip reader.

## What the Mouth Region extractor must produce

Restating the deliverable in VoxLens's own vocabulary, for whoever picks up the
extractor work:

> For each Frame of a Clip, produce a **96 × 96 RGB** Mouth Region, at **25 fps**,
> obtained by similarity-warping the Frame onto the shared 68-point mean face using
> the 8 stable landmarks (28, 33, 36, 39, 42, 45, 48, 54) and cropping about the
> centroid of landmarks 48–68. The model consumes the **centre 88 × 88**, converted
> to **greyscale**, scaled to [0, 1] and normalised with **mean 0.421, std 0.165**.
> Landmarks must be smoothed over a 12-Frame window, and the Clip must be resampled
> to 25 fps — nothing downstream does this for you.

If the Mouth Region is persisted to disk between stages, prefer a **lossless** codec.
RAVEn writes FFV1 `.avi`; AV-HuBERT writes MP4 at `-crf 20`, which is lossy and
discards detail in exactly the region that carries the signal.

Two things the reference implementation does *not* settle, and VoxLens must decide:

- **Occlusion.** The reference code linearly interpolates across Frames with no
  detected landmarks and copies the nearest valid set at the ends — silently. VoxLens
  needs this surfaced instead, since a run of interpolated Frames is precisely what
  should mark a Transcript span as Inferred Text rather than Read Text.
- **Detector choice.** RetinaFace + FAN (68 landmarks) matches how the checkpoints
  were trained; MediaPipe (4 keypoints, different crop centre) is the practical
  Apple Silicon path. The accuracy cost of that substitution is unmeasured.

## Open questions

- Whether RAVEn's MIT grant extends to the **weights** — **UNVERIFIED**. The licence
  is a personal copyright with no weights terms attached, and co-authors are
  Meta-affiliated. Worth legal review before commercial reliance.
- Whether the released `vsr_trlrs3vox2_base.pth` was initialised from the
  LRW-pretrained front-end — **UNVERIFIED**, and load-bearing for the
  lower-exposure argument above.
- Parameter counts for **USR 2.0** and **VSRML** — not published by either.
- End-to-end wall-clock inference time for a Clip on the M4 Pro — memory is settled,
  but **speed is not measured**, and the bias-free Conv3d issue makes it
  version-dependent. Worth benchmarking on PyTorch 2.13 vs 2.14 before designing
  around any latency assumption (ADR-0001 puts latency in evaluation from the first
  baseline).
- Measured WER penalty of the MediaPipe 4-keypoint crop versus the RetinaFace/FAN
  68-landmark crop — unmeasured anywhere found here, and directly relevant since
  MediaPipe is the practical Apple Silicon path.
- 2025–2026 VSR work was not exhaustively surveyed. The candidates above are the
  established, downloadable ones; a later sweep for newer permissively-licensed
  checkpoints is worthwhile but should not block the first baseline. Given the
  dataset-provenance findings above, a *newer* checkpoint is unlikely to be
  *cleaner* unless it was trained on a new corpus.

## Sources

- Auto-AVSR repo — https://github.com/mpc001/auto_avsr
- Auto-AVSR paper (arXiv:2303.14307) — https://arxiv.org/abs/2303.14307
- Auto-AVSR preprocessing source — https://github.com/mpc001/auto_avsr/tree/main/preparation
- VSRML repo — https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
- VSRML LICENSE — https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/blob/master/LICENSE
- RAVEn / BRAVEn repo — https://github.com/ahaliassos/raven
- RAVEn paper (arXiv:2212.06246) — https://arxiv.org/abs/2212.06246
- BRAVEn paper (arXiv:2404.02098) — https://arxiv.org/abs/2404.02098
- AV-HuBERT repo (archived) — https://github.com/facebookresearch/av_hubert
- AV-HuBERT LICENSE — https://github.com/facebookresearch/av_hubert/blob/main/LICENSE
- USR repo — https://github.com/ahaliassos/usr
- Whisper-Flamingo repo — https://github.com/roudimit/whisper-flamingo
- Chaplin (reference Mac implementation) — https://github.com/amanvirparhar/chaplin
- BBC R&D lip-reading datasets (LRS2/LRW terms) — https://www.bbc.co.uk/rd/projects/lip-reading-datasets
- TED Talks usage policy (CC BY-NC-ND upstream of LRS3) — https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy
- VGG lip-reading dataset index (LRS3 link now 404s) — https://www.robots.ox.ac.uk/~vgg/data/lip_reading/
- mmai.io lip-reading mirror ("Downloads are no longer available") — https://mmai.io/datasets/lip_reading/
- USR 2.0 — https://github.com/ahaliassos/usr2
- PyTorch MPS bias-free Conv3d performance issue — https://github.com/pytorch/pytorch/issues/192213
- PyTorch MPS Conv3d fix — https://github.com/pytorch/pytorch/issues/192229
- PyTorch MPS environment variables — https://docs.pytorch.org/docs/2.13/mps_environment_variables.html
