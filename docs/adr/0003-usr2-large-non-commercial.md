# USR 2.0 Large, and VoxLens as non-commercial research

**Status:** accepted — decided and implemented.

VoxLens builds on the **USR 2.0 Large** checkpoint (CC BY-NC 4.0, ungated), and is positioned as **non-commercial research**. The two decisions are inseparable: no cleanly-licensed visual speech recognition checkpoint exists for commercial use, because every candidate trains on LRS3, which is TED-derived under CC BY-NC-ND — and LRS3 is no longer distributed, so retraining clean is not an escape hatch either.

Large rather than Huge: Huge scores better (17.6% vs 21.5% published) but its parameter count is unpublished, and its training data includes LRS2, whose terms restrict commercially-sold models by name. Against an on-device target, defaulting to an unmeasured larger model was the wrong risk.

## Consequences

- **"Research" is not "academic".** Corpora that gate on institutional affiliation — LRS2, LRW, CAS-VSR, CMLR, TCD-TIMIT, MEAD — stay closed to VoxLens regardless of its non-commercial status. This surprises people and is the single most common misreading of the decision.
- A commercial path would require starting over on data provenance, not swapping a checkpoint.
- AV-HuBERT stays rejected on a separate ground: its licence bars "surveillance" and "biometric processing" outright, which is not a commercial clause and is squarely on point for a lip-reader.

Detail: VoxLens issues #3 and #10, and `docs/research/pretrained-checkpoints.md`.
