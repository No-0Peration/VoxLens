# Two evaluation bars, and WildVSR as the authoritative one

**Status:** accepted — decided and implemented.

VoxLens measures WER against **two** corpora with different jobs. LRS3's test split is the **calibration** bar — it is what published figures are quoted against, so its purpose is to confirm preprocessing is correct. WildVSR is the **honest** bar, and the authoritative one for judging whether the approach works.

A single LRS3 number would be self-flattering. The same model scores 17.6% on LRS3 and 73.7% on WildVSR in published results; measured here at beam 1, 34.22% and 47.85%. Reporting only the first would describe benchmark fit rather than capability.

## Consequences

- The calibration bar has **already done its job**: 24.3% measured at beam 40 against a published 21.5% confirmed the mouth-crop convention is correct. Beyond that check, LRS3 is not evidence about VoxLens.
- The recorded bars are **regression bars**. A **viability** bar — whether ~48% WER clears a product threshold — is deliberately unset, because that is a judgement about what VoxLens is for, not a measurement.
- **These numbers are not claims of parity with published work.** This harness measures 47.85% on WildVSR where 73.7% is published — better, using a smaller configuration, where it should be worse. Most likely a text-normalisation difference. Until reconciled, the bars are valid against themselves and nothing else.
- Both corpora ship **pre-cropped mouth regions**, so they exercise the recogniser while bypassing face detection, extraction and Occlusion entirely. Those stages are unmeasured.

Detail: VoxLens issues #2 and #5.
