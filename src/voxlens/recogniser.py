"""Mouth Regions to a Transcript.

The encoder and the search run on different devices by default (ADR-0002), so
they are kept separate here rather than hidden behind one ``transcribe`` call
on the whole model. The encoder is also the streaming-critical component and
the natural Core ML conversion unit, so it must stay separable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from voxlens.devices import DevicePlan
from voxlens.upstream import BACKBONE, ensure_importable, load_config, vendored_path

__all__ = ["CheckpointError", "Recogniser", "collapse_ctc", "load_recogniser"]


class CheckpointError(RuntimeError):
    """The checkpoint is missing, unreadable, or not this architecture.

    Named so the CLI can map it to an exit code without swallowing genuine
    programming errors, which must still surface as tracebacks.
    """

# CTC reserves index 0 for the blank label, which is what makes a greedy
# reading possible: collapse repeats, then drop blanks.
CTC_BLANK = 0

# The crop contract the checkpoint was trained against. Changing any of these
# silently degrades accuracy rather than failing, so they are stated once.
CROP = 88
NORM_MEAN = 0.421
NORM_STD = 0.165


def collapse_ctc(ids) -> list[int]:
    """Frame-wise argmax ids to a token sequence, by the CTC rule.

    Collapse runs of the same id, then drop blanks — in that order, because a
    blank between two identical ids is what marks them as two genuine tokens
    rather than one held across Frames. Reversing it would silently merge
    doubled words.

    Pure, and a seam of its own for the reason ADR-0007 gives: the rule has
    exactly the edge cases that are painful to provoke through a 4 GB model.
    """
    collapsed = []
    previous = None
    for token_id in ids:
        if token_id != previous and token_id != CTC_BLANK:
            collapsed.append(token_id)
        previous = token_id
    return collapsed


def _text_from_tokens(ids, tokens: list) -> str:
    """Token ids to a Transcript, normalised as the beam path normalises.

    Both decoders must render text the same way or their disagreement would
    partly be a difference in punctuation and case rather than in what was
    read.
    """
    text = "".join(tokens[i] for i in ids)
    return text.replace("<eos>", "").replace("\u2581", " ").strip().lower()


def _video_transform():
    from torchvision.transforms import CenterCrop, Compose, Grayscale, Lambda

    ensure_importable()
    from data.transforms import NormalizeVideo

    return Compose([
        Lambda(lambda x: x / 255.0),
        CenterCrop(CROP),
        Lambda(lambda x: x.transpose(0, 1)),
        Grayscale(),
        Lambda(lambda x: x.transpose(0, 1)),
        NormalizeVideo(mean=(NORM_MEAN,), std=(NORM_STD,)),
        Lambda(lambda x: x.squeeze(0)),
    ])


@dataclass
class Recogniser:
    """A loaded recogniser, placed across devices per a DevicePlan."""

    model: object
    beam_search: object
    plan: DevicePlan
    tokens: list

    def encode(self, crops: np.ndarray) -> torch.Tensor:
        """Mouth Region crops -> encoder output, one step per Frame."""
        video = torch.from_numpy(crops).permute(3, 0, 1, 2).float()
        features = _video_transform()(video).to(self.plan.encoder)
        with torch.no_grad():
            encoded = self.model.encoder(xs_v=features.unsqueeze(0))
        if self.plan.needs_transfer:
            encoded = encoded.to(self.plan.decoder)
        return encoded

    def decode(self, encoded: torch.Tensor) -> str:
        """Encoder output -> Transcript."""
        ensure_importable()
        from espnet.asr.asr_utils import parse_hypothesis

        with torch.no_grad():
            hypotheses = self.beam_search(
                x=encoded.squeeze(0),
                modality="v",
                maxlenratio=self.maxlenratio,
                minlenratio=self.minlenratio,
            )
        text, _, _, _ = parse_hypothesis(hypotheses[0].asdict(), self.tokens)
        return text.replace("<eos>", "").replace("▁", " ").strip().lower()

    def decode_ctc(self, encoded: torch.Tensor) -> str:
        """Encoder output -> the CTC head's own Transcript, read greedily.

        Deliberately given the *same* encoder output the beam search reads
        (ADR-0011). Two decoders disagreeing about one encoding says something
        about that encoding; two models disagreeing would say nothing.

        The beam search already scores CTC alongside the decoder, so this is
        not a second opinion from a fresh source — it is the CTC head's
        unassisted reading, which is why it can differ from the joint result.
        """
        with torch.no_grad():
            logprobs = self.model.ctc_v.log_softmax(encoded)
        ids = logprobs.argmax(dim=-1).squeeze(0).tolist()
        return _text_from_tokens(collapse_ctc(ids), self.tokens)

    def transcribe(self, crops: np.ndarray) -> str:
        return self.decode(self.encode(crops))


def load_recogniser(checkpoint_path, plan: DevicePlan, beam: int = 1) -> Recogniser:
    """Load the recogniser and place it across devices per ``plan``."""
    from pathlib import Path

    ensure_importable()
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
    from espnet.nets.scorers.length_bonus import LengthBonus
    from utils.utils import UNIGRAM1000_LIST

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise CheckpointError(
            f"Checkpoint not found: {checkpoint_path}. VoxLens never downloads it — "
            "see docs/setup.md for where to obtain USR 2.0 Large."
        )

    cfg = load_config()
    cfg.decode.beam_size = beam

    model = E2E(len(UNIGRAM1000_LIST), cfg.model.backbone)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # unpickling, truncation, wrong format
        raise CheckpointError(
            f"Could not read the checkpoint at {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(state, dict):
        raise CheckpointError(
            f"{checkpoint_path} does not contain a state dict "
            f"(found {type(state).__name__}). Is this a USR 2.0 checkpoint?"
        )
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    if any(k.startswith("model.backbone.") for k in state):
        state = {
            k.replace("model.backbone.", "", 1): v
            for k, v in state.items()
            if k.startswith("model.backbone.")
        }
    try:
        model.load_state_dict(state)
    except (RuntimeError, KeyError) as exc:
        raise CheckpointError(
            f"{checkpoint_path} is not a {BACKBONE} checkpoint: {exc}"
        ) from exc
    model.eval()

    # The encoder goes where the plan says; everything else follows the decoder,
    # which is what makes the hybrid split possible.
    model.to(plan.decoder)
    model.encoder.to(plan.encoder)
    for param in model.parameters():
        param.requires_grad_(False)

    scorers = model.scorers()
    scorers["length_bonus"] = LengthBonus(len(UNIGRAM1000_LIST))
    beam_search = BatchBeamSearch(
        beam_size=cfg.decode.beam_size,
        vocab_size=len(UNIGRAM1000_LIST),
        weights=dict(
            decoder=1.0 - cfg.decode.ctc_weight,
            ctc=cfg.decode.ctc_weight,
            length_bonus=cfg.decode.penalty,
        ),
        scorers=scorers,
        sos=len(UNIGRAM1000_LIST) - 1,
        eos=len(UNIGRAM1000_LIST) - 1,
        token_list=UNIGRAM1000_LIST,
        pre_beam_score_key=None if cfg.decode.ctc_weight == 1.0 else "decoder",
    )
    beam_search.to(device=plan.decoder, dtype=torch.float32).eval()

    recogniser = Recogniser(
        model=model, beam_search=beam_search, plan=plan, tokens=UNIGRAM1000_LIST
    )
    recogniser.maxlenratio = cfg.decode.maxlenratio
    recogniser.minlenratio = cfg.decode.minlenratio
    return recogniser
