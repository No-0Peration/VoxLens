"""Tests for the divergence arithmetic (#22).

A pure seam, for the reason ADR-0007 gives about Occlusion spans: the edge
cases live in the arithmetic — ties, zero variance, one side empty, a Clip read
300% wrong — and provoking them through a 4 GB model would be absurd.

What these do NOT test is whether divergence predicts being wrong. That is a
measurement over a corpus, not a property of the code, and ADR-0011 requires
it before the number may be called confidence.
"""
from __future__ import annotations

import pytest

from voxlens.confidence import buckets, correlate, divergence, pearson, spearman


# --- divergence -----------------------------------------------------------

def test_decoders_that_agree_word_for_word_diverge_by_zero():
    assert divergence("well guess what", "well guess what") == 0.0


def test_decoders_with_nothing_in_common_diverge_by_one():
    assert divergence("well guess what", "the other thing entirely") == 1.0


def test_divergence_is_symmetric():
    """Neither decoder is the reference — unlike WER, where one side is truth."""
    left, right = "this is the only part", "this is the wrong part again"
    assert divergence(left, right) == divergence(right, left)


def test_divergence_scales_by_the_longer_reading():
    """One word in four, not one word in three: the denominator is the longer
    side, so a Clip cannot look more divergent for being read at length."""
    assert divergence("the cat sat", "the cat sat down") == pytest.approx(0.25)


def test_divergence_never_exceeds_one():
    assert divergence("a", "w x y z") == 1.0


def test_divergence_ignores_case_and_spacing():
    """The two decoders render text through the same normalisation, so any
    difference that survives here is a difference in what was read."""
    assert divergence("Well  guess WHAT", "well guess what") == 0.0


def test_two_decoders_reading_nothing_are_not_treated_as_disagreeing():
    """A Clip with no Transcript is reported as unreadable elsewhere; it must
    not also arrive here as maximum doubt."""
    assert divergence("", "") == 0.0
    assert divergence("something", "") == 1.0


# --- correlation ----------------------------------------------------------

def test_pearson_spans_perfect_agreement_to_perfect_inversion():
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)


def test_a_series_that_does_not_vary_has_no_correlation():
    """None, not zero: 'no relationship' and 'not answerable' are different
    findings, and a decision would be made differently on each."""
    assert pearson([1, 1, 1], [1, 2, 3]) is None
    assert spearman([2, 2, 2], [1, 2, 3]) is None


def test_a_single_clip_is_not_a_correlation():
    assert pearson([0.5], [0.5]) is None
    assert correlate([0.5], [0.5])["spearman"] is None


def test_spearman_sees_a_monotone_relationship_pearson_understates():
    """Which is why it is the one to read first: per-Clip WER is unbounded
    above, so the relationship need not be linear to be real."""
    divergences = [0.1, 0.2, 0.3, 0.4]
    wers = [0.01, 0.04, 0.09, 3.0]
    assert spearman(divergences, wers) == pytest.approx(1.0)
    assert pearson(divergences, wers) < 0.95


def test_ties_are_ranked_by_their_average():
    assert spearman([1, 1, 2], [5, 5, 9]) == pytest.approx(1.0)


def test_mismatched_series_are_refused_rather_than_zipped_short():
    with pytest.raises(ValueError):
        pearson([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


# --- buckets --------------------------------------------------------------

def test_buckets_report_wer_from_most_agreement_to_least():
    """The question a decision rests on: if the decoders agree, how much
    better is the text, in points of WER?"""
    divergences = [0.0, 0.1, 0.4, 0.5, 0.9, 1.0]
    wers = [0.0, 0.1, 0.3, 0.4, 0.9, 1.1]
    report = buckets(divergences, wers, groups=3)

    assert [band["clips"] for band in report] == [2, 2, 2]
    assert report[0]["mean_wer_pct"] == 5.0
    assert report[-1]["mean_wer_pct"] == 100.0
    assert report[0]["divergence_to"] <= report[-1]["divergence_from"]


def test_buckets_survive_a_corpus_too_small_to_split():
    assert buckets([], []) == []
    assert len(buckets([0.5], [0.5], groups=3)) == 1


def test_correlate_reports_what_a_decision_needs_and_claims_nothing():
    report = correlate([0.0, 0.2, 0.5, 0.9], [0.0, 0.1, 0.4, 1.2])

    assert report["clips"] == 4
    assert report["spearman"] == pytest.approx(1.0)
    assert report["mean_divergence"] == pytest.approx(0.4)
    assert report["buckets"]
    assert "confidence" not in report, "this module measures; it does not conclude"


# --- the second reading ---------------------------------------------------
# The CTC collapse rule is the one part of #22 that touches the model, and it
# is pure arithmetic over token ids, so it is tested here rather than through
# a checkpoint.

from voxlens.recogniser import collapse_ctc  # noqa: E402
from voxlens.upstream import is_vendored  # noqa: E402

needs_upstream = pytest.mark.skipif(
    not is_vendored(), reason="upstream not vendored — run: uv run python scripts/vendor.py"
)


def test_a_token_held_across_frames_is_read_once():
    assert collapse_ctc([5, 5, 5, 5]) == [5]


def test_blanks_are_dropped():
    assert collapse_ctc([0, 0, 7, 0, 0]) == [7]


def test_a_blank_between_repeats_marks_two_tokens_not_one():
    """The rule that makes CTC work, and the one an ordinary dedupe gets
    wrong: without it, a doubled word silently becomes a single one."""
    assert collapse_ctc([5, 5, 0, 5, 5]) == [5, 5]


def test_a_reading_of_pure_blanks_is_empty_rather_than_invented():
    assert collapse_ctc([0, 0, 0]) == []
    assert collapse_ctc([]) == []


@needs_upstream
def test_the_blank_index_is_where_this_vocabulary_puts_it():
    """collapse_ctc drops index 0. If the vocabulary ever moved the blank, the
    CTC reading would come out as noise rather than failing."""
    from voxlens.recogniser import CTC_BLANK
    from voxlens.upstream import ensure_importable

    ensure_importable()
    from utils.utils import UNIGRAM1000_LIST

    assert UNIGRAM1000_LIST[CTC_BLANK] == "<blank>"


# --- the measured bands (#22) ---------------------------------------------
# Thresholds are tercile boundaries from 571 WildVSR Clips, not taste. These
# tests pin the mapping, not the choice of boundary — moving a boundary means
# re-running the measurement and amending ADR-0011.

def test_divergence_maps_to_the_band_it_was_measured_into():
    from voxlens.confidence import DOUBTFUL_ABOVE, FIRM_BELOW, band

    assert band(0.0) == "firm"
    assert band(FIRM_BELOW - 0.001) == "firm"
    assert band(FIRM_BELOW) == "uncertain"
    assert band(DOUBTFUL_ABOVE - 0.001) == "uncertain"
    assert band(DOUBTFUL_ABOVE) == "doubtful"
    assert band(1.0) == "doubtful"


def test_every_band_carries_what_clips_in_it_actually_scored():
    """A reader told "uncertain" and nothing else has been given a mood."""
    from voxlens.confidence import BAND_WER_PCT

    assert set(BAND_WER_PCT) == {"firm", "uncertain", "doubtful"}
    assert BAND_WER_PCT["firm"] < BAND_WER_PCT["uncertain"] < BAND_WER_PCT["doubtful"]
    # Firm is not right: better than one word in four is still wrong.
    assert BAND_WER_PCT["firm"] > 20
