"""Scoring, and the normalisation it depends on.

WER is meaningless without a stated normalisation, so it is tested as
deliberately as the edit distance itself.
"""
import json

import pytest

from voxlens.evaluate import load_corpus, normalise, wer


def rate(reference: str, hypothesis: str) -> float:
    errors, words = wer(reference, hypothesis)
    return 100 * errors / words


def test_an_exact_match_scores_zero():
    assert rate("the cat sat", "the cat sat") == 0


def test_a_substitution_costs_one_word():
    assert rate("the cat sat", "the bat sat") == pytest.approx(100 / 3)


def test_a_deletion_costs_one_word():
    assert rate("the cat sat", "the sat") == pytest.approx(100 / 3)


def test_an_insertion_costs_one_word():
    assert rate("the cat sat", "the cat sat down") == pytest.approx(100 / 3)


def test_a_completely_wrong_hypothesis_scores_one_hundred():
    assert rate("alpha beta", "gamma delta") == 100


def test_an_empty_hypothesis_is_all_deletions():
    assert rate("alpha beta", "") == 100


def test_wer_can_exceed_one_hundred_percent():
    """Insertions are unbounded — a bar set at 100% would be unreachable, not safe."""
    assert rate("hi", "hi there and also this") > 100


def test_case_is_normalised_away():
    """WildVSR ships uppercase references, LRS3 lowercase. Both must score alike."""
    assert rate("THE CAT SAT", "the cat sat") == 0


def test_surrounding_and_repeated_whitespace_is_normalised_away():
    assert rate("  the   cat  sat ", "the cat sat") == 0


def test_apostrophes_are_kept_as_part_of_the_word():
    """'don't' is one word; splitting it would silently inflate every score."""
    assert normalise("DON'T") == "don't"
    assert rate("don't", "do not") == 200  # 1 sub + 1 insertion against a 1-word ref


def test_an_empty_reference_counts_hypothesis_words_as_errors():
    errors, words = wer("", "spurious words here")
    assert (errors, words) == (3, 0)


def test_an_unknown_corpus_is_rejected_by_name(tmp_path):
    with pytest.raises(ValueError, match="wibble"):
        load_corpus(tmp_path, "wibble")


def test_wildvsr_manifest_is_read_and_strided(tmp_path):
    base = tmp_path / "wildvsr" / "WildVSR"
    (base / "videos").mkdir(parents=True)
    labels = {}
    for index in range(10):
        name = f"{index:04d}.mp4"
        (base / "videos" / name).write_bytes(b"")
        labels[name] = f"REFERENCE {index}"
    (base / "labels.json").write_text(json.dumps(labels))

    every = load_corpus(tmp_path, "wildvsr")
    assert len(every) == 10
    assert every[0][1] == "REFERENCE 0"

    strided = load_corpus(tmp_path, "wildvsr", stride=5)
    assert [path.name for path, _ in strided] == ["0000.mp4", "0005.mp4"]


def test_lrs3_manifest_skips_the_root_header_row(tmp_path):
    """The fairseq manifest's first line is a root path, not a Clip. Off-by-one
    here would silently misalign every reference in the corpus."""
    en = tmp_path / "muavic" / "en"
    (en / "video" / "test" / "SPEAKER").mkdir(parents=True)
    for index in (1, 2):
        (en / "video" / "test" / "SPEAKER" / f"{index:05d}.mp4").write_bytes(b"")
    (en / "test.tsv").write_text("/\nSPEAKER/00001\tpath\nSPEAKER/00002\tpath\n")
    (en / "test.wrd").write_text("first reference\nsecond reference\n")

    pairs = load_corpus(tmp_path, "lrs3")
    assert [path.name for path, _ in pairs] == ["00001.mp4", "00002.mp4"]
    assert [ref for _, ref in pairs] == ["first reference", "second reference"]
