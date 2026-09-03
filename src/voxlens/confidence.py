"""How much the two decoders disagree, and whether that predicts being wrong.

The CTC head and the beam search read the same encoder output and routinely
produce different Transcripts of the same Clip. ADR-0008 records that as an
obstacle — it is why per-word Occlusion marking was abandoned. ADR-0011
proposes reusing it as a signal: two decoders agreeing is evidence, two
decoders disagreeing is doubt.

**Divergence was measured before it was believed.** ADR-0011 required a
correlation against per-Clip WER before this could be called confidence, and
that correlation was run: Spearman **0.755** over 571 WildVSR Clips, with mean
WER of 27.6% where the decoders agree most and 76.5% where they agree least.
It is not clip length in disguise — divergence against reference length
correlates at -0.05, and within Clips of eight words or more the relationship
is unchanged at 0.756. The bands below come from that measurement, and nothing
here is a threshold anybody guessed.

Pure, so it is a seam of its own for the reason ADR-0007 gives about Occlusion
spans: the edge cases are in the arithmetic, and provoking them through a 4 GB
model would be absurd.
"""
from __future__ import annotations

from voxlens.evaluate import normalise, wer

__all__ = ["band", "buckets", "correlate", "divergence", "pearson", "spearman"]

# Where the bands fall, measured rather than chosen (#22): the tercile
# boundaries of divergence over 571 WildVSR Clips at beam 1.
#
#     divergence          Clips   mean WER
#     0.00 - 0.25           190      27.6%
#     0.25 - 0.47           191      46.6%
#     0.47 - 1.00           190      76.5%
#
# A spread of forty-nine points is what makes a label worth showing a reader.
# A signal that moved WER from 30% to 34% would not have earned one.
FIRM_BELOW = 0.25
DOUBTFUL_ABOVE = 0.47


def band(value: float) -> str:
    """Which measured band a divergence falls in.

    "firm" is not "right": Clips in that band still average 27.6% word errors,
    and roughly one in thirty comes back word-perfect. It means the two
    decoders agreed, which is the best evidence available short of knowing.
    """
    if value < FIRM_BELOW:
        return "firm"
    if value < DOUBTFUL_ABOVE:
        return "uncertain"
    return "doubtful"


# Mean WER measured in each band, so a reader is told what the label costs
# rather than being asked to trust the word.
BAND_WER_PCT = {"firm": 27.6, "uncertain": 46.6, "doubtful": 76.5}


def divergence(beam: str, ctc: str) -> float:
    """How far apart two readings of one Clip are, from 0.0 to 1.0.

    Word-level edit distance, divided by the longer of the two readings. That
    denominator makes it **symmetric** — neither decoder is the reference here,
    unlike in WER, where one side is ground truth — and keeps it comparable
    across Clips of different lengths.

    0.0 is word-for-word agreement. 1.0 is no shared reading at all.
    """
    left, right = normalise(beam).split(), normalise(ctc).split()
    if not left and not right:
        # Both read nothing. They agree, and saying so is more useful than a
        # division by zero: a Clip with no Transcript is reported elsewhere.
        return 0.0
    distance, _ = wer(" ".join(left), " ".join(right))
    return min(1.0, distance / max(len(left), len(right)))


def pearson(xs, ys) -> float | None:
    """Linear correlation, or None when either series does not vary."""
    xs, ys = list(xs), list(ys)
    if len(xs) != len(ys):
        raise ValueError("pearson needs two series of the same length")
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    numerator = sum(a * b for a, b in zip(dx, dy))
    denominator = (sum(a * a for a in dx) * sum(b * b for b in dy)) ** 0.5
    if denominator == 0:
        return None
    return numerator / denominator


def _ranks(values) -> list[float]:
    """Ranks with ties averaged, which is what makes Spearman well defined."""
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position
        while end + 1 < len(ordered) and values[ordered[end + 1]] == values[ordered[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for index in ordered[position : end + 1]:
            ranks[index] = average
        position = end + 1
    return ranks


def spearman(xs, ys) -> float | None:
    """Rank correlation.

    The one to read first. Divergence is a bounded ratio and per-Clip WER is
    unbounded above — a Clip can be 300% wrong — so a monotone relationship
    matters more here than a linear one.
    """
    xs, ys = list(xs), list(ys)
    if len(xs) != len(ys):
        raise ValueError("spearman needs two series of the same length")
    if len(xs) < 2:
        return None
    return pearson(_ranks(xs), _ranks(ys))


def buckets(divergences, wers, groups: int = 3) -> list[dict]:
    """Mean WER per divergence band, ordered from most agreement to least.

    A correlation coefficient answers "is there a relationship". This answers
    the question a decision actually rests on: if the decoders agree, how much
    better is the text, in points of WER? A coefficient of 0.3 that moves WER
    from 30% to 34% is not worth a user-facing number.
    """
    pairs = sorted(zip(divergences, wers))
    if not pairs or groups < 1:
        return []

    size = len(pairs) / groups
    report = []
    for group in range(groups):
        start, end = round(group * size), round((group + 1) * size)
        chunk = pairs[start:end]
        if not chunk:
            continue
        report.append(
            {
                "clips": len(chunk),
                "divergence_from": round(chunk[0][0], 3),
                "divergence_to": round(chunk[-1][0], 3),
                "mean_wer_pct": round(100 * sum(w for _, w in chunk) / len(chunk), 2),
            }
        )
    return report


def correlate(divergences, wers) -> dict:
    """The evidence ADR-0011 asks for, in one dictionary.

    Reported as measurement, with no verdict attached: adopting divergence as
    a user-facing confidence, or falling back to the beam hypothesis score, is
    a decision to record in the ADR against these numbers.
    """
    divergences, wers = list(divergences), list(wers)
    return {
        "clips": len(divergences),
        "spearman": _rounded(spearman(divergences, wers)),
        "pearson": _rounded(pearson(divergences, wers)),
        "mean_divergence": _rounded(
            sum(divergences) / len(divergences) if divergences else None
        ),
        "buckets": buckets(divergences, wers),
    }


def _rounded(value: float | None, places: int = 3) -> float | None:
    return None if value is None else round(value, places)
