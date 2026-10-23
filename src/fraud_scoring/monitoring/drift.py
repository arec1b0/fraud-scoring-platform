"""Deterministic prediction distribution drift detection."""

from collections.abc import Sequence
from math import fsum, isfinite, log


def _histogram(
    scores: Sequence[float],
    *,
    bins: int,
    name: str,
) -> list[int]:
    """Count scores in fixed-width bins over the interval [0, 1]."""
    if len(scores) == 0:
        raise ValueError(f"{name} must not be empty")

    counts = [0] * bins

    for score in scores:
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise ValueError(
                f"{name} must contain finite scores in [0, 1]"
            )

        # The final bin includes the score 1.0.
        index = min(int(score * bins), bins - 1)
        counts[index] += 1

    return counts


def score_population_stability_index(
    reference_scores: Sequence[float],
    observed_scores: Sequence[float],
    *,
    bins: int = 10,
    smoothing: float = 1e-6,
) -> float:
    """Calculate PSI between reference and observed score populations.

    The histogram boundaries are fixed over [0, 1], with equal-width
    bins. Both populations use identical boundaries.

    Additive pseudocount smoothing prevents logarithms of zero.

    No threshold, alert, or automatic remediation is applied.
    """
    if type(bins) is not int or bins < 2:
        raise ValueError("bins must be an integer >= 2")

    if (
        isinstance(smoothing, bool)
        or not isinstance(smoothing, (int, float))
        or not isfinite(smoothing)
        or smoothing <= 0
        or not isfinite(smoothing * bins)
    ):
        raise ValueError(
            "smoothing must be a positive finite number"
        )

    reference_counts = _histogram(
        reference_scores,
        bins=bins,
        name="reference_scores",
    )

    observed_counts = _histogram(
        observed_scores,
        bins=bins,
        name="observed_scores",
    )

    reference_total = (
        sum(reference_counts) + bins * smoothing
    )
    observed_total = (
        sum(observed_counts) + bins * smoothing
    )

    contributions: list[float] = []

    for reference_count, observed_count in zip(
        reference_counts,
        observed_counts,
        strict=True,
    ):
        reference_proportion = (
            reference_count + smoothing
        ) / reference_total

        observed_proportion = (
            observed_count + smoothing
        ) / observed_total

        contribution = (
            observed_proportion - reference_proportion
        ) * log(observed_proportion / reference_proportion)

        contributions.append(contribution)

    # PSI is mathematically non-negative. Clamp tiny rounding errors.
    return max(0.0, fsum(contributions))
