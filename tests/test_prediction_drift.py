"""Tests for deterministic fraud score population drift."""

from math import isfinite

import pytest

from fraud_scoring.monitoring import (
    score_population_stability_index,
)


def test_identical_distributions_have_zero_psi() -> None:
    scores = [0.05, 0.15, 0.25, 0.45, 0.75, 0.95]

    result = score_population_stability_index(scores, scores)

    assert result == pytest.approx(0.0)


def test_shifted_scores_produce_positive_psi() -> None:
    reference = [0.05, 0.15, 0.25, 0.35]
    observed = [0.65, 0.75, 0.85, 0.95]

    result = score_population_stability_index(
        reference,
        observed,
    )

    assert result > 0.0
    assert isfinite(result)


def test_proportional_populations_are_stable() -> None:
    reference = [0.1, 0.9]
    observed = [0.1] * 100 + [0.9] * 100

    result = score_population_stability_index(
        reference,
        observed,
        bins=2,
    )

    assert result == pytest.approx(0.0, abs=1e-12)


def test_score_endpoints_are_included() -> None:
    result = score_population_stability_index(
        [0.0, 1.0],
        [1.0, 0.0],
        bins=2,
    )

    assert result == pytest.approx(0.0)


def test_disjoint_populations_produce_finite_psi() -> None:
    result = score_population_stability_index(
        [0.05] * 10,
        [0.95] * 10,
    )

    assert isfinite(result)
    assert result > 0.0


def test_psi_is_symmetric() -> None:
    reference = [0.1, 0.2, 0.3]
    observed = [0.7, 0.8, 0.9]

    forward = score_population_stability_index(
        reference,
        observed,
    )
    backward = score_population_stability_index(
        observed,
        reference,
    )

    assert forward == pytest.approx(backward)


def test_sample_order_does_not_change_psi() -> None:
    reference = [0.0, 0.3, 0.7, 1.0]
    observed = [0.0, 0.0, 1.0, 1.0]

    original = score_population_stability_index(
        reference,
        observed,
    )
    reordered = score_population_stability_index(
        list(reversed(reference)),
        list(reversed(observed)),
    )

    assert original == pytest.approx(reordered)


@pytest.mark.parametrize(
    ("reference", "observed"),
    [
        ([], [0.5]),
        ([0.5], []),
    ],
)
def test_empty_populations_are_rejected(
    reference: list[float],
    observed: list[float],
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        score_population_stability_index(
            reference,
            observed,
        )


@pytest.mark.parametrize("side", ["reference", "observed"])
@pytest.mark.parametrize(
    "bad_score",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
        True,
        "0.5",
    ],
)
def test_invalid_scores_are_rejected(
    side: str,
    bad_score: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="must contain finite scores",
    ):
        if side == "reference":
            score_population_stability_index(
                [bad_score],
                [0.3],
            )
        else:
            score_population_stability_index(
                [0.3],
                [bad_score],
            )


@pytest.mark.parametrize(
    "invalid_bins",
    [0, 1, True, 2.5],
)
def test_invalid_bin_count_is_rejected(
    invalid_bins: object,
) -> None:
    with pytest.raises(ValueError, match="bins"):
        score_population_stability_index(
            [0.1],
            [0.9],
            bins=invalid_bins,
        )


@pytest.mark.parametrize(
    "invalid_smoothing",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        True,
    ],
)
def test_invalid_smoothing_is_rejected(
    invalid_smoothing: object,
) -> None:
    with pytest.raises(ValueError, match="smoothing"):
        score_population_stability_index(
            [0.1],
            [0.9],
            smoothing=invalid_smoothing,
        )
