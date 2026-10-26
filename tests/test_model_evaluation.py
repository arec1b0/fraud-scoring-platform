"""Tests for synthetic evaluation metrics and promotion policies."""

from copy import deepcopy

import numpy as np
import pytest

from fraud_scoring.scoring.evaluation import (
    compute_metrics,
    enforce_quality_policy,
    evaluation_dataset,
)


def good_report() -> dict[str, object]:
    return {
        "dataset_sha256": "a" * 64,
        "classification_threshold": 0.5,
        "metrics": {
            "roc_auc": 0.80,
            "pr_auc": 0.50,
            "precision": 0.70,
            "recall": 0.50,
            "f1": 0.58,
            "brier_score": 0.10,
        },
    }


def test_perfect_ranking_metrics() -> None:
    metrics = compute_metrics(
        np.array([1, 0, 1, 0]),
        np.array([0.9, 0.1, 0.8, 0.2]),
    )

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(0.025)


def test_all_tied_scores() -> None:
    metrics = compute_metrics(
        np.array([1, 0, 1, 0]),
        np.full(4, 0.5),
    )

    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["pr_auc"] == pytest.approx(0.5)


def test_no_positive_predictions() -> None:
    metrics = compute_metrics(
        np.array([1, 0, 1, 0]),
        np.array([0.3, 0.1, 0.2, 0.1]),
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([1, 1], [0.8, 0.9]),
        ([0, 0], [0.1, 0.2]),
        ([0, 1], [0.2]),
        ([0, 2], [0.1, 0.9]),
        ([0, 1], [float("nan"), 0.7]),
        ([0, 1], [0.1, 1.2]),
    ],
)
def test_invalid_metric_inputs(
    labels: list[float],
    scores: list[float],
) -> None:
    with pytest.raises(ValueError):
        compute_metrics(
            np.asarray(labels),
            np.asarray(scores),
        )


def test_evaluation_dataset_is_deterministic() -> None:
    first_x, first_y, first_hash = evaluation_dataset()
    second_x, second_y, second_hash = evaluation_dataset()

    assert first_hash == second_hash
    assert first_x.shape == (1024, 2)
    assert first_y.shape == (1024,)
    assert np.array_equal(first_x, second_x)
    assert np.array_equal(first_y, second_y)
    assert set(first_y.tolist()) == {0, 1}


def test_valid_quality_report_passes() -> None:
    enforce_quality_policy(good_report())


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("roc_auc", 0.59),
        ("pr_auc", 0.11),
        ("precision", 0.09),
        ("recall", 0.09),
        ("f1", 0.09),
        ("brier_score", 0.31),
    ],
)
def test_absolute_quality_gate_rejects_failure(
    metric: str,
    value: float,
) -> None:
    report = good_report()
    report["metrics"][metric] = value

    with pytest.raises(ValueError, match="Quality gate"):
        enforce_quality_policy(report)


def test_baseline_equal_quality_passes() -> None:
    baseline = good_report()
    candidate = deepcopy(baseline)

    enforce_quality_policy(
        candidate,
        baseline=baseline,
    )


@pytest.mark.parametrize(
    ("metric", "candidate_value"),
    [
        ("roc_auc", 0.77),
        ("pr_auc", 0.47),
        ("brier_score", 0.13),
    ],
)
def test_baseline_regression_gate(
    metric: str,
    candidate_value: float,
) -> None:
    baseline = good_report()
    candidate = deepcopy(baseline)
    candidate["metrics"][metric] = candidate_value

    with pytest.raises(
        ValueError,
        match="Baseline regression gate",
    ):
        enforce_quality_policy(
            candidate,
            baseline=baseline,
        )


def test_baseline_dataset_mismatch_rejected() -> None:
    baseline = good_report()
    candidate = deepcopy(baseline)
    candidate["dataset_sha256"] = "b" * 64

    with pytest.raises(
        ValueError,
        match="incompatible evaluation dataset",
    ):
        enforce_quality_policy(
            candidate,
            baseline=baseline,
        )
