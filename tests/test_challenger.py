"""Tests for offline champion-challenger scoring comparison."""

import json
import math
from pathlib import Path

import pytest

from fraud_scoring.scoring.challenger import (
    compare_predictions,
    compare_registered_candidate,
)
from fraud_scoring.scoring.model_registry import (
    RegistryError,
    evaluate_registered,
    promote_to_staging,
    register_demo_model,
)
from fraud_scoring.scoring.train_demo import train_demo_model


@pytest.fixture(scope="module")
def demo_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    directory = tmp_path_factory.mktemp("challenger-model")
    path = directory / "model.txt"
    train_demo_model(path, seed=42)
    return path


@pytest.fixture
def ready_registry(
    tmp_path: Path,
    demo_model: Path,
) -> Path:
    """Register two versions of one artifact with valid evidence."""
    for version in ("demo-v1", "demo-v2"):
        register_demo_model(
            tmp_path,
            demo_model,
            version,
        )
        evaluate_registered(tmp_path, version)

    promote_to_staging(
        tmp_path,
        "demo-v1",
        expected_current=None,
        confirm=True,
    )

    return tmp_path


def test_registered_candidate_with_identical_artifact_passes(
    ready_registry: Path,
) -> None:
    report = compare_registered_candidate(
        ready_registry,
        "demo-v2",
    )

    assert report["passed"] is True
    assert report["baseline_version"] == "demo-v1"
    assert report["candidate_version"] == "demo-v2"
    assert report["same_artifact"] is True
    assert report["samples"] == 1024
    assert report["production_approved"] is False
    assert report["metrics"]["mean_abs_score_delta"] == 0.0
    assert report["metrics"]["decision_disagreement_rate"] == 0.0
    assert report["failures"] == []


def test_comparison_does_not_change_staging(
    ready_registry: Path,
) -> None:
    pointer_path = ready_registry / "staging.json"
    original_bytes = pointer_path.read_bytes()

    compare_registered_candidate(
        ready_registry,
        "demo-v2",
    )

    assert pointer_path.read_bytes() == original_bytes


def test_comparing_staging_with_itself_is_rejected(
    ready_registry: Path,
) -> None:
    with pytest.raises(
        RegistryError,
        match="different registered version",
    ):
        compare_registered_candidate(
            ready_registry,
            "demo-v1",
        )


def test_missing_candidate_evidence_fails_closed(
    ready_registry: Path,
) -> None:
    (
        ready_registry
        / "evaluations"
        / "demo-v2.json"
    ).unlink()

    with pytest.raises(
        RegistryError,
        match="Missing or unsafe evaluation evidence",
    ):
        compare_registered_candidate(
            ready_registry,
            "demo-v2",
        )


def test_tampered_candidate_evidence_is_rejected(
    ready_registry: Path,
) -> None:
    path = (
        ready_registry
        / "evaluations"
        / "demo-v2.json"
    )

    report = json.loads(
        path.read_text(encoding="utf-8")
    )
    report["metrics"]["roc_auc"] = 0.999

    path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    with pytest.raises(
        RegistryError,
        match="Evaluation evidence mismatch",
    ):
        compare_registered_candidate(
            ready_registry,
            "demo-v2",
        )


def test_missing_baseline_evidence_is_rejected(
    ready_registry: Path,
) -> None:
    (
        ready_registry
        / "evaluations"
        / "demo-v1.json"
    ).unlink()

    with pytest.raises(
        RegistryError,
        match="Missing or unsafe evaluation evidence",
    ):
        compare_registered_candidate(
            ready_registry,
            "demo-v2",
        )


def test_unregistered_candidate_is_rejected(
    ready_registry: Path,
) -> None:
    with pytest.raises(
        RegistryError,
        match="Cannot read registry document",
    ):
        compare_registered_candidate(
            ready_registry,
            "unknown-v3",
        )


def test_identical_predictions_pass() -> None:
    result = compare_predictions(
        [0.1, 0.6, 0.95],
        [0.1, 0.6, 0.95],
    )

    assert result["passed"] is True
    assert result["metrics"]["mean_abs_score_delta"] == 0.0
    assert result["metrics"]["decision_disagreement_rate"] == 0.0
    assert result["metrics"]["decline_rate_increase"] == 0.0
    assert sum(result["transitions"].values()) == 3


def test_mean_absolute_score_delta_gate() -> None:
    result = compare_predictions(
        [0.1, 0.1],
        [0.3, 0.3],
    )

    assert result["passed"] is False
    assert "mean_abs_score_delta" in result["failures"]
    assert result["metrics"]["decision_disagreement_rate"] == 0.0


def test_decision_disagreement_gate() -> None:
    result = compare_predictions(
        [0.49, 0.49],
        [0.51, 0.51],
    )

    assert result["passed"] is False
    assert "decision_disagreement_rate" in result["failures"]
    assert result["metrics"]["decision_disagreement_rate"] == 1.0


def test_decline_rate_increase_gate() -> None:
    result = compare_predictions(
        [0.89, 0.89, 0.89, 0.89],
        [0.91, 0.91, 0.91, 0.91],
    )

    assert result["passed"] is False
    assert "decline_rate_increase" in result["failures"]
    assert result["metrics"]["decline_rate_increase"] == 1.0


def test_decision_transition_matrix() -> None:
    result = compare_predictions(
        [0.1, 0.6, 0.95, 0.6],
        [0.2, 0.95, 0.8, 0.6],
    )

    transitions = result["transitions"]

    assert transitions["APPROVE->APPROVE"] == 1
    assert transitions["REVIEW->DECLINE"] == 1
    assert transitions["DECLINE->REVIEW"] == 1
    assert transitions["REVIEW->REVIEW"] == 1
    assert sum(transitions.values()) == 4


def test_exact_score_delta_budget_is_allowed() -> None:
    result = compare_predictions(
        [0.2] * 10,
        [0.3] * 10,
    )

    assert result["passed"] is True
    assert result["metrics"]["mean_abs_score_delta"] == (
        pytest.approx(0.10)
    )


@pytest.mark.parametrize(
    ("baseline", "candidate"),
    [
        ([], []),
        ([0.1], [0.1, 0.2]),
        ([float("nan")], [0.5]),
        ([0.5], [float("inf")]),
        ([-0.1], [0.5]),
        ([0.5], [1.1]),
    ],
)
def test_invalid_prediction_vectors_are_rejected(
    baseline: list[float],
    candidate: list[float],
) -> None:
    with pytest.raises(ValueError):
        compare_predictions(
            baseline,
            candidate,
        )


@pytest.mark.parametrize(
    ("review", "decline"),
    [
        (0.9, 0.9),
        (0.95, 0.90),
        (-0.1, 0.9),
        (math.nan, 0.9),
        (True, 0.9),
    ],
)
def test_invalid_decision_thresholds_are_rejected(
    review: float,
    decline: float,
) -> None:
    with pytest.raises(ValueError):
        compare_predictions(
            [0.2],
            [0.2],
            review_threshold=review,
            decline_threshold=decline,
        )
