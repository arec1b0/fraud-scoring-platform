"""Offline champion-challenger comparison for synthetic models.

This module never promotes or deploys models. It compares registered,
evidence-verified models on the same fixed synthetic dataset.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

from fraud_scoring.scoring.decision import resolve_decision
from fraud_scoring.scoring.evaluation import (
    enforce_quality_policy,
    evaluation_dataset,
)
from fraud_scoring.scoring.model_registry import (
    RegistryError,
    verify_evaluation,
    verify_registered,
    verify_staging,
)


DECISIONS = ("APPROVE", "REVIEW", "DECLINE")

MAX_MEAN_ABS_SCORE_DELTA = 0.10
MAX_DECISION_DISAGREEMENT_RATE = 0.10
MAX_DECLINE_RATE_INCREASE = 0.05


def _validate_scores(
    baseline_scores: object,
    candidate_scores: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Require equal, nonempty vectors of valid risk scores."""
    try:
        baseline = np.asarray(
            baseline_scores,
            dtype=np.float64,
        )
        candidate = np.asarray(
            candidate_scores,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Scores must be numeric arrays") from exc

    if (
        baseline.ndim != 1
        or candidate.ndim != 1
        or baseline.size == 0
        or baseline.shape != candidate.shape
    ):
        raise ValueError(
            "Score vectors must have equal, nonzero 1-D shapes"
        )

    for scores in (baseline, candidate):
        if (
            not np.all(np.isfinite(scores))
            or np.any(scores < 0.0)
            or np.any(scores > 1.0)
        ):
            raise ValueError(
                "Scores must contain finite values in [0, 1]"
            )

    return baseline, candidate


def compare_predictions(
    baseline_scores: object,
    candidate_scores: object,
    *,
    review_threshold: float = 0.50,
    decline_threshold: float = 0.90,
) -> dict[str, object]:
    """Compare scores and policy decisions using deterministic limits."""
    baseline, candidate = _validate_scores(
        baseline_scores,
        candidate_scores,
    )

    # Delegate threshold validation to the production decision policy.
    resolve_decision(
        risk_score=0.5,
        findings=(),
        review_threshold=review_threshold,
        decline_threshold=decline_threshold,
    )

    baseline_decisions = [
        resolve_decision(
            risk_score=float(score),
            findings=(),
            review_threshold=review_threshold,
            decline_threshold=decline_threshold,
        )
        for score in baseline
    ]

    candidate_decisions = [
        resolve_decision(
            risk_score=float(score),
            findings=(),
            review_threshold=review_threshold,
            decline_threshold=decline_threshold,
        )
        for score in candidate
    ]

    transitions = {
        f"{before}->{after}": sum(
            old == before and new == after
            for old, new in zip(
                baseline_decisions,
                candidate_decisions,
                strict=True,
            )
        )
        for before in DECISIONS
        for after in DECISIONS
    }

    count = baseline.size
    absolute_delta = np.abs(candidate - baseline)

    mean_delta = float(np.mean(absolute_delta))
    sorted_delta = np.sort(absolute_delta)
    p95_index = max(0, math.ceil(0.95 * count) - 1)
    p95_delta = float(sorted_delta[p95_index])
    max_delta = float(sorted_delta[-1])

    disagreement_rate = (
        sum(
            old != new
            for old, new in zip(
                baseline_decisions,
                candidate_decisions,
                strict=True,
            )
        )
        / count
    )

    baseline_counts = {
        decision: baseline_decisions.count(decision)
        for decision in DECISIONS
    }
    candidate_counts = {
        decision: candidate_decisions.count(decision)
        for decision in DECISIONS
    }

    decline_rate_increase = (
        candidate_counts["DECLINE"]
        - baseline_counts["DECLINE"]
    ) / count

    failures: list[str] = []

    if mean_delta > MAX_MEAN_ABS_SCORE_DELTA + 1e-12:
        failures.append("mean_abs_score_delta")

    if (
        disagreement_rate
        > MAX_DECISION_DISAGREEMENT_RATE + 1e-12
    ):
        failures.append("decision_disagreement_rate")

    if (
        decline_rate_increase
        > MAX_DECLINE_RATE_INCREASE + 1e-12
    ):
        failures.append("decline_rate_increase")

    return {
        "passed": not failures,
        "samples": int(count),
        "metrics": {
            "mean_abs_score_delta": round(mean_delta, 10),
            "p95_abs_score_delta": round(p95_delta, 10),
            "max_abs_score_delta": round(max_delta, 10),
            "decision_disagreement_rate": round(
                disagreement_rate,
                10,
            ),
            "decline_rate_increase": round(
                decline_rate_increase,
                10,
            ),
        },
        "decision_counts": {
            "baseline": baseline_counts,
            "candidate": candidate_counts,
        },
        "transitions": transitions,
        "thresholds": {
            "review": review_threshold,
            "decline": decline_threshold,
        },
        "policy": {
            "max_mean_abs_score_delta": (
                MAX_MEAN_ABS_SCORE_DELTA
            ),
            "max_decision_disagreement_rate": (
                MAX_DECISION_DISAGREEMENT_RATE
            ),
            "max_decline_rate_increase": (
                MAX_DECLINE_RATE_INCREASE
            ),
        },
        "failures": failures,
    }


def _predict(
    artifact_path: Path,
    features: np.ndarray,
) -> np.ndarray:
    """Run native LightGBM batch inference without HTTP or Redis."""
    model = lgb.Booster(model_file=str(artifact_path))

    if tuple(model.feature_name()) != (
        "amount",
        "transactions_5m",
    ):
        raise RegistryError("Comparison model feature schema mismatch")

    predictions = np.asarray(
        model.predict(
            features,
            num_threads=1,
        ),
        dtype=np.float64,
    )

    if predictions.shape != (features.shape[0],):
        raise RegistryError("Unexpected batch prediction shape")

    return predictions


def compare_registered_candidate(
    root: Path,
    candidate_version: str,
    *,
    review_threshold: float = 0.50,
    decline_threshold: float = 0.90,
) -> dict[str, object]:
    """Evaluate the challenger against the current STAGING version.

    Both models must have reproducible Step 24 evidence. This function
    is read-only and never changes the STAGING pointer.
    """
    root = Path(root)

    starting_pointer = verify_staging(root)
    baseline_version = starting_pointer["version"]

    if candidate_version == baseline_version:
        raise RegistryError(
            "Candidate must be a different registered version"
        )

    baseline_manifest = verify_registered(
        root,
        baseline_version,
    )
    candidate_manifest = verify_registered(
        root,
        candidate_version,
    )

    baseline_evidence = verify_evaluation(
        root,
        baseline_version,
    )
    candidate_evidence = verify_evaluation(
        root,
        candidate_version,
    )

    try:
        enforce_quality_policy(
            candidate_evidence,
            baseline=baseline_evidence,
        )
    except ValueError as exc:
        raise RegistryError(
            f"Candidate quality gate failed: {exc}"
        ) from exc

    features, _, dataset_sha256 = evaluation_dataset()

    if (
        baseline_evidence["dataset_sha256"]
        != dataset_sha256
        or candidate_evidence["dataset_sha256"]
        != dataset_sha256
    ):
        raise RegistryError(
            "Comparison dataset does not match evaluation evidence"
        )

    baseline_path = (
        root / "artifacts" / f"{baseline_version}.txt"
    )
    candidate_path = (
        root / "artifacts" / f"{candidate_version}.txt"
    )

    baseline_scores = _predict(
        baseline_path,
        features,
    )
    candidate_scores = _predict(
        candidate_path,
        features,
    )

    comparison = compare_predictions(
        baseline_scores,
        candidate_scores,
        review_threshold=review_threshold,
        decline_threshold=decline_threshold,
    )

    # Detect a staging change that occurred during evaluation.
    if verify_staging(root) != starting_pointer:
        raise RegistryError(
            "STAGING changed during champion-challenger comparison"
        )

    return {
        "comparison_version": 1,
        "data_origin": "synthetic",
        "baseline_version": baseline_version,
        "baseline_sha256": baseline_manifest["sha256"],
        "candidate_version": candidate_version,
        "candidate_sha256": candidate_manifest["sha256"],
        "same_artifact": (
            baseline_manifest["sha256"]
            == candidate_manifest["sha256"]
        ),
        "dataset_sha256": dataset_sha256,
        "production_approved": False,
        **comparison,
    }


def main() -> int:
    """Run a read-only champion-challenger comparison from the CLI."""
    parser = argparse.ArgumentParser(
        description="Compare a registered candidate with STAGING."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/model_registry"),
    )
    parser.add_argument(
        "--candidate",
        required=True,
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--decline-threshold",
        type=float,
        default=0.90,
    )

    args = parser.parse_args()

    try:
        report = compare_registered_candidate(
            args.registry,
            args.candidate,
            review_threshold=args.review_threshold,
            decline_threshold=args.decline_threshold,
        )
    except (
        ValueError,
        OSError,
        RuntimeError,
        lgb.basic.LightGBMError,
    ) as exc:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps(report, indent=2))

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
