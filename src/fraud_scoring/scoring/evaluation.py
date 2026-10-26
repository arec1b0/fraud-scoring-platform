"""Reproducible evaluation of synthetic-only fraud models.

Evaluation data is simulated, not independently verified fraud data.
The resulting metrics cannot establish production model suitability.
"""

import hashlib
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np

from fraud_scoring.scoring.train_demo import (
    FEATURE_NAMES,
    _synthetic_batch,
)


EVALUATION_VERSION = 1
EVALUATION_SEED = 202604
EVALUATION_ROWS = 1024
CLASSIFICATION_THRESHOLD = 0.50

QUALITY_POLICY = {
    "min_roc_auc": 0.60,
    "min_pr_auc": 0.12,
    "min_precision": 0.10,
    "min_recall": 0.10,
    "min_f1": 0.10,
    "max_brier_score": 0.30,
    "max_roc_auc_regression": 0.02,
    "max_pr_auc_regression": 0.02,
    "max_brier_regression": 0.02,
}


def evaluation_dataset() -> tuple[np.ndarray, np.ndarray, str]:
    """Create one fixed, independent synthetic evaluation sample."""
    generator = np.random.default_rng(EVALUATION_SEED)
    features, labels = _synthetic_batch(
        generator,
        EVALUATION_ROWS,
    )

    features = np.asarray(
        features,
        dtype="<f8",
        order="C",
    )
    labels = np.asarray(
        labels,
        dtype="<i4",
        order="C",
    )

    digest = hashlib.sha256()
    digest.update(b"synthetic-fraud-evaluation-v1\n")
    digest.update(
        np.asarray(features.shape, dtype="<i8").tobytes()
    )
    digest.update(features.tobytes(order="C"))
    digest.update(labels.tobytes(order="C"))

    return features, labels, digest.hexdigest()


def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> dict[str, float]:
    """Compute binary metrics, including tie-aware ROC-AUC and AP."""
    y = np.asarray(labels)
    p = np.asarray(scores, dtype=np.float64)

    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape:
        raise ValueError("Labels and scores must be 1-D equal-length arrays")

    if y.size == 0 or not np.all((y == 0) | (y == 1)):
        raise ValueError("Labels must contain only binary outcomes")

    if np.unique(y).size != 2:
        raise ValueError("Both outcome classes are required")

    if not np.all(np.isfinite(p)) or np.any(
        (p < 0.0) | (p > 1.0)
    ):
        raise ValueError("Scores must be finite values in [0, 1]")

    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("Classification threshold must be in (0, 1)")

    positives = int(np.count_nonzero(y == 1))
    negatives = y.size - positives

    order = np.argsort(-p, kind="mergesort")
    sorted_scores = p[order]
    sorted_labels = y[order].astype(np.int64)

    # Evaluate every unique score threshold; never split ties.
    group_ends = np.concatenate(
        (
            np.flatnonzero(np.diff(sorted_scores) != 0),
            np.array([y.size - 1]),
        )
    )

    cumulative_tp = np.cumsum(sorted_labels)
    tp = cumulative_tp[group_ends].astype(np.float64)
    fp = (group_ends + 1).astype(np.float64) - tp

    tpr = np.concatenate(([0.0], tp / positives))
    fpr = np.concatenate(([0.0], fp / negatives))

    roc_auc = float(np.trapezoid(tpr, fpr))

    precision_curve = tp / (tp + fp)
    previous_tp = np.concatenate(([0.0], tp[:-1]))

    # Average Precision: stepwise area under precision-recall.
    pr_auc = float(
        np.sum(
            ((tp - previous_tp) / positives)
            * precision_curve
        )
    )

    predicted_positive = p >= threshold

    true_positive = int(
        np.count_nonzero(predicted_positive & (y == 1))
    )
    false_positive = int(
        np.count_nonzero(predicted_positive & (y == 0))
    )
    false_negative = int(
        np.count_nonzero(~predicted_positive & (y == 1))
    )

    predicted_count = true_positive + false_positive

    precision = (
        true_positive / predicted_count
        if predicted_count
        else 0.0
    )
    recall = true_positive / (true_positive + false_negative)

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    brier = float(
        np.mean(np.square(p - y.astype(np.float64)))
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier_score": brier,
    }


def make_evaluation_report(
    *,
    version: str,
    artifact_path: Path,
    artifact_sha256: str,
) -> dict[str, object]:
    """Recompute all evidence from the registered model artifact."""
    features, labels, dataset_hash = evaluation_dataset()

    model = lgb.Booster(model_file=str(artifact_path))

    if tuple(model.feature_name()) != FEATURE_NAMES:
        raise ValueError("Evaluation model feature schema mismatch")

    predictions = np.asarray(
        model.predict(features, num_threads=1),
        dtype=np.float64,
    )

    metrics = compute_metrics(labels, predictions)

    return {
        "evaluation_version": EVALUATION_VERSION,
        "version": version,
        "artifact_sha256": artifact_sha256,
        "dataset_sha256": dataset_hash,
        "dataset_origin": "synthetic",
        "evaluation_seed": EVALUATION_SEED,
        "evaluation_rows": EVALUATION_ROWS,
        "positive_rows": int(np.count_nonzero(labels)),
        "feature_names": list(FEATURE_NAMES),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "pr_auc_definition": "average_precision",
        "metrics": {
            name: round(value, 10)
            for name, value in metrics.items()
        },
        "policy": dict(QUALITY_POLICY),
        "production_approved": False,
    }


def enforce_quality_policy(
    report: dict[str, object],
    *,
    baseline: dict[str, object] | None = None,
) -> None:
    """Reject quality failures or excessive baseline degradation."""
    metrics = report["metrics"]

    for metric in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
        minimum = QUALITY_POLICY[f"min_{metric}"]

        if metrics[metric] < minimum:
            raise ValueError(
                f"Quality gate failed: {metric} below {minimum}"
            )

    if metrics["brier_score"] > QUALITY_POLICY["max_brier_score"]:
        raise ValueError("Quality gate failed: Brier score too high")

    if baseline is None:
        return

    if (
        baseline["dataset_sha256"] != report["dataset_sha256"]
        or baseline["classification_threshold"]
        != report["classification_threshold"]
    ):
        raise ValueError("Baseline uses an incompatible evaluation dataset")

    reference = baseline["metrics"]

    for metric in ("roc_auc", "pr_auc"):
        permitted = QUALITY_POLICY[
            f"max_{metric}_regression"
        ]

        if metrics[metric] + permitted < reference[metric] - 1e-10:
            raise ValueError(
                f"Baseline regression gate failed: {metric}"
            )

    if (
        metrics["brier_score"]
        > reference["brier_score"]
        + QUALITY_POLICY["max_brier_regression"]
        + 1e-10
    ):
        raise ValueError(
            "Baseline regression gate failed: brier_score"
        )
