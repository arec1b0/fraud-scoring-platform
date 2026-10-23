"""Tests for the native LightGBM inference boundary."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from fraud_scoring.scoring.ml import LightGBMScorer


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Train a small disposable classifier to exercise real model I/O."""
    values = np.asarray(
        [[float(i), float(i % 8)] for i in range(80)],
        dtype=np.float64,
    )
    labels = np.asarray(
        [int(i >= 40) for i in range(80)],
        dtype=np.int64,
    )

    train_data = lgb.Dataset(
        values,
        label=labels,
        feature_name=["amount", "transactions_5m"],
    )

    model = lgb.train(
        {
            "objective": "binary",
            "verbosity": -1,
            "num_threads": 1,
            "min_data_in_leaf": 2,
        },
        train_data,
        num_boost_round=5,
    )

    path = tmp_path / "binary_model.txt"
    model.save_model(str(path))
    return path


def test_scores_real_model_with_artifact_feature_order(
    model_path: Path,
) -> None:
    scorer = LightGBMScorer(model_path)

    actual = scorer.score({
        "transactions_5m": 3.0,
        "amount": 75.0,
    })

    loaded = lgb.Booster(model_file=str(model_path))
    expected = float(
        loaded.predict(
            np.asarray([[75.0, 3.0]], dtype=np.float64)
        )[0]
    )

    assert scorer.feature_names == (
        "amount",
        "transactions_5m",
    )
    assert actual == pytest.approx(expected)
    assert 0.0 <= actual <= 1.0


def test_missing_model_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Model artifact not found",
    ):
        LightGBMScorer(tmp_path / "missing.txt")


def test_non_binary_model_is_rejected(
    tmp_path: Path,
) -> None:
    dataset = lgb.Dataset(
        np.asarray([[float(i)] for i in range(30)]),
        label=np.asarray([float(i) for i in range(30)]),
        feature_name=["value"],
    )

    model = lgb.train(
        {
            "objective": "regression",
            "verbosity": -1,
            "num_threads": 1,
        },
        dataset,
        num_boost_round=2,
    )

    path = tmp_path / "regressor.txt"
    model.save_model(str(path))

    with pytest.raises(
        ValueError,
        match="binary objective",
    ):
        LightGBMScorer(path)


@pytest.mark.parametrize(
    "features",
    [
        {"amount": 25.0},
        {
            "amount": 25.0,
            "transactions_5m": 1.0,
            "unknown": 0.0,
        },
    ],
)
def test_feature_schema_mismatch_is_rejected(
    model_path: Path,
    features: dict[str, float],
) -> None:
    scorer = LightGBMScorer(model_path)

    with pytest.raises(
        ValueError,
        match="Feature schema mismatch",
    ):
        scorer.score(features)


@pytest.mark.parametrize(
    "invalid",
    [
        float("nan"),
        float("inf"),
        True,
        "25",
    ],
)
def test_invalid_feature_is_rejected(
    model_path: Path,
    invalid: object,
) -> None:
    scorer = LightGBMScorer(model_path)

    with pytest.raises(
        ValueError,
        match="must be a finite number",
    ):
        scorer.score({
            "amount": invalid,
            "transactions_5m": 1.0,
        })