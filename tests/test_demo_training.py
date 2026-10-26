"""Tests for reproducible synthetic LightGBM demonstration training."""

from pathlib import Path

import pytest

from fraud_scoring.scoring.ml import LightGBMScorer
from fraud_scoring.scoring.train_demo import train_demo_model


def test_trains_model_compatible_with_scoring_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "demo_model.txt"

    brier = train_demo_model(path)

    assert path.is_file()
    assert 0.0 <= brier <= 1.0

    scorer = LightGBMScorer(path)

    assert scorer.feature_names == (
        "amount",
        "transactions_5m",
    )

    score = scorer.score(
        {
            "amount": 42.50,
            "transactions_5m": 3.0,
        }
    )

    assert 0.0 <= score <= 1.0


def test_training_is_reproducible_for_same_seed(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"

    first_brier = train_demo_model(
        first_path,
        seed=17,
    )

    second_brier = train_demo_model(
        second_path,
        seed=17,
    )

    assert first_brier == pytest.approx(
        second_brier,
        abs=1e-12,
    )

    first = LightGBMScorer(first_path)
    second = LightGBMScorer(second_path)

    for amount, velocity in (
        (10.0, 0.0),
        (42.5, 3.0),
        (400.0, 9.0),
    ):
        features = {
            "amount": amount,
            "transactions_5m": velocity,
        }

        assert first.score(features) == pytest.approx(
            second.score(features),
            abs=1e-12,
        )


def test_existing_model_is_never_overwritten(
    tmp_path: Path,
) -> None:
    path = tmp_path / "existing.txt"
    original_content = "existing-model-content"

    path.write_text(
        original_content,
        encoding="utf-8",
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        train_demo_model(path)

    assert path.read_text(
        encoding="utf-8"
    ) == original_content


def test_missing_output_directory_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing" / "model.txt"

    with pytest.raises(
        FileNotFoundError,
        match="Model directory",
    ):
        train_demo_model(path)

    assert not path.exists()


@pytest.mark.parametrize(
    "invalid_seed",
    [
        -1,
        True,
        1.5,
        2**31,
    ],
)
def test_invalid_random_seed_is_rejected(
    tmp_path: Path,
    invalid_seed: object,
) -> None:
    path = tmp_path / "demo.txt"

    with pytest.raises(ValueError, match="seed"):
        train_demo_model(
            path,
            seed=invalid_seed,
        )

    assert not path.exists()
