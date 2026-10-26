"""Train a reproducible, synthetic-only LightGBM demonstration model.

This module does not use real transaction data or verified fraud labels.
Its output must never be treated as a production fraud risk model.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np


FEATURE_NAMES = ("amount", "transactions_5m")
TRAIN_ROWS = 2048
VALIDATION_ROWS = 512
BOOSTING_ROUNDS = 60


def _synthetic_batch(
    rng: np.random.Generator,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic transaction features and simulated outcomes."""
    amount = np.minimum(
        np.round(
            rng.lognormal(
                mean=4.0,
                sigma=0.85,
                size=size,
            ),
            2,
        ),
        1500.0,
    )

    velocity = rng.poisson(
        lam=3.0,
        size=size,
    )

    # Artificial labeling process, not an empirical fraud model.
    log_odds = (
        -4.6
        + 0.008 * amount
        + 0.52 * velocity
    )

    simulated_probability = 1.0 / (
        1.0 + np.exp(-np.clip(log_odds, -30.0, 30.0))
    )

    labels = (
        rng.random(size) < simulated_probability
    ).astype(np.int32)

    features = np.column_stack(
        (amount, velocity)
    ).astype(np.float64)

    return features, labels


def train_demo_model(
    output_path: Path,
    *,
    seed: int = 42,
) -> float:
    """Train and save a demo model; return validation Brier score.

    The output file must not already exist. Training and validation
    samples are generated independently from the seeded random stream.

    The validation dataset is never passed into model training.
    """
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError(
            "seed must be an integer in [0, 2147483647]"
        )

    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(
            f"Model artifact already exists: {output_path}"
        )

    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"Model directory does not exist: {output_path.parent}"
        )

    rng = np.random.default_rng(seed)

    train_features, train_labels = _synthetic_batch(
        rng,
        TRAIN_ROWS,
    )

    validation_features, validation_labels = _synthetic_batch(
        rng,
        VALIDATION_ROWS,
    )

    if (
        len(np.unique(train_labels)) != 2
        or len(np.unique(validation_labels)) != 2
    ):
        raise RuntimeError(
            "Synthetic data must contain both outcome classes"
        )

    training_dataset = lgb.Dataset(
        train_features,
        label=train_labels,
        feature_name=list(FEATURE_NAMES),
    )

    model = lgb.train(
        params={
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.06,
            "num_leaves": 15,
            "min_data_in_leaf": 20,
            "verbosity": -1,
            "num_threads": 1,
            "deterministic": True,
            "force_col_wise": True,
            "seed": seed,
        },
        train_set=training_dataset,
        num_boost_round=BOOSTING_ROUNDS,
    )

    if tuple(model.feature_name()) != FEATURE_NAMES:
        raise RuntimeError(
            "Trained model feature names do not match the contract"
        )

    predictions = np.asarray(
        model.predict(validation_features),
        dtype=np.float64,
    )

    if (
        predictions.shape != (VALIDATION_ROWS,)
        or not np.all(np.isfinite(predictions))
        or np.any(predictions < 0.0)
        or np.any(predictions > 1.0)
    ):
        raise RuntimeError(
            "Model produced invalid validation predictions"
        )

    brier_score = float(
        np.mean(
            np.square(
                predictions - validation_labels
            )
        )
    )

    # Exclusive creation prevents accidental model replacement.
    with output_path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        stream.write(model.model_to_string())

    return brier_score


def main() -> None:
    """Train a local demonstration artifact from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Train a synthetic-only LightGBM demonstration model."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/demo_fraud_model.txt"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    brier_score = train_demo_model(
        args.output,
        seed=args.seed,
    )

    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "data_origin": "synthetic",
                "model_type": "lightgbm_binary",
                "features": list(FEATURE_NAMES),
                "seed": args.seed,
                "training_rows": TRAIN_ROWS,
                "validation_rows": VALIDATION_ROWS,
                "validation_brier_score": round(
                    brier_score,
                    6,
                ),
                "production_approved": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
