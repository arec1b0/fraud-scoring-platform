"""Native LightGBM inference with an explicit feature contract."""

from collections.abc import Mapping
from math import isfinite
from pathlib import Path

import lightgbm as lgb
import numpy as np


class LightGBMScorer:
    """Load one binary fraud model once and score named numeric features."""

    def __init__(self, model_path: Path) -> None:
        """Fail fast if the model is absent or not a binary classifier."""
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}"
            )

        self._model = lgb.Booster(model_file=str(model_path))
        objective = str(
            self._model.dump_model().get("objective", "")
        ).split()

        if not objective or objective[0] != "binary":
            raise ValueError(
                "Model must use the LightGBM binary objective"
            )

        self.feature_names = tuple(self._model.feature_name())

        if not self.feature_names or len(
            set(self.feature_names)
        ) != len(self.feature_names):
            raise ValueError(
                "Model must have unique, named features"
            )

    def score(self, features: Mapping[str, float]) -> float:
        """Return the model's positive-class output, without a decision."""
        expected = set(self.feature_names)
        actual = set(features)

        if expected != actual:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                f"Feature schema mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

        values: list[float] = []

        for name in self.feature_names:
            value = features[name]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(
                    f"Feature {name!r} must be a finite number"
                )

            values.append(float(value))

        row = np.asarray([values], dtype=np.float64)
        predictions = np.asarray(
            self._model.predict(row, raw_score=False)
        )

        if predictions.shape != (1,):
            raise RuntimeError(
                "Model did not return one binary prediction"
            )

        score = float(predictions[0])

        if not isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError(
                "Model returned an invalid risk score"
            )

        return score