"""Public monitoring interfaces."""

from fraud_scoring.monitoring.drift import (
    score_population_stability_index,
)

__all__ = ["score_population_stability_index"]
