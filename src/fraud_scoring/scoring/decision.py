"""Deterministic aggregation of model scores and fraud rule findings."""

from collections.abc import Iterable
from math import isfinite
from typing import Literal

from fraud_scoring.scoring.rules import RuleFinding


def _validate_unit_interval(value: object, name: str) -> float:
    """Accept only finite numeric values in the closed unit interval."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 <= value <= 1.0
        or not isfinite(value)
    ):
        raise ValueError(
            f"{name} must be a finite number in [0, 1]"
        )

    return float(value)


def resolve_decision(
    *,
    risk_score: float,
    findings: Iterable[RuleFinding],
    review_threshold: float,
    decline_threshold: float,
) -> Literal["APPROVE", "REVIEW", "DECLINE"]:
    """Resolve one decision using an explicit, deterministic policy.

    A DECLINE recommendation takes precedence over REVIEW.
    A REVIEW recommendation takes precedence over APPROVE.

    Thresholds are caller-provided policy inputs, not model
    calibration parameters or fraud probability guarantees.

    All findings are validated before returning a decision.
    """
    score = _validate_unit_interval(
        risk_score,
        "risk_score",
    )
    review = _validate_unit_interval(
        review_threshold,
        "review_threshold",
    )
    decline = _validate_unit_interval(
        decline_threshold,
        "decline_threshold",
    )

    if review >= decline:
        raise ValueError(
            "review_threshold must be below decline_threshold"
        )

    rule_decision: Literal[
        "APPROVE", "REVIEW", "DECLINE"
    ] = "APPROVE"

    for finding in findings:
        if not isinstance(finding, RuleFinding):
            raise TypeError(
                "findings must contain RuleFinding instances"
            )

        if finding.rule_id == "MERCHANT_BLOCKED":
            if finding.suggested_action != "DECLINE":
                raise ValueError(
                    "MERCHANT_BLOCKED must recommend DECLINE"
                )

            rule_decision = "DECLINE"

        elif finding.rule_id == "AMOUNT_REVIEW_LIMIT":
            if finding.suggested_action != "REVIEW":
                raise ValueError(
                    "AMOUNT_REVIEW_LIMIT must recommend REVIEW"
                )

            if rule_decision == "APPROVE":
                rule_decision = "REVIEW"

        else:
            raise ValueError(
                f"Unknown rule identifier: {finding.rule_id!r}"
            )

    if rule_decision == "DECLINE" or score >= decline:
        return "DECLINE"

    if rule_decision == "REVIEW" or score >= review:
        return "REVIEW"

    return "APPROVE"
