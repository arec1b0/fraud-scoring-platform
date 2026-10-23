"""Tests for deterministic fraud decision aggregation."""

import pytest

from fraud_scoring.scoring import RuleFinding, resolve_decision


REVIEW = RuleFinding(
    rule_id="AMOUNT_REVIEW_LIMIT",
    suggested_action="REVIEW",
)

DECLINE = RuleFinding(
    rule_id="MERCHANT_BLOCKED",
    suggested_action="DECLINE",
)


def decide(
    score: float,
    findings: tuple[RuleFinding, ...] = (),
) -> str:
    """Evaluate the policy using test-only thresholds."""
    return resolve_decision(
        risk_score=score,
        findings=findings,
        review_threshold=0.50,
        decline_threshold=0.90,
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, "APPROVE"),
        (0.49, "APPROVE"),
        (0.50, "REVIEW"),
        (0.8999, "REVIEW"),
        (0.90, "DECLINE"),
        (1.0, "DECLINE"),
    ],
)
def test_model_score_thresholds(
    score: float,
    expected: str,
) -> None:
    assert decide(score) == expected


@pytest.mark.parametrize(
    ("score", "findings", "expected"),
    [
        (0.10, (REVIEW,), "REVIEW"),
        (0.10, (DECLINE,), "DECLINE"),
        (0.70, (DECLINE,), "DECLINE"),
        (0.99, (REVIEW,), "DECLINE"),
        (0.10, (REVIEW, DECLINE), "DECLINE"),
    ],
)
def test_rule_precedence(
    score: float,
    findings: tuple[RuleFinding, ...],
    expected: str,
) -> None:
    assert decide(score, findings) == expected


def test_finding_order_does_not_change_decision() -> None:
    first = decide(0.10, (REVIEW, DECLINE))
    second = decide(0.10, (DECLINE, REVIEW))

    assert first == second == "DECLINE"


def test_accepts_iterable_findings() -> None:
    findings = (finding for finding in (REVIEW,))

    result = resolve_decision(
        risk_score=0.10,
        findings=findings,
        review_threshold=0.50,
        decline_threshold=0.90,
    )

    assert result == "REVIEW"


@pytest.mark.parametrize(
    "invalid_score",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
        True,
        "0.5",
    ],
)
def test_invalid_model_score_is_rejected(
    invalid_score: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="risk_score",
    ):
        resolve_decision(
            risk_score=invalid_score,
            findings=(),
            review_threshold=0.50,
            decline_threshold=0.90,
        )


@pytest.mark.parametrize(
    ("review_threshold", "decline_threshold"),
    [
        (-0.10, 0.90),
        (0.50, 1.10),
        (0.50, 0.50),
        (0.90, 0.50),
        (float("nan"), 0.90),
        (0.50, float("inf")),
        (True, 0.90),
        (0.50, True),
    ],
)
def test_invalid_thresholds_are_rejected(
    review_threshold: object,
    decline_threshold: object,
) -> None:
    with pytest.raises(ValueError):
        resolve_decision(
            risk_score=0.30,
            findings=(),
            review_threshold=review_threshold,
            decline_threshold=decline_threshold,
        )


@pytest.mark.parametrize(
    ("finding", "error"),
    [
        ("invalid", TypeError),
        (
            RuleFinding(
                "MERCHANT_BLOCKED",
                "REVIEW",
            ),
            ValueError,
        ),
        (
            RuleFinding(
                "AMOUNT_REVIEW_LIMIT",
                "DECLINE",
            ),
            ValueError,
        ),
        (
            RuleFinding(
                "UNKNOWN_RULE",
                "DECLINE",
            ),
            ValueError,
        ),
    ],
)
def test_invalid_rule_findings_are_rejected(
    finding: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        resolve_decision(
            risk_score=0.10,
            findings=(finding,),
            review_threshold=0.50,
            decline_threshold=0.90,
        )


def test_valid_decline_does_not_hide_malformed_finding() -> None:
    malformed = RuleFinding(
        "AMOUNT_REVIEW_LIMIT",
        "DECLINE",
    )

    with pytest.raises(ValueError):
        decide(0.99, (DECLINE, malformed))


def test_threshold_endpoints_are_supported() -> None:
    result = resolve_decision(
        risk_score=0.0,
        findings=(),
        review_threshold=0.0,
        decline_threshold=1.0,
    )

    assert result == "REVIEW"


def test_duplicate_rules_do_not_change_decision() -> None:
    result = decide(0.10, (REVIEW, REVIEW))

    assert result == "REVIEW"
