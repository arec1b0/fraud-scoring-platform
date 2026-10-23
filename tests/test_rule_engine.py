"""Tests for deterministic fraud rule evaluation."""

from decimal import Decimal

import pytest

from fraud_scoring.scoring import RuleFinding, evaluate_rules


@pytest.mark.parametrize(
    ("amount", "merchant_id", "expected"),
    [
        (
            Decimal("50"),
            "merchant-ok",
            (),
        ),
        (
            Decimal("100"),
            "merchant-ok",
            (
                RuleFinding(
                    "AMOUNT_REVIEW_LIMIT",
                    "REVIEW",
                ),
            ),
        ),
        (
            Decimal("150"),
            "merchant-ok",
            (
                RuleFinding(
                    "AMOUNT_REVIEW_LIMIT",
                    "REVIEW",
                ),
            ),
        ),
        (
            Decimal("50"),
            "merchant-blocked",
            (
                RuleFinding(
                    "MERCHANT_BLOCKED",
                    "DECLINE",
                ),
            ),
        ),
        (
            Decimal("120"),
            "merchant-blocked",
            (
                RuleFinding(
                    "MERCHANT_BLOCKED",
                    "DECLINE",
                ),
                RuleFinding(
                    "AMOUNT_REVIEW_LIMIT",
                    "REVIEW",
                ),
            ),
        ),
    ],
)
def test_evaluates_rules(
    amount: Decimal,
    merchant_id: str,
    expected: tuple[RuleFinding, ...],
) -> None:
    result = evaluate_rules(
        amount=amount,
        currency="CHF",
        merchant_id=merchant_id,
        blocked_merchant_ids=frozenset(
            {"merchant-blocked"}
        ),
        review_limits={"CHF": Decimal("100")},
    )

    assert result == expected


@pytest.mark.parametrize(
    "invalid_amount",
    [
        Decimal("0"),
        Decimal("-1"),
        Decimal("NaN"),
        Decimal("Infinity"),
    ],
)
def test_rejects_invalid_amount(
    invalid_amount: Decimal,
) -> None:
    with pytest.raises(ValueError, match="Amount"):
        evaluate_rules(
            amount=invalid_amount,
            currency="CHF",
            merchant_id="merchant-ok",
            blocked_merchant_ids=frozenset(),
            review_limits={"CHF": Decimal("100")},
        )


@pytest.mark.parametrize(
    "invalid_limit",
    [
        Decimal("0"),
        Decimal("-1"),
        Decimal("NaN"),
        Decimal("Infinity"),
    ],
)
def test_rejects_invalid_review_limit(
    invalid_limit: Decimal,
) -> None:
    with pytest.raises(ValueError, match="Review limit"):
        evaluate_rules(
            amount=Decimal("50"),
            currency="CHF",
            merchant_id="merchant-ok",
            blocked_merchant_ids=frozenset(),
            review_limits={"CHF": invalid_limit},
        )


def test_rejects_missing_currency_policy() -> None:
    with pytest.raises(ValueError, match="No review limit"):
        evaluate_rules(
            amount=Decimal("50"),
            currency="EUR",
            merchant_id="merchant-ok",
            blocked_merchant_ids=frozenset(),
            review_limits={"CHF": Decimal("100")},
        )


def test_rejects_empty_merchant_id() -> None:
    with pytest.raises(ValueError, match="Merchant ID"):
        evaluate_rules(
            amount=Decimal("50"),
            currency="CHF",
            merchant_id="",
            blocked_merchant_ids=frozenset(),
            review_limits={"CHF": Decimal("100")},
        )


def test_uses_currency_specific_threshold() -> None:
    policy = {
        "CHF": Decimal("100"),
        "USD": Decimal("300"),
    }

    common = {
        "amount": Decimal("200"),
        "merchant_id": "merchant-ok",
        "blocked_merchant_ids": frozenset(),
        "review_limits": policy,
    }

    chf_result = evaluate_rules(
        currency="CHF",
        **common,
    )

    usd_result = evaluate_rules(
        currency="USD",
        **common,
    )

    assert chf_result == (
        RuleFinding("AMOUNT_REVIEW_LIMIT", "REVIEW"),
    )
    assert usd_result == ()