"""Deterministic, side-effect-free fraud rule evaluation."""

from collections.abc import Mapping, Set
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal


@dataclass(frozen=True, slots=True)
class RuleFinding:
    """A matched rule and its recommended action."""

    rule_id: Literal["MERCHANT_BLOCKED", "AMOUNT_REVIEW_LIMIT"]
    suggested_action: Literal["DECLINE", "REVIEW"]


def evaluate_rules(
    *,
    amount: Decimal,
    currency: str,
    merchant_id: str,
    blocked_merchant_ids: Set[str],
    review_limits: Mapping[str, Decimal],
) -> tuple[RuleFinding, ...]:
    """Evaluate merchant and amount rules for one transaction.

    The caller supplies all policy values explicitly.

    No I/O, external services, model inference, or final decision
    aggregation occurs in this function.
    """
    if not amount.is_finite() or amount <= 0:
        raise ValueError("Amount must be finite and positive")

    if not merchant_id:
        raise ValueError("Merchant ID must not be empty")

    if currency not in review_limits:
        raise ValueError(
            f"No review limit configured for currency {currency!r}"
        )

    limit = review_limits[currency]

    if not limit.is_finite() or limit <= 0:
        raise ValueError(
            "Review limit must be finite and positive"
        )

    findings: list[RuleFinding] = []

    if merchant_id in blocked_merchant_ids:
        findings.append(
            RuleFinding(
                rule_id="MERCHANT_BLOCKED",
                suggested_action="DECLINE",
            )
        )

    if amount >= limit:
        findings.append(
            RuleFinding(
                rule_id="AMOUNT_REVIEW_LIMIT",
                suggested_action="REVIEW",
            )
        )

    return tuple(findings)