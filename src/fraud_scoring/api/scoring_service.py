"""Coordinate rule evaluation, online features, and ML scoring."""

from collections.abc import Mapping, Set
from datetime import datetime
from decimal import Decimal

from fraud_scoring.api.contracts import ScoreRequest, ScoreResponse
from fraud_scoring.features.online import (
    FeatureUnavailableError,
    OnlineVelocityStore,
)
from fraud_scoring.scoring.decision import resolve_decision
from fraud_scoring.scoring.ml import LightGBMScorer
from fraud_scoring.scoring.rules import evaluate_rules


def run_scoring_pipeline(
    transaction: ScoreRequest,
    *,
    feature_store: OnlineVelocityStore,
    model: LightGBMScorer,
    evaluated_at: datetime,
    max_feature_age_seconds: int,
    blocked_merchant_ids: Set[str],
    review_limits: Mapping[str, Decimal],
    review_threshold: float,
    decline_threshold: float,
) -> ScoreResponse:
    """Score one transaction with an explicit, fail-closed policy.

    The caller owns infrastructure initialization and configuration.

    Rule findings never substitute for missing model predictions.
    Fresh online features are required for every completed decision.
    """
    if (
        not isinstance(evaluated_at, datetime)
        or evaluated_at.tzinfo is None
        or evaluated_at.utcoffset() is None
    ):
        raise ValueError("evaluated_at must be timezone-aware")

    if transaction.occurred_at > evaluated_at:
        raise ValueError(
            "Transaction timestamp is in the future"
        )

    findings = evaluate_rules(
        amount=transaction.amount,
        currency=transaction.currency,
        merchant_id=transaction.merchant_id,
        blocked_merchant_ids=blocked_merchant_ids,
        review_limits=review_limits,
    )

    snapshot = feature_store.get_fresh(
        transaction.customer_id,
        now=evaluated_at,
        max_age_seconds=max_feature_age_seconds,
    )

    # Never use features computed after the event being scored.
    if snapshot.computed_at > transaction.occurred_at:
        raise FeatureUnavailableError(
            "Online feature snapshot was computed after "
            "the transaction timestamp"
        )

    features = {
        "amount": float(transaction.amount),
        "transactions_5m": float(snapshot.transactions_5m),
    }

    risk_score = model.score(features)

    decision = resolve_decision(
        risk_score=risk_score,
        findings=findings,
        review_threshold=review_threshold,
        decline_threshold=decline_threshold,
    )

    return ScoreResponse(
        transaction_id=transaction.transaction_id,
        risk_score=risk_score,
        decision=decision,
    )
