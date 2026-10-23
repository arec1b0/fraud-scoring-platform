"""Tests for the real-time fraud scoring orchestration."""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import Mock

import pytest

from fraud_scoring.api.contracts import (
    Decision,
    ScoreRequest,
    ScoreResponse,
)
from fraud_scoring.api.scoring_service import (
    run_scoring_pipeline,
)
from fraud_scoring.features.online import (
    FeatureUnavailableError,
    OnlineVelocityStore,
    VelocitySnapshot,
)
from fraud_scoring.scoring.ml import LightGBMScorer


NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
TRANSACTION_TIME = NOW - timedelta(seconds=5)


@pytest.fixture
def transaction() -> ScoreRequest:
    return ScoreRequest(
        transaction_id="txn-001",
        customer_id="customer-001",
        merchant_id="merchant-001",
        amount=Decimal("42.50"),
        currency="CHF",
        occurred_at=TRANSACTION_TIME,
    )


@pytest.fixture
def store() -> Mock:
    instance = Mock(spec=OnlineVelocityStore)
    instance.get_fresh.return_value = VelocitySnapshot(
        transactions_5m=3,
        computed_at=NOW - timedelta(seconds=10),
    )
    return instance


@pytest.fixture
def model() -> Mock:
    instance = Mock(spec=LightGBMScorer)
    instance.score.return_value = 0.20
    return instance


def execute(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
    *,
    evaluated_at: datetime = NOW,
    blocked_merchants: frozenset[str] = frozenset(
        {"merchant-blocked"}
    ),
    review_limits: Mapping[str, Decimal] | None = None,
    review_threshold: float = 0.50,
    decline_threshold: float = 0.90,
) -> ScoreResponse:
    """Run the pipeline using an explicit test-only policy."""
    if review_limits is None:
        review_limits = {"CHF": Decimal("100.00")}

    return run_scoring_pipeline(
        transaction,
        feature_store=store,
        model=model,
        evaluated_at=evaluated_at,
        max_feature_age_seconds=30,
        blocked_merchant_ids=blocked_merchants,
        review_limits=review_limits,
        review_threshold=review_threshold,
        decline_threshold=decline_threshold,
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.20, Decision.APPROVE),
        (0.50, Decision.REVIEW),
        (0.90, Decision.DECLINE),
    ],
)
def test_model_score_produces_expected_decision(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
    score: float,
    expected: Decision,
) -> None:
    model.score.return_value = score

    result = execute(transaction, store, model)

    assert result.transaction_id == "txn-001"
    assert result.risk_score == score
    assert result.decision == expected

    store.get_fresh.assert_called_once_with(
        "customer-001",
        now=NOW,
        max_age_seconds=30,
    )

    model.score.assert_called_once_with(
        {
            "amount": 42.5,
            "transactions_5m": 3.0,
        }
    )


def test_amount_rule_requires_review(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
) -> None:
    high_amount = transaction.model_copy(
        update={"amount": Decimal("120.00")}
    )

    result = execute(high_amount, store, model)

    assert result.risk_score == 0.20
    assert result.decision == Decision.REVIEW


def test_blocked_merchant_requires_decline(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
) -> None:
    blocked = transaction.model_copy(
        update={"merchant_id": "merchant-blocked"}
    )

    result = execute(blocked, store, model)

    assert result.decision == Decision.DECLINE


def test_decline_overrides_amount_review(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
) -> None:
    blocked_and_expensive = transaction.model_copy(
        update={
            "merchant_id": "merchant-blocked",
            "amount": Decimal("120.00"),
        }
    )

    result = execute(blocked_and_expensive, store, model)

    assert result.decision == Decision.DECLINE


def test_snapshot_at_transaction_time_is_accepted(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
) -> None:
    store.get_fresh.return_value = VelocitySnapshot(
        transactions_5m=7,
        computed_at=TRANSACTION_TIME,
    )

    result = execute(transaction, store, model)

    assert result.decision == Decision.APPROVE
    model.score.assert_called_once_with(
        {
            "amount": 42.5,
            "transactions_5m": 7.0,
        }
    )


def test_future_feature_snapshot_is_rejected(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
) -> None:
    store.get_fresh.return_value = VelocitySnapshot(
        transactions_5m=9,
        computed_at=NOW - timedelta(seconds=3),
    )

    with pytest.raises(
        FeatureUnavailableError,
        match="after the transaction",
    ):
        execute(transaction, store, model)

    model.score.assert_not_called()


def test_future_transaction_is_rejected(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
) -> None:
    future_transaction = transaction.model_copy(
        update={"occurred_at": NOW + timedelta(seconds=1)}
    )

    with pytest.raises(
        ValueError,
        match="Transaction timestamp is in the future",
    ):
        execute(future_transaction, store, model)

    store.get_fresh.assert_not_called()
    model.score.assert_not_called()


def test_naive_evaluation_timestamp_is_rejected(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
) -> None:
    with pytest.raises(
        ValueError,
        match="evaluated_at must be timezone-aware",
    ):
        execute(
            transaction,
            store,
            model,
            evaluated_at=NOW.replace(tzinfo=None),
        )

    store.get_fresh.assert_not_called()


@pytest.mark.parametrize(
    "reason",
    [
        "Customer velocity snapshot is stale",
        "Customer velocity snapshot is missing",
        "Online feature read failed",
    ],
)
def test_feature_unavailability_fails_closed(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
    reason: str,
) -> None:
    store.get_fresh.side_effect = FeatureUnavailableError(
        reason
    )

    with pytest.raises(
        FeatureUnavailableError,
        match=reason,
    ):
        execute(transaction, store, model)

    model.score.assert_not_called()


def test_model_failure_fails_closed(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
) -> None:
    model.score.side_effect = RuntimeError(
        "Model inference failed"
    )

    with pytest.raises(
        RuntimeError,
        match="Model inference failed",
    ):
        execute(transaction, store, model)


def test_model_feature_schema_error_is_propagated(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
) -> None:
    model.score.side_effect = ValueError(
        "Feature schema mismatch"
    )

    with pytest.raises(
        ValueError,
        match="Feature schema mismatch",
    ):
        execute(transaction, store, model)


def test_invalid_model_score_is_rejected(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
) -> None:
    model.score.return_value = float("nan")

    with pytest.raises(
        ValueError,
        match="risk_score",
    ):
        execute(transaction, store, model)


def test_missing_currency_policy_is_rejected(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
) -> None:
    euro_transaction = transaction.model_copy(
        update={"currency": "EUR"}
    )

    with pytest.raises(
        ValueError,
        match="No review limit configured",
    ):
        execute(euro_transaction, store, model)

    store.get_fresh.assert_not_called()
    model.score.assert_not_called()


def test_invalid_decision_thresholds_are_rejected(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
) -> None:
    with pytest.raises(
        ValueError,
        match="review_threshold must be below",
    ):
        execute(
            transaction,
            store,
            model,
            review_threshold=0.90,
            decline_threshold=0.50,
        )


def test_blocked_merchant_cannot_hide_model_failure(
    transaction: ScoreRequest,
    store: Mock,
    model: Mock,
) -> None:
    blocked = transaction.model_copy(
        update={"merchant_id": "merchant-blocked"}
    )
    model.score.side_effect = RuntimeError(
        "Model inference failed"
    )

    with pytest.raises(
        RuntimeError,
        match="Model inference failed",
    ):
        execute(blocked, store, model)

    model.score.assert_called_once()
