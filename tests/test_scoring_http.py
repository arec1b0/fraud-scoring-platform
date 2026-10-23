"""HTTP integration tests for real-time fraud scoring."""

import json
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import RedisError

import fraud_scoring.api.scoring_runtime as runtime
from fraud_scoring.api.app import app, get_scoring_handler
from fraud_scoring.api.contracts import Decision, ScoreResponse
from fraud_scoring.features.online import FeatureUnavailableError


def transaction_payload() -> dict[str, object]:
    """Build a valid transaction dated shortly before evaluation."""
    occurred_at = datetime.now(timezone.utc) - timedelta(
        seconds=5
    )
    return {
        "transaction_id": "txn-http-001",
        "customer_id": "customer-001",
        "merchant_id": "merchant-001",
        "amount": "42.50",
        "currency": "CHF",
        "occurred_at": occurred_at.isoformat(),
    }


@pytest.fixture(autouse=True)
def reset_scoring_dependency() -> Iterator[None]:
    """Isolate cached scoring configuration and dependency overrides."""
    get_scoring_handler.cache_clear()

    try:
        yield
    finally:
        app.dependency_overrides.pop(get_scoring_handler, None)
        get_scoring_handler.cache_clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def handler() -> Mock:
    mock = Mock()
    mock.return_value = ScoreResponse(
        transaction_id="txn-http-001",
        risk_score=0.75,
        decision=Decision.REVIEW,
    )

    app.dependency_overrides[get_scoring_handler] = (
        lambda: mock
    )
    return mock


@pytest.fixture
def configured_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Configure a deterministic, test-only fraud policy."""
    values = {
        "FRAUD_REDIS_URL": "redis://127.0.0.1:6379/0",
        "FRAUD_MODEL_PATH": str(tmp_path / "model.txt"),
        "FRAUD_REVIEW_LIMITS_JSON": '{"CHF":"100.00"}',
        "FRAUD_BLOCKED_MERCHANT_IDS_JSON": "[]",
        "FRAUD_REVIEW_THRESHOLD": "0.50",
        "FRAUD_DECLINE_THRESHOLD": "0.90",
        "FRAUD_MAX_FEATURE_AGE_SECONDS": "30",
    }

    for name, value in values.items():
        monkeypatch.setenv(name, value)


def configure_mock_backends(
    monkeypatch: pytest.MonkeyPatch,
    *,
    feature_available: bool = True,
) -> tuple[Mock, Mock, Mock]:
    """Replace infrastructure while retaining real pipeline execution."""
    model = Mock()
    model.feature_names = ("amount", "transactions_5m")
    model.score.return_value = 0.75

    redis_client = Mock()

    if feature_available:
        computed_at = datetime.now(
            timezone.utc
        ) - timedelta(seconds=10)

        redis_client.get.return_value = json.dumps(
            {
                "transactions_5m": 3,
                "computed_at": computed_at.isoformat(),
            }
        )
    else:
        redis_client.get.return_value = None

    model_factory = Mock(return_value=model)
    redis_factory = Mock(return_value=redis_client)

    monkeypatch.setattr(
        runtime,
        "LightGBMScorer",
        model_factory,
    )
    monkeypatch.setattr(
        runtime.Redis,
        "from_url",
        redis_factory,
    )

    return model, model_factory, redis_factory


def test_successful_http_scoring(
    client: TestClient,
    handler: Mock,
) -> None:
    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "transaction_id": "txn-http-001",
        "risk_score": 0.75,
        "decision": "REVIEW",
    }
    handler.assert_called_once()


@pytest.mark.parametrize(
    "reason",
    [
        "Customer velocity snapshot is missing",
        "Customer velocity snapshot is stale",
        "Online feature read failed",
    ],
)
def test_feature_failures_return_503(
    client: TestClient,
    handler: Mock,
    reason: str,
) -> None:
    handler.side_effect = FeatureUnavailableError(reason)

    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == (
        "FEATURE_UNAVAILABLE"
    )


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("Inference failed"),
        ValueError("Feature schema mismatch"),
        RedisError("Redis unavailable"),
    ],
)
def test_scoring_failures_return_503(
    client: TestClient,
    handler: Mock,
    failure: Exception,
) -> None:
    handler.side_effect = failure

    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == (
        "SCORING_UNAVAILABLE"
    )


def test_future_transaction_returns_422(
    client: TestClient,
    handler: Mock,
) -> None:
    payload = transaction_payload()
    payload["occurred_at"] = "2999-01-01T00:00:00Z"

    response = client.post("/score", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == (
        "INVALID_TRANSACTION_TIME"
    )
    handler.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"amount": "0"},
        {"currency": "chf"},
        {"occurred_at": "2026-10-23T08:00:00"},
    ],
)
def test_invalid_payload_returns_422(
    client: TestClient,
    handler: Mock,
    changes: dict[str, object],
) -> None:
    payload = {
        **transaction_payload(),
        **changes,
    }

    response = client.post("/score", json=payload)

    assert response.status_code == 422
    handler.assert_not_called()


def test_missing_configuration_returns_503(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FRAUD_REDIS_URL", raising=False)

    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == (
        "SCORING_UNAVAILABLE"
    )


def test_invalid_thresholds_return_503(
    client: TestClient,
    configured_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRAUD_REVIEW_THRESHOLD", "0.95")

    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 503


def test_invalid_policy_json_returns_503(
    client: TestClient,
    configured_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "FRAUD_REVIEW_LIMITS_JSON",
        "invalid-json",
    )

    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 503


def test_incompatible_model_features_return_503(
    client: TestClient,
    configured_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Mock()
    model.feature_names = ("unexpected_feature",)

    monkeypatch.setattr(
        runtime,
        "LightGBMScorer",
        Mock(return_value=model),
    )

    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == (
        "SCORING_UNAVAILABLE"
    )


def test_missing_model_artifact_returns_503(
    client: TestClient,
    configured_environment: None,
) -> None:
    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == (
        "SCORING_UNAVAILABLE"
    )


def test_http_executes_real_scoring_pipeline(
    client: TestClient,
    configured_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, _ = configure_mock_backends(monkeypatch)

    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 200
    assert response.json()["decision"] == "REVIEW"
    assert response.json()["risk_score"] == 0.75

    model.score.assert_called_once_with(
        {
            "amount": 42.5,
            "transactions_5m": 3.0,
        }
    )


def test_model_and_redis_are_initialized_once(
    client: TestClient,
    configured_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, model_factory, redis_factory = (
        configure_mock_backends(monkeypatch)
    )

    payload = transaction_payload()

    first = client.post("/score", json=payload)
    second = client.post("/score", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200

    model_factory.assert_called_once()
    redis_factory.assert_called_once()


def test_missing_real_online_feature_returns_503(
    client: TestClient,
    configured_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, _ = configure_mock_backends(
        monkeypatch,
        feature_available=False,
    )

    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == (
        "FEATURE_UNAVAILABLE"
    )
    model.score.assert_not_called()
