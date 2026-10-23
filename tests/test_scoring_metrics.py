"""HTTP tests for Prometheus scoring instrumentation."""

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from fraud_scoring.api.app import app, get_scoring_handler
from fraud_scoring.api.contracts import Decision, ScoreResponse
from fraud_scoring.features.online import FeatureUnavailableError
from fraud_scoring.monitoring.metrics import REGISTRY


def transaction_payload() -> dict[str, object]:
    """Create a valid transaction without relying on a fixed date."""
    timestamp = datetime.now(timezone.utc) - timedelta(
        seconds=5
    )

    return {
        "transaction_id": "txn-metrics-001",
        "customer_id": "customer-001",
        "merchant_id": "merchant-001",
        "amount": "42.50",
        "currency": "CHF",
        "occurred_at": timestamp.isoformat(),
    }


def metric_value(
    name: str,
    **labels: str,
) -> float:
    """Return the current value of a Prometheus metric sample."""
    value = REGISTRY.get_sample_value(
        name,
        labels=labels,
    )
    return float(value) if value is not None else 0.0


@pytest.fixture(autouse=True)
def reset_dependencies() -> Iterator[None]:
    """Keep application dependency overrides isolated between tests."""
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
        transaction_id="txn-metrics-001",
        risk_score=0.75,
        decision=Decision.REVIEW,
    )

    app.dependency_overrides[get_scoring_handler] = (
        lambda: mock
    )

    return mock


def test_metrics_endpoint_exposes_prometheus_format(
    client: TestClient,
) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "text/plain"
    )

    for metric_name in (
        "fraud_scoring_http_requests_total",
        "fraud_scoring_http_duration_seconds",
        "fraud_scoring_decisions_total",
        "fraud_scoring_risk_score",
    ):
        assert metric_name in response.text


def test_successful_scoring_records_request_and_decision(
    client: TestClient,
    handler: Mock,
) -> None:
    before_requests = metric_value(
        "fraud_scoring_http_requests_total",
        status_code="200",
    )
    before_duration = metric_value(
        "fraud_scoring_http_duration_seconds_count",
        status_code="200",
    )
    before_decisions = metric_value(
        "fraud_scoring_decisions_total",
        decision="REVIEW",
    )
    before_scores = metric_value(
        "fraud_scoring_risk_score_count",
    )

    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 200

    assert metric_value(
        "fraud_scoring_http_requests_total",
        status_code="200",
    ) == before_requests + 1

    assert metric_value(
        "fraud_scoring_http_duration_seconds_count",
        status_code="200",
    ) == before_duration + 1

    assert metric_value(
        "fraud_scoring_decisions_total",
        decision="REVIEW",
    ) == before_decisions + 1

    assert metric_value(
        "fraud_scoring_risk_score_count",
    ) == before_scores + 1


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (
            FeatureUnavailableError("Missing online feature"),
            "FEATURE_UNAVAILABLE",
        ),
        (
            RuntimeError("Model inference failed"),
            "SCORING_UNAVAILABLE",
        ),
    ],
)
def test_failed_scoring_records_http_but_not_decision(
    client: TestClient,
    handler: Mock,
    failure: Exception,
    expected_code: str,
) -> None:
    handler.side_effect = failure

    before_http = metric_value(
        "fraud_scoring_http_requests_total",
        status_code="503",
    )
    before_scores = metric_value(
        "fraud_scoring_risk_score_count",
    )

    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == expected_code

    assert metric_value(
        "fraud_scoring_http_requests_total",
        status_code="503",
    ) == before_http + 1

    assert metric_value(
        "fraud_scoring_risk_score_count",
    ) == before_scores


def test_validation_failure_records_422(
    client: TestClient,
    handler: Mock,
) -> None:
    before = metric_value(
        "fraud_scoring_http_requests_total",
        status_code="422",
    )

    payload = transaction_payload()
    payload["amount"] = "-10.00"

    response = client.post("/score", json=payload)

    assert response.status_code == 422

    assert metric_value(
        "fraud_scoring_http_requests_total",
        status_code="422",
    ) == before + 1

    handler.assert_not_called()


def test_missing_configuration_records_503(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FRAUD_REDIS_URL", raising=False)

    before = metric_value(
        "fraud_scoring_http_requests_total",
        status_code="503",
    )

    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == (
        "SCORING_UNAVAILABLE"
    )

    assert metric_value(
        "fraud_scoring_http_requests_total",
        status_code="503",
    ) == before + 1


def test_unhandled_exception_records_500(
    handler: Mock,
) -> None:
    handler.side_effect = KeyError("Unexpected internal failure")

    before = metric_value(
        "fraud_scoring_http_requests_total",
        status_code="500",
    )

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/score",
            json=transaction_payload(),
        )

    assert response.status_code == 500

    assert metric_value(
        "fraud_scoring_http_requests_total",
        status_code="500",
    ) == before + 1


def test_health_and_metrics_are_not_counted_as_scoring(
    client: TestClient,
) -> None:
    before = metric_value(
        "fraud_scoring_http_requests_total",
        status_code="200",
    )

    health = client.get("/health")
    scrape = client.get("/metrics")

    assert health.status_code == 200
    assert scrape.status_code == 200

    assert metric_value(
        "fraud_scoring_http_requests_total",
        status_code="200",
    ) == before


def test_metrics_do_not_expose_transaction_identifiers(
    client: TestClient,
    handler: Mock,
) -> None:
    payload = transaction_payload()
    payload["transaction_id"] = "sensitive-txn-unique-8271"

    handler.return_value = ScoreResponse(
        transaction_id="sensitive-txn-unique-8271",
        risk_score=0.75,
        decision=Decision.REVIEW,
    )

    response = client.post("/score", json=payload)

    assert response.status_code == 200

    metrics = client.get("/metrics")

    assert "sensitive-txn-unique-8271" not in metrics.text
    assert "customer-001" not in metrics.text
    assert "merchant-001" not in metrics.text


@pytest.mark.parametrize(
    "risk_score",
    [0.0, 1.0],
)
def test_score_histogram_accepts_endpoint_values(
    client: TestClient,
    handler: Mock,
    risk_score: float,
) -> None:
    decision = (
        Decision.APPROVE
        if risk_score == 0.0
        else Decision.DECLINE
    )

    handler.return_value = ScoreResponse(
        transaction_id="txn-metrics-001",
        risk_score=risk_score,
        decision=decision,
    )

    before = metric_value(
        "fraud_scoring_risk_score_count",
    )

    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 200
    assert metric_value(
        "fraud_scoring_risk_score_count",
    ) == before + 1


def test_latency_histogram_contains_50ms_boundary(
    client: TestClient,
    handler: Mock,
) -> None:
    response = client.post(
        "/score",
        json=transaction_payload(),
    )

    assert response.status_code == 200

    bucket = REGISTRY.get_sample_value(
        "fraud_scoring_http_duration_seconds_bucket",
        labels={
            "status_code": "200",
            "le": "0.05",
        },
    )

    assert bucket is not None
