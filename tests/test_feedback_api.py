"""HTTP tests for delayed fraud feedback ingestion."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from fraud_scoring.api.app import (
    app,
    get_feedback_store,
)
from fraud_scoring.feedback import SQLiteFeedbackStore


VALID_FEEDBACK = {
    "transaction_id": "txn-001",
    "outcome": "FRAUD",
    "source": "CHARGEBACK",
    "transaction_at": "2026-10-23T08:00:00Z",
    "labeled_at": "2026-10-25T08:00:00Z",
}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """Provide an HTTP client with isolated persistent storage."""
    store = SQLiteFeedbackStore(
        tmp_path / "feedback.sqlite3"
    )

    app.dependency_overrides[get_feedback_store] = lambda: store

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_feedback_store, None)


def test_new_feedback_returns_created(
    client: TestClient,
) -> None:
    response = client.post(
        "/feedback",
        json=VALID_FEEDBACK,
    )

    assert response.status_code == 201
    assert response.json() == {
        "transaction_id": "txn-001",
        "created": True,
    }


def test_exact_replay_returns_ok(
    client: TestClient,
) -> None:
    first = client.post(
        "/feedback",
        json=VALID_FEEDBACK,
    )
    second = client.post(
        "/feedback",
        json=VALID_FEEDBACK,
    )

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json() == {
        "transaction_id": "txn-001",
        "created": False,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("outcome", "LEGITIMATE"),
        ("source", "MANUAL_REVIEW"),
        ("labeled_at", "2026-10-26T08:00:00Z"),
    ],
)
def test_conflicting_feedback_returns_409(
    client: TestClient,
    field: str,
    value: str,
) -> None:
    original = client.post(
        "/feedback",
        json=VALID_FEEDBACK,
    )

    conflicting = {
        **VALID_FEEDBACK,
        field: value,
    }

    rejected = client.post(
        "/feedback",
        json=conflicting,
    )

    replay = client.post(
        "/feedback",
        json=VALID_FEEDBACK,
    )

    assert original.status_code == 201
    assert rejected.status_code == 409
    assert rejected.json()["detail"]["code"] == (
        "FEEDBACK_CONFLICT"
    )

    # The original record must remain unchanged.
    assert replay.status_code == 200
    assert replay.json()["created"] is False


def test_equivalent_timezone_offsets_are_idempotent(
    client: TestClient,
) -> None:
    equivalent = {
        **VALID_FEEDBACK,
        "transaction_at": "2026-10-23T10:00:00+02:00",
        "labeled_at": "2026-10-25T10:00:00+02:00",
    }

    first = client.post(
        "/feedback",
        json=VALID_FEEDBACK,
    )
    replay = client.post(
        "/feedback",
        json=equivalent,
    )

    assert first.status_code == 201
    assert replay.status_code == 200
    assert replay.json()["created"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"transaction_id": ""},
        {"transaction_id": 123},
        {"outcome": "UNKNOWN"},
        {"source": "AUTOMATED"},
        {"transaction_at": "2026-10-23T08:00:00"},
        {"labeled_at": "2026-10-25T08:00:00"},
        {"labeled_at": "2026-10-22T08:00:00Z"},
        {"unexpected_field": "not-allowed"},
    ],
)
def test_invalid_feedback_returns_422(
    client: TestClient,
    changes: dict[str, object],
) -> None:
    payload = {
        **VALID_FEEDBACK,
        **changes,
    }

    response = client.post(
        "/feedback",
        json=payload,
    )

    assert response.status_code == 422


def test_missing_configuration_returns_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "FRAUD_FEEDBACK_DB_PATH",
        raising=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/feedback",
            json=VALID_FEEDBACK,
        )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == (
        "FEEDBACK_STORAGE_UNAVAILABLE"
    )


def test_invalid_database_path_returns_503(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = (
        tmp_path / "missing-directory" / "feedback.sqlite3"
    )

    monkeypatch.setenv(
        "FRAUD_FEEDBACK_DB_PATH",
        str(database_path),
    )

    with TestClient(app) as client:
        response = client.post(
            "/feedback",
            json=VALID_FEEDBACK,
        )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == (
        "FEEDBACK_STORAGE_UNAVAILABLE"
    )


def test_database_write_failure_returns_503() -> None:
    store = Mock(spec=SQLiteFeedbackStore)
    store.record.side_effect = sqlite3.OperationalError(
        "Simulated database failure"
    )

    app.dependency_overrides[get_feedback_store] = lambda: store

    try:
        with TestClient(app) as client:
            response = client.post(
                "/feedback",
                json=VALID_FEEDBACK,
            )
    finally:
        app.dependency_overrides.pop(get_feedback_store, None)

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == (
        "FEEDBACK_STORAGE_UNAVAILABLE"
    )


def test_existing_api_endpoints_remain_unchanged(
    client: TestClient,
) -> None:
    health = client.get("/health")

    scoring = client.post(
        "/score",
        json={
            "transaction_id": "txn-001",
            "customer_id": "customer-001",
            "merchant_id": "merchant-001",
            "amount": "42.50",
            "currency": "CHF",
            "occurred_at": "2026-10-23T08:00:00Z",
        },
    )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    assert scoring.status_code == 503
    assert scoring.json()["detail"]["code"] == (
        "SCORING_UNAVAILABLE"
    )
