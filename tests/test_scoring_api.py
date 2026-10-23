"""HTTP scoring contract tests."""

import pytest
from fastapi.testclient import TestClient

from fraud_scoring.api.app import app


VALID_TRANSACTION = {
    "transaction_id": "txn-001",
    "customer_id": "customer-001",
    "merchant_id": "merchant-001",
    "amount": "42.50",
    "currency": "CHF",
    "occurred_at": "2026-10-23T08:00:00Z",
}


def test_valid_request_does_not_return_a_fabricated_score() -> None:
    with TestClient(app) as client:
        response = client.post("/score", json=VALID_TRANSACTION)

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "SCORING_UNAVAILABLE",
            "message": "A scoring backend is not configured.",
        }
    }


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"amount": "0"},
        {"currency": "chf"},
        {"occurred_at": "2026-10-23T08:00:00"},
    ],
)
def test_invalid_transaction_is_rejected(invalid_field: dict[str, str]) -> None:
    transaction = {**VALID_TRANSACTION, **invalid_field}

    with TestClient(app) as client:
        response = client.post("/score", json=transaction)

    assert response.status_code == 422