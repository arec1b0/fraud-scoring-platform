"""Health endpoint tests."""

from fastapi.testclient import TestClient

from fraud_scoring.api.app import app


def test_health_returns_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}