"""Tests for the live scoring latency measurement utility."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from fraud_scoring.monitoring import load_test as probe
from fraud_scoring.monitoring.load_test import (
    RequestResult,
    run_load_test,
    summarize_results,
)


class FakeHTTPResponse:
    """Minimal HTTP response used by transport-boundary tests."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_successful_samples_pass_below_budget() -> None:
    samples = [
        RequestResult(10.0, "OK"),
        RequestResult(20.0, "OK"),
        RequestResult(30.0, "OK"),
        RequestResult(40.0, "OK"),
    ]

    report = summarize_results(
        samples,
        elapsed_seconds=0.5,
        p95_budget_ms=50.0,
    )

    assert report["requests"] == 4
    assert report["successful"] == 4
    assert report["failed"] == 0
    assert report["p50_ms"] == 20.0
    assert report["p95_ms"] == 40.0
    assert report["p99_ms"] == 40.0
    assert report["throughput_rps"] == 8.0
    assert report["passed"] is True


def test_exact_latency_budget_fails() -> None:
    report = summarize_results(
        [RequestResult(50.0, "OK")],
        elapsed_seconds=0.1,
        p95_budget_ms=50.0,
    )

    assert report["p95_ms"] == 50.0
    assert report["passed"] is False


def test_http_failure_fails_slo() -> None:
    report = summarize_results(
        [
            RequestResult(10.0, "OK"),
            RequestResult(12.0, "HTTP_503"),
        ],
        elapsed_seconds=0.1,
        p95_budget_ms=50.0,
    )

    assert report["successful"] == 1
    assert report["failed"] == 1
    assert report["outcomes"]["HTTP_503"] == 1
    assert report["passed"] is False


def test_transport_failure_fails_slo() -> None:
    report = summarize_results(
        [
            RequestResult(8.0, "OK"),
            RequestResult(2000.0, "TRANSPORT_ERROR"),
        ],
        elapsed_seconds=2.1,
        p95_budget_ms=50.0,
    )

    assert report["failed"] == 1
    assert report["passed"] is False


def test_all_failed_requests_have_no_percentiles() -> None:
    report = summarize_results(
        [RequestResult(5.0, "HTTP_503")],
        elapsed_seconds=0.1,
        p95_budget_ms=50.0,
    )

    assert report["p50_ms"] is None
    assert report["p95_ms"] is None
    assert report["p99_ms"] is None
    assert report["passed"] is False


def test_http_request_accepts_valid_scoring_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(
        request: Request,
        *,
        timeout: float,
    ) -> FakeHTTPResponse:
        assert request.get_method() == "POST"
        assert timeout == 2.0

        payload = json.loads(request.data)

        assert payload["customer_id"] == "customer-test"
        assert payload["amount"] == "42.50"
        assert payload["currency"] == "CHF"
        assert payload["occurred_at"].endswith("+00:00")

        response = {
            "transaction_id": payload["transaction_id"],
            "risk_score": 0.25,
            "decision": "APPROVE",
        }

        return FakeHTTPResponse(
            json.dumps(response).encode("utf-8")
        )

    monkeypatch.setattr(probe, "urlopen", fake_urlopen)

    result = probe._send_once(
        "http://127.0.0.1:8000/score",
        "customer-test",
        2.0,
    )

    assert result.outcome == "OK"
    assert result.latency_ms >= 0.0


@pytest.mark.parametrize(
    "failure_mode",
    [
        "malformed_json",
        "wrong_transaction",
        "invalid_score",
        "invalid_decision",
    ],
)
def test_invalid_success_response_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    def fake_urlopen(
        request: Request,
        *,
        timeout: float,
    ) -> FakeHTTPResponse:
        payload = json.loads(request.data)

        result = {
            "transaction_id": payload["transaction_id"],
            "risk_score": 0.25,
            "decision": "APPROVE",
        }

        if failure_mode == "malformed_json":
            return FakeHTTPResponse(b"not-json")

        if failure_mode == "wrong_transaction":
            result["transaction_id"] = "wrong"

        if failure_mode == "invalid_score":
            result["risk_score"] = float("nan")

        if failure_mode == "invalid_decision":
            result["decision"] = "UNKNOWN"

        return FakeHTTPResponse(
            json.dumps(result).encode("utf-8")
        )

    monkeypatch.setattr(probe, "urlopen", fake_urlopen)

    result = probe._send_once(
        "http://127.0.0.1:8000/score",
        "customer-test",
        2.0,
    )

    assert result.outcome == "INVALID_RESPONSE"


def test_http_error_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(
        request: Request,
        *,
        timeout: float,
    ) -> FakeHTTPResponse:
        raise HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            {},
            None,
        )

    monkeypatch.setattr(probe, "urlopen", fail)

    result = probe._send_once(
        "http://127.0.0.1:8000/score",
        "customer-test",
        2.0,
    )

    assert result.outcome == "HTTP_503"


def test_network_error_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(
        request: Request,
        *,
        timeout: float,
    ) -> FakeHTTPResponse:
        raise URLError("Connection refused")

    monkeypatch.setattr(probe, "urlopen", fail)

    result = probe._send_once(
        "http://127.0.0.1:8000/score",
        "customer-test",
        2.0,
    )

    assert result.outcome == "TRANSPORT_ERROR"


def test_failed_warmup_aborts_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_send(
        url: str,
        customer_id: str,
        timeout_seconds: float,
    ) -> RequestResult:
        calls.append(1)
        return RequestResult(3.0, "HTTP_503")

    monkeypatch.setattr(probe, "_send_once", fake_send)

    with pytest.raises(
        RuntimeError,
        match="Warm-up request 1 failed",
    ):
        run_load_test(
            url="http://127.0.0.1:8000/score",
            customer_id="customer-test",
            requests=4,
            concurrency=2,
            warmup=2,
        )

    assert len(calls) == 1


def test_warmup_is_excluded_from_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_send(
        url: str,
        customer_id: str,
        timeout_seconds: float,
    ) -> RequestResult:
        calls.append(1)
        return RequestResult(1.0, "OK")

    monkeypatch.setattr(probe, "_send_once", fake_send)

    report = run_load_test(
        url="http://127.0.0.1:8000/score",
        customer_id="customer-test",
        requests=4,
        concurrency=2,
        warmup=2,
    )

    assert len(calls) == 6
    assert report["requests"] == 4
    assert report["successful"] == 4
    assert report["p95_ms"] == 1.0
    assert report["passed"] is True


@pytest.mark.parametrize(
    ("parameter", "invalid_value"),
    [
        ("requests", 0),
        ("requests", True),
        ("concurrency", 0),
        ("concurrency", 65),
        ("warmup", -1),
        ("timeout_seconds", 0),
        ("p95_budget_ms", float("nan")),
        ("customer_id", ""),
        ("url", "ftp://127.0.0.1:8000/score"),
    ],
)
def test_invalid_configuration_is_rejected(
    parameter: str,
    invalid_value: object,
) -> None:
    settings = {
        "url": "http://127.0.0.1:8000/score",
        "customer_id": "customer-test",
        "requests": 4,
        "concurrency": 2,
        "warmup": 0,
        "timeout_seconds": 2.0,
        "p95_budget_ms": 50.0,
    }

    settings[parameter] = invalid_value

    with pytest.raises(ValueError):
        run_load_test(**settings)
