"""Unit tests for the live-stack integration verifier."""

import math

import pytest

from fraud_scoring.monitoring.stack_smoke import (
    target_is_up,
    validate_score_response,
)


def valid_response() -> dict[str, object]:
    return {
        "transaction_id": "txn-test",
        "risk_score": 0.25,
        "decision": "APPROVE",
    }


def test_valid_scoring_response() -> None:
    validate_score_response(
        valid_response(),
        transaction_id="txn-test",
        expected_score=0.25,
        expected_decision="APPROVE",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transaction_id", "wrong-id"),
        ("risk_score", None),
        ("risk_score", math.nan),
        ("risk_score", 1.5),
        ("risk_score", True),
        ("decision", "UNKNOWN"),
    ],
)
def test_invalid_scoring_response(
    field: str,
    value: object,
) -> None:
    response = valid_response()
    response[field] = value

    with pytest.raises(ValueError):
        validate_score_response(
            response,
            transaction_id="txn-test",
            expected_score=0.25,
            expected_decision="APPROVE",
        )


def test_unexpected_model_score_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="differs from native inference",
    ):
        validate_score_response(
            valid_response(),
            transaction_id="txn-test",
            expected_score=0.75,
            expected_decision="APPROVE",
        )


def test_non_object_response_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must be an object",
    ):
        validate_score_response(
            [],
            transaction_id="txn-test",
            expected_score=0.25,
            expected_decision="APPROVE",
        )


def test_prometheus_up_response() -> None:
    result = {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {
                    "metric": {
                        "job": "fraud-scoring-api",
                    },
                    "value": [1234567890, "1"],
                }
            ],
        },
    }

    assert target_is_up(result) is True


def test_prometheus_down_response() -> None:
    result = {
        "status": "success",
        "data": {
            "result": [
                {
                    "value": [1234567890, "0"],
                }
            ],
        },
    }

    assert target_is_up(result) is False


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"status": "error"},
    ],
)
def test_malformed_prometheus_response(
    response: object,
) -> None:
    assert target_is_up(response) is False
