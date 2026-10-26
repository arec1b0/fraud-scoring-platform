"""Verify the live Docker Compose scoring stack.

Requires the API, Redis, and Prometheus services to be running.
Uses only synthetic transactions and the local demonstration model.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

from fraud_scoring.scoring.decision import resolve_decision
from fraud_scoring.scoring.ml import LightGBMScorer


SEED_SCRIPT = """
import os
import sys
from datetime import datetime, timedelta, timezone
from redis import Redis
from fraud_scoring.features.online import OnlineVelocityStore, VelocitySnapshot

customer_id = sys.argv[1]
client = Redis.from_url(
    os.environ["FRAUD_REDIS_URL"],
    decode_responses=True,
)
store = OnlineVelocityStore(client)

store.put(
    customer_id,
    VelocitySnapshot(
        transactions_5m=3,
        computed_at=datetime.now(timezone.utc) - timedelta(seconds=3),
    ),
    ttl_seconds=240,
)
print("Redis feature seeded")
"""


def run_compose(*arguments: str) -> str:
    """Execute a Docker Compose command and reject failures."""
    command = ["docker", "compose", *arguments]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Compose command failed: {' '.join(command[:4])}\n"
            f"{result.stderr.strip()}"
        )

    return result.stdout.strip()


def request_http(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, bytes]:
    """Send an HTTP request, retaining non-2xx response bodies."""
    data = (
        json.dumps(payload).encode("utf-8")
        if payload is not None
        else None
    )

    request = Request(
        url,
        data=data,
        headers=(
            {"Content-Type": "application/json"}
            if data is not None
            else {}
        ),
        method=method,
    )

    try:
        with urlopen(request, timeout=5.0) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def decode_object(content: bytes) -> dict[str, Any]:
    """Decode an HTTP response as a JSON object."""
    value = json.loads(content)

    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")

    return value


def validate_score_response(
    response: object,
    *,
    transaction_id: str,
    expected_score: float,
    expected_decision: str,
) -> None:
    """Verify identity, real inference score, and decision."""
    if not isinstance(response, dict):
        raise ValueError("Scoring response must be an object")

    if response.get("transaction_id") != transaction_id:
        raise ValueError("Transaction ID mismatch")

    score = response.get("risk_score")

    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
    ):
        raise ValueError("Invalid risk score")

    if not math.isclose(
        score,
        expected_score,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("HTTP score differs from native inference")

    if response.get("decision") != expected_decision:
        raise ValueError("Unexpected fraud decision")


def target_is_up(response: object) -> bool:
    """Inspect a Prometheus instant-query result for an UP target."""
    if not isinstance(response, dict):
        return False

    if response.get("status") != "success":
        return False

    data = response.get("data")

    if not isinstance(data, dict):
        return False

    results = data.get("result")

    if not isinstance(results, list):
        return False

    for item in results:
        if not isinstance(item, dict):
            continue

        value = item.get("value")

        if not isinstance(value, list) or len(value) != 2:
            continue

        try:
            if float(value[1]) == 1.0:
                return True
        except (TypeError, ValueError):
            continue

    return False


def wait_for_prometheus(
    prometheus_url: str,
    *,
    timeout_seconds: float = 90.0,
) -> None:
    """Wait for Prometheus to observe an UP scoring target."""
    query = urlencode(
        {"query": 'up{job="fraud-scoring-api"}'}
    )
    url = (
        f"{prometheus_url.rstrip('/')}/api/v1/query?{query}"
    )

    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        try:
            status_code, body = request_http("GET", url)

            if (
                status_code == 200
                and target_is_up(decode_object(body))
            ):
                return
        except (URLError, OSError, ValueError):
            pass

        time.sleep(2.0)

    raise RuntimeError("Prometheus scoring target did not become UP")


def verify_live_stack(
    *,
    api_url: str,
    prometheus_url: str,
    model_path: Path,
) -> dict[str, object]:
    """Exercise actual Redis, LightGBM, HTTP, and Prometheus."""
    api_url = api_url.rstrip("/")

    scorer = LightGBMScorer(model_path)

    if tuple(scorer.feature_names) != (
        "amount",
        "transactions_5m",
    ):
        raise ValueError("Demo model has an incompatible feature schema")

    health_status, health_body = request_http(
        "GET",
        f"{api_url}/health",
    )

    if (
        health_status != 200
        or decode_object(health_body).get("status") != "ok"
    ):
        raise RuntimeError("API health verification failed")

    customer_id = "ci-" + uuid4().hex
    redis_key = f"fraud:features:customer:{customer_id}"
    seeded = False

    try:
        run_compose(
            "exec",
            "-T",
            "api",
            "python",
            "-c",
            SEED_SCRIPT,
            customer_id,
        )
        seeded = True

        transaction_id = f"txn-{customer_id}"

        transaction = {
            "transaction_id": transaction_id,
            "customer_id": customer_id,
            "merchant_id": "merchant-demo",
            "amount": "42.50",
            "currency": "CHF",
            "occurred_at": (
                datetime.now(timezone.utc)
                - timedelta(seconds=1)
            ).isoformat(),
        }

        expected_score = scorer.score(
            {
                "amount": 42.50,
                "transactions_5m": 3.0,
            }
        )

        expected_decision = resolve_decision(
            risk_score=expected_score,
            findings=(),
            review_threshold=0.50,
            decline_threshold=0.90,
        )

        status_code, body = request_http(
            "POST",
            f"{api_url}/score",
            transaction,
        )

        if status_code != 200:
            raise RuntimeError(
                f"Live scoring returned HTTP {status_code}: "
                f"{body.decode('utf-8', errors='replace')}"
            )

        validate_score_response(
            decode_object(body),
            transaction_id=transaction_id,
            expected_score=expected_score,
            expected_decision=expected_decision,
        )

        blocked = {
            **transaction,
            "transaction_id": f"{transaction_id}-blocked",
            "merchant_id": "merchant-demo-blocked",
        }

        blocked_status, blocked_body = request_http(
            "POST",
            f"{api_url}/score",
            blocked,
        )

        if blocked_status != 200:
            raise RuntimeError(
                f"Blocked merchant returned HTTP {blocked_status}"
            )

        validate_score_response(
            decode_object(blocked_body),
            transaction_id=blocked["transaction_id"],
            expected_score=expected_score,
            expected_decision="DECLINE",
        )

        missing = {
            **transaction,
            "transaction_id": f"{transaction_id}-missing",
            "customer_id": "missing-" + uuid4().hex,
        }

        missing_status, missing_body = request_http(
            "POST",
            f"{api_url}/score",
            missing,
        )

        if missing_status != 503:
            raise RuntimeError(
                "Missing features did not fail with HTTP 503"
            )

        missing_error = decode_object(missing_body)

        if missing_error.get("detail", {}).get("code") != (
            "FEATURE_UNAVAILABLE"
        ):
            raise RuntimeError(
                "Missing features returned the wrong error code"
            )

        metrics_status, metrics_body = request_http(
            "GET",
            f"{api_url}/metrics",
        )

        if metrics_status != 200:
            raise RuntimeError("Metrics endpoint returned non-200")

        metrics_text = metrics_body.decode("utf-8")

        for sample in (
            'fraud_scoring_http_requests_total{status_code="200"}',
            'fraud_scoring_http_requests_total{status_code="503"}',
            'fraud_scoring_decisions_total{decision="DECLINE"}',
            "fraud_scoring_risk_score_count",
        ):
            if sample not in metrics_text:
                raise RuntimeError(
                    f"Prometheus metric missing: {sample}"
                )

        wait_for_prometheus(prometheus_url)

        return {
            "passed": True,
            "data_origin": "synthetic",
            "model_inference": "verified",
            "redis_feature_lookup": "verified",
            "http_scoring": 200,
            "blocked_merchant_decision": "DECLINE",
            "missing_features_http": 503,
            "prometheus_metrics": "verified",
            "prometheus_scrape": "UP",
            "production_approved": False,
        }
    finally:
        if seeded:
            run_compose(
                "exec",
                "-T",
                "redis",
                "redis-cli",
                "DEL",
                redis_key,
            )


def main() -> int:
    """Run the end-to-end integration gate."""
    parser = argparse.ArgumentParser(
        description="Verify the live FraudScoring Docker stack."
    )

    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--prometheus-url",
        default="http://127.0.0.1:9090",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/demo_fraud_model.txt"),
    )

    args = parser.parse_args()

    try:
        result = verify_live_stack(
            api_url=args.api_url,
            prometheus_url=args.prometheus_url,
            model_path=args.model_path,
        )
    except (
        RuntimeError,
        ValueError,
        OSError,
        URLError,
    ) as exc:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
