"""Measure live HTTP fraud-scoring latency against an explicit SLO.

This is a closed-loop load test for the development scoring endpoint.
It does not generate training data or mock inference infrastructure.
"""

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class RequestResult:
    """One completed HTTP attempt and its measured latency."""

    latency_ms: float
    outcome: str


def _send_once(
    url: str,
    customer_id: str,
    timeout_seconds: float,
) -> RequestResult:
    """Send one transaction and validate the complete HTTP response."""
    transaction_id = f"load-{uuid4().hex}"

    payload = {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "merchant_id": "merchant-demo",
        "amount": "42.50",
        "currency": "CHF",
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = perf_counter()

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = response.status
            body = response.read()
    except HTTPError as exc:
        return RequestResult(
            (perf_counter() - started) * 1000.0,
            f"HTTP_{exc.code}",
        )
    except (URLError, TimeoutError, OSError):
        return RequestResult(
            (perf_counter() - started) * 1000.0,
            "TRANSPORT_ERROR",
        )

    latency_ms = (perf_counter() - started) * 1000.0

    if status_code != 200:
        return RequestResult(latency_ms, f"HTTP_{status_code}")

    try:
        result = json.loads(body)

        if not isinstance(result, dict):
            raise ValueError("Response must be an object")

        score = result.get("risk_score")

        valid = (
            result.get("transaction_id") == transaction_id
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(score)
            and 0.0 <= score <= 1.0
            and result.get("decision")
            in ("APPROVE", "REVIEW", "DECLINE")
        )

        if not valid:
            raise ValueError("Invalid scoring response")
    except (ValueError, TypeError, UnicodeError):
        return RequestResult(latency_ms, "INVALID_RESPONSE")

    return RequestResult(latency_ms, "OK")


def _percentile(
    sorted_values: Sequence[float],
    percentile: float,
) -> float | None:
    """Calculate a nearest-rank percentile from sorted observations."""
    if not sorted_values:
        return None

    rank = math.ceil(
        percentile / 100.0 * len(sorted_values)
    )

    return sorted_values[max(0, rank - 1)]


def summarize_results(
    results: Sequence[RequestResult],
    *,
    elapsed_seconds: float,
    p95_budget_ms: float,
) -> dict[str, object]:
    """Calculate performance measurements and the SLO verdict."""
    if not results:
        raise ValueError("At least one measured request is required")

    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")

    if not math.isfinite(p95_budget_ms) or p95_budget_ms <= 0:
        raise ValueError("p95_budget_ms must be positive")

    outcomes = Counter(result.outcome for result in results)

    successful_latencies = sorted(
        result.latency_ms
        for result in results
        if result.outcome == "OK"
    )

    p50 = _percentile(successful_latencies, 50)
    p95 = _percentile(successful_latencies, 95)
    p99 = _percentile(successful_latencies, 99)

    passed = (
        len(successful_latencies) == len(results)
        and p95 is not None
        and p95 < p95_budget_ms
    )

    def display(value: float | None) -> float | None:
        return None if value is None else round(value, 6)

    return {
        "requests": len(results),
        "successful": len(successful_latencies),
        "failed": len(results) - len(successful_latencies),
        "outcomes": dict(sorted(outcomes.items())),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "throughput_rps": round(
            len(results) / elapsed_seconds,
            6,
        ),
        "p50_ms": display(p50),
        "p95_ms": display(p95),
        "p99_ms": display(p99),
        "p95_budget_ms": p95_budget_ms,
        "passed": passed,
    }


def run_load_test(
    *,
    url: str,
    customer_id: str,
    requests: int = 200,
    concurrency: int = 8,
    warmup: int = 20,
    timeout_seconds: float = 2.0,
    p95_budget_ms: float = 50.0,
) -> dict[str, object]:
    """Warm the live endpoint, then measure concurrent HTTP requests."""
    parsed = urlsplit(url)

    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.path != "/score"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("url must identify an HTTP /score endpoint")

    if not 1 <= len(customer_id) <= 64:
        raise ValueError("customer_id must contain 1 to 64 characters")

    if type(requests) is not int or not 1 <= requests <= 10000:
        raise ValueError("requests must be an integer in [1, 10000]")

    if type(concurrency) is not int or not 1 <= concurrency <= 64:
        raise ValueError("concurrency must be an integer in [1, 64]")

    if type(warmup) is not int or not 0 <= warmup <= 10000:
        raise ValueError("warmup must be an integer in [0, 10000]")

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    if not math.isfinite(p95_budget_ms) or p95_budget_ms <= 0:
        raise ValueError("p95_budget_ms must be positive")

    # Warm-up traffic is deliberately excluded from measurements.
    for index in range(warmup):
        result = _send_once(
            url,
            customer_id,
            timeout_seconds,
        )

        if result.outcome != "OK":
            raise RuntimeError(
                f"Warm-up request {index + 1} failed: "
                f"{result.outcome}"
            )

    started = perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _send_once,
                url,
                customer_id,
                timeout_seconds,
            )
            for _ in range(requests)
        ]

        results = [
            future.result()
            for future in as_completed(futures)
        ]

    elapsed_seconds = perf_counter() - started

    return summarize_results(
        results,
        elapsed_seconds=elapsed_seconds,
        p95_budget_ms=p95_budget_ms,
    )


def main() -> int:
    """Execute a live SLO check and emit a machine-readable JSON report."""
    parser = argparse.ArgumentParser(
        description="Measure the live FraudScoring HTTP p95 latency SLO."
    )

    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/score",
    )
    parser.add_argument(
        "--customer-id",
        required=True,
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--p95-budget-ms",
        type=float,
        default=50.0,
    )

    args = parser.parse_args()

    try:
        report = run_load_test(
            url=args.url,
            customer_id=args.customer_id,
            requests=args.requests,
            concurrency=args.concurrency,
            warmup=args.warmup,
            timeout_seconds=args.timeout_seconds,
            p95_budget_ms=args.p95_budget_ms,
        )
    except (ValueError, RuntimeError) as exc:
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

    print(json.dumps(report, indent=2))

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
