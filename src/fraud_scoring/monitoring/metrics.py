"""Prometheus instrumentation for real-time fraud scoring."""

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)


REGISTRY = CollectorRegistry()

HTTP_REQUESTS = Counter(
    "fraud_scoring_http_requests_total",
    "Number of scoring HTTP requests by response status.",
    labelnames=("status_code",),
    registry=REGISTRY,
)

HTTP_DURATION = Histogram(
    "fraud_scoring_http_duration_seconds",
    "Duration of scoring HTTP requests in seconds.",
    labelnames=("status_code",),
    buckets=(
        0.005,
        0.010,
        0.025,
        0.050,
        0.075,
        0.100,
        0.250,
        0.500,
        1.000,
        2.500,
        5.000,
    ),
    registry=REGISTRY,
)

DECISIONS = Counter(
    "fraud_scoring_decisions_total",
    "Number of completed fraud decisions.",
    labelnames=("decision",),
    registry=REGISTRY,
)

RISK_SCORES = Histogram(
    "fraud_scoring_risk_score",
    "Distribution of model scores for completed decisions.",
    buckets=tuple(index / 10 for index in range(11)),
    registry=REGISTRY,
)


def observe_scoring_http(
    *,
    status_code: int,
    duration_seconds: float,
) -> None:
    """Record the HTTP outcome and elapsed request duration."""
    label = str(status_code)

    HTTP_REQUESTS.labels(status_code=label).inc()
    HTTP_DURATION.labels(status_code=label).observe(
        duration_seconds
    )


def observe_scoring_result(
    *,
    decision: str,
    risk_score: float,
) -> None:
    """Record a successfully completed scoring decision."""
    DECISIONS.labels(decision=decision).inc()
    RISK_SCORES.observe(risk_score)


def render_metrics() -> bytes:
    """Render the process-local registry in Prometheus text format."""
    return generate_latest(REGISTRY)
