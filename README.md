# FraudScoring Platform

A Python-based platform for real-time fraud risk scoring.

## Requirements

- Python 3.12
- uv
- Git

Live scoring additionally requires Redis and a trained LightGBM
binary-classification model.

## Installation

From the repository root:

```powershell
uv sync --locked
```

## Local Configuration

Create the feedback database directory:

```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
```

Configure local feedback storage:

```powershell
$env:FRAUD_FEEDBACK_DB_PATH = Join-Path (Get-Location).Path "data\feedback.sqlite3"
```

Configure scoring:

```powershell
$env:FRAUD_REDIS_URL = "redis://127.0.0.1:6379/0"
$env:FRAUD_MODEL_PATH = Join-Path (Get-Location).Path "models\fraud_model.txt"
$env:FRAUD_REVIEW_LIMITS_JSON = '{"CHF":"100.00"}'
$env:FRAUD_BLOCKED_MERCHANT_IDS_JSON = '[]'
$env:FRAUD_REVIEW_THRESHOLD = "0.50"
$env:FRAUD_DECLINE_THRESHOLD = "0.90"
$env:FRAUD_MAX_FEATURE_AGE_SECONDS = "30"
```

The limits and thresholds above are illustrative development values.
They have not been validated for production fraud decisions.

The LightGBM model must use a binary objective and contain exactly:

- `amount`
- `transactions_5m`

Required Redis features must already exist and satisfy the configured
freshness limit.

## Run the API

```powershell
uv run --locked uvicorn fraud_scoring.api.app:app --host 127.0.0.1 --port 8000
```

Available endpoints:

| Endpoint | Responsibility |
|---|---|
| `GET /health` | Process health |
| `POST /score` | Real-time fraud scoring |
| `POST /feedback` | Delayed fraud labels |
| `GET /metrics` | Prometheus metrics |
| `GET /docs` | Interactive API documentation |

## Real-Time Scoring

Example `POST /score` request:

```json
{
  "transaction_id": "txn-001",
  "customer_id": "customer-001",
  "merchant_id": "merchant-001",
  "amount": "42.50",
  "currency": "CHF",
  "occurred_at": "2026-10-23T08:00:00Z"
}
```

Example successful response:

```json
{
  "transaction_id": "txn-001",
  "risk_score": 0.75,
  "decision": "REVIEW"
}
```

The response illustrates the contract. The actual risk score is
produced by the configured model.

### Execution

1. Validate transaction data.
2. Evaluate deterministic fraud rules.
3. Fetch fresh customer velocity features from Redis.
4. Reject features calculated after the transaction timestamp.
5. Run LightGBM inference.
6. Apply the deterministic decision policy.
7. Return the model score and final decision.

Missing features and model errors never produce fallback scores.

### Error Responses

| HTTP status | Error code | Meaning |
|---|---|---|
| 200 | - | Scoring completed |
| 422 | INVALID_TRANSACTION_TIME | Future timestamp |
| 422 | - | Request validation failure |
| 503 | FEATURE_UNAVAILABLE | Missing, stale, or inaccessible features |
| 503 | SCORING_UNAVAILABLE | Configuration, model, or inference failure |

The scoring configuration and model are cached per process.
Restart the API after changing either.

## Delayed Feedback

`POST /feedback` receives verified terminal fraud labels.

Example request:

```json
{
  "transaction_id": "txn-001",
  "outcome": "FRAUD",
  "source": "CHARGEBACK",
  "transaction_at": "2026-10-23T08:00:00Z",
  "labeled_at": "2026-10-25T08:00:00Z"
}
```

Outcomes are `FRAUD` and `LEGITIMATE`.

Sources are `CHARGEBACK` and `MANUAL_REVIEW`.

| HTTP status | Meaning |
|---|---|
| 201 | New label stored |
| 200 | Idempotent replay |
| 409 | Conflicting terminal label |
| 422 | Invalid input |
| 503 | Feedback storage unavailable |

Terminal feedback is immutable in this implementation.

## Prometheus Monitoring

The `GET /metrics` endpoint exposes four metric families.

| Metric | Type |
|---|---|
| `fraud_scoring_http_requests_total` | Counter |
| `fraud_scoring_http_duration_seconds` | Histogram |
| `fraud_scoring_decisions_total` | Counter |
| `fraud_scoring_risk_score` | Histogram |

Inspect metrics locally:

```powershell
(Invoke-WebRequest http://127.0.0.1:8000/metrics).Content
```

### p95 Scoring Latency

PromQL query:

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(fraud_scoring_http_duration_seconds_bucket[5m])
  )
)
```

Latency is measured in seconds.

The project's provisional target is p95 below 0.050 seconds.

The histogram uses a bucket boundary at 0.050 seconds.
The resulting percentile is an estimate.

Instrumentation is not proof that the latency SLO has been met.
The SLO requires load testing against a configured scoring stack.

### Scoring Error Rate

```promql
sum(
  rate(fraud_scoring_http_requests_total{
    status_code=~"5.."
  }[5m])
)
/
sum(
  rate(fraud_scoring_http_requests_total[5m])
)
```

This measures the fraction of scoring responses that were server errors.

### Approval Rate

```promql
sum(
  rate(fraud_scoring_decisions_total{
    decision="APPROVE"
  }[5m])
)
/
sum(
  rate(fraud_scoring_decisions_total[5m])
)
```

This is the approval rate among completed decisions, not all
incoming transactions.

### Monitoring Limitations

- Metrics are process-local.
- Only one application worker is supported for reliable aggregation.
- Histograms approximate score and latency distributions.
- The metrics endpoint is not authenticated.
- Prometheus scraping, Grafana dashboards, and alerts are not configured.
- No automated model rollback or canary promotion is implemented.
- Transaction and customer identifiers are not metric labels.

## Testing

Run the complete test suite:

```powershell
uv run --locked pytest -q
```

Run monitoring instrumentation tests:

```powershell
uv run --locked pytest -q tests/test_scoring_metrics.py
```

## Health Check

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

This reports process availability, not dependency readiness.

## Security

The scoring API, feedback API, and metrics endpoint do not have
authentication or authorization.

Run locally on the loopback interface. Do not expose the application
publicly without access controls and a reviewed deployment design.

## Current Implementation

- FastAPI scoring and feedback endpoints
- Deterministic fraud rules
- Native LightGBM inference
- Redis online features with freshness validation
- Parquet historical feature storage
- PSI prediction drift detection
- SQLite delayed-feedback persistence
- Deterministic decision aggregation
- Integrated HTTP scoring pipeline
- Prometheus HTTP, latency, decision, and score metrics

Production authentication, model training, canary deployment,
Grafana dashboards, distributed tracing, and latency SLO
verification are not implemented.
