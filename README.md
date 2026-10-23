# FraudScoring Platform

A Python-based platform for real-time fraud risk scoring.

## Requirements

- Python 3.12
- uv
- Git

## Installation

Run from the repository root:

```powershell
uv sync --locked
```

## Run the API

Create the local feedback data directory:

```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
```

Configure the SQLite database path:

```powershell
$env:FRAUD_FEEDBACK_DB_PATH = Join-Path (Get-Location).Path "data\feedback.sqlite3"
```

Start the API:

```powershell
uv run --locked uvicorn fraud_scoring.api.app:app --host 127.0.0.1 --port 8000
```

Available routes:

- `GET /health`
- `POST /score`
- `POST /feedback`
- `GET /docs`

Interactive documentation:

http://127.0.0.1:8000/docs

## Scoring API

`POST /score` validates transaction data.

Example request:

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

The scoring endpoint currently returns HTTP 503 with
`SCORING_UNAVAILABLE` because decision-pipeline integration is
not implemented.

## Delayed Feedback API

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

Possible outcomes:

- `FRAUD`
- `LEGITIMATE`

Supported label sources:

- `CHARGEBACK`
- `MANUAL_REVIEW`

HTTP responses:

| Status | Meaning |
|---|---|
| 201 | New terminal feedback stored |
| 200 | Exact feedback replay |
| 409 | Conflicting feedback already exists |
| 422 | Invalid request |
| 503 | Feedback storage unavailable |

Example response for newly stored feedback:

```json
{
  "transaction_id": "txn-001",
  "created": true
}
```

Feedback is immutable. Correcting an existing label is not supported
by this endpoint.

The feedback database path must be configured using
`FRAUD_FEEDBACK_DB_PATH`.

This endpoint has no authentication and must not be exposed on
a public network.

## Run tests

```powershell
uv run --locked pytest -q
```

## Health check

With the API running, open another PowerShell terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

## Current implementation

- FastAPI application and transaction contracts
- Deterministic rule engine
- Native LightGBM inference module
- Redis online velocity feature storage
- Parquet offline velocity history
- Population Stability Index drift detector
- Persistent delayed feedback storage
- Feedback HTTP endpoint

The scoring endpoint is not yet integrated with the rule engine,
feature stores, or ML inference module.
