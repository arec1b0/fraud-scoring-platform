# FraudScoring Platform

A Python-based platform for real-time fraud risk scoring.

## Requirements

- Python 3.12
- uv
- Git

## Installation

Run from the repository root:

```powershell
uv sync
```

## Run the API

```powershell
uv run uvicorn fraud_scoring.api.app:app --host 127.0.0.1 --port 8000
```

The API is available at:

- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/health

## Scoring API

`POST /score` accepts transaction data.

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

Amounts should be supplied as decimal strings to preserve precision.

The API validates the request schema. At this stage, no scoring
backend is configured, so valid requests return HTTP 503:

```json
{
  "detail": {
    "code": "SCORING_UNAVAILABLE",
    "message": "A scoring backend is not configured."
  }
}
```

Invalid requests return HTTP 422.

Once a scoring engine is implemented, successful responses will
contain a risk score between 0 and 1 and a decision of APPROVE,
REVIEW, or DECLINE.

## Run tests

```powershell
uv run pytest -q
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

- Python project configuration
- FastAPI application
- Health endpoint
- Transaction scoring request/response contracts
- Input validation
- Explicit scoring-unavailable response
- Automated API tests