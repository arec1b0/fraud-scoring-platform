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
- Automated health endpoint test