"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="FraudScoring Platform",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return the API process health status."""
    return {"status": "ok"}