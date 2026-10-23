"""FastAPI application and HTTP endpoint wiring."""

from fastapi import FastAPI, HTTPException, status

from fraud_scoring.api.contracts import ScoreRequest, ScoreResponse

app = FastAPI(title="FraudScoring Platform", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Return the API process health status."""
    return {"status": "ok"}


@app.post(
    "/score",
    response_model=ScoreResponse,
    responses={503: {"description": "Scoring backend is not configured."}},
)
def score_transaction(transaction: ScoreRequest) -> ScoreResponse:
    """Validate the transaction; reject scoring until a backend exists."""
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": "SCORING_UNAVAILABLE",
            "message": "A scoring backend is not configured.",
        },
    )