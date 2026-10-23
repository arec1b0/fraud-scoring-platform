"""FastAPI application and HTTP endpoint wiring."""

import os
import sqlite3
from pathlib import Path
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Response,
    status,
)

from fraud_scoring.api.contracts import (
    FeedbackRequest,
    FeedbackResponse,
    ScoreRequest,
    ScoreResponse,
)
from fraud_scoring.feedback import (
    FeedbackConflictError,
    FeedbackRecord,
    SQLiteFeedbackStore,
)


app = FastAPI(
    title="FraudScoring Platform",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return the API process health status."""
    return {"status": "ok"}


@app.post(
    "/score",
    response_model=ScoreResponse,
    responses={
        503: {
            "description": "Scoring backend is not configured.",
        }
    },
)
def score_transaction(
    transaction: ScoreRequest,
) -> ScoreResponse:
    """Validate transactions until scoring integration is available."""
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": "SCORING_UNAVAILABLE",
            "message": "A scoring backend is not configured.",
        },
    )


def get_feedback_store() -> SQLiteFeedbackStore:
    """Resolve the configured feedback database for an HTTP request."""
    location = os.environ.get("FRAUD_FEEDBACK_DB_PATH")

    if location is None or not location.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "FEEDBACK_STORAGE_UNAVAILABLE",
                "message": "Feedback storage is not configured.",
            },
        )

    try:
        return SQLiteFeedbackStore(Path(location))
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "FEEDBACK_STORAGE_UNAVAILABLE",
                "message": "Feedback storage is unavailable.",
            },
        ) from exc


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {
            "model": FeedbackResponse,
            "description": "Existing feedback replayed.",
        },
        409: {
            "description": "Conflicting terminal feedback.",
        },
        503: {
            "description": "Feedback storage unavailable.",
        },
    },
)
def record_feedback(
    feedback: FeedbackRequest,
    response: Response,
    store: Annotated[
        SQLiteFeedbackStore,
        Depends(get_feedback_store),
    ],
) -> FeedbackResponse:
    """Persist terminal feedback with idempotent replay handling."""
    record = FeedbackRecord(
        transaction_id=feedback.transaction_id,
        outcome=feedback.outcome,
        source=feedback.source,
        transaction_at=feedback.transaction_at,
        labeled_at=feedback.labeled_at,
    )

    try:
        created = store.record(record)
    except FeedbackConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "FEEDBACK_CONFLICT",
                "message": (
                    "Transaction already has a different "
                    "terminal feedback record."
                ),
            },
        ) from exc
    except (sqlite3.Error, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "FEEDBACK_STORAGE_UNAVAILABLE",
                "message": "Feedback storage is unavailable.",
            },
        ) from exc

    if not created:
        response.status_code = status.HTTP_200_OK

    return FeedbackResponse(
        transaction_id=feedback.transaction_id,
        created=created,
    )
