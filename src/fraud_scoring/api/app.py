"""FastAPI application and HTTP endpoint wiring."""

import os
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from lightgbm.basic import LightGBMError
from prometheus_client import CONTENT_TYPE_LATEST
from redis.exceptions import RedisError

from fraud_scoring.api.contracts import (
    FeedbackRequest,
    FeedbackResponse,
    ScoreRequest,
    ScoreResponse,
)
from fraud_scoring.api.scoring_runtime import (
    ScoringHandler,
    build_scoring_handler,
)
from fraud_scoring.features.online import FeatureUnavailableError
from fraud_scoring.feedback import (
    FeedbackConflictError,
    FeedbackRecord,
    SQLiteFeedbackStore,
)
from fraud_scoring.monitoring.metrics import (
    observe_scoring_http,
    observe_scoring_result,
    render_metrics,
)


app = FastAPI(
    title="FraudScoring Platform",
    version="0.1.0",
)


@app.middleware("http")
async def measure_scoring_requests(
    request: Request,
    call_next,
) -> Response:
    """Measure POST /score requests, including failed requests."""
    if request.url.path != "/score" or request.method != "POST":
        return await call_next(request)

    started_at = perf_counter()
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        observe_scoring_http(
            status_code=status_code,
            duration_seconds=perf_counter() - started_at,
        )


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose process-local Prometheus metrics."""
    return Response(
        content=render_metrics(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Return API process health."""
    return {"status": "ok"}


@lru_cache(maxsize=1)
def get_scoring_handler() -> ScoringHandler:
    """Return a lazily initialized, process-cached scoring handler.

    FastAPI resolves dependencies before completing request-body
    validation. Returning a lightweight callable ensures that
    invalid transactions receive HTTP 422 without requiring
    scoring infrastructure to be available.

    The backend initializes on the first valid scoring attempt.
    A lock prevents concurrent initialization within the process.
    """
    initialization_lock = Lock()
    backend: ScoringHandler | None = None

    def handle(
        transaction: ScoreRequest,
        evaluated_at: datetime,
    ) -> ScoreResponse:
        nonlocal backend

        if backend is None:
            with initialization_lock:
                if backend is None:
                    try:
                        backend = build_scoring_handler()
                    except (
                        ValueError,
                        TypeError,
                        OSError,
                        RuntimeError,
                        LightGBMError,
                        RedisError,
                    ) as exc:
                        raise HTTPException(
                            status_code=503,
                            detail={
                                "code": "SCORING_UNAVAILABLE",
                                "message": (
                                    "Scoring service is not configured "
                                    "or available."
                                ),
                            },
                        ) from exc

        return backend(transaction, evaluated_at)

    return handle


@app.post(
    "/score",
    response_model=ScoreResponse,
    responses={
        422: {
            "description": "Invalid transaction.",
        },
        503: {
            "description": "Scoring or features unavailable.",
        },
    },
)
def score_transaction(
    transaction: ScoreRequest,
    handler: Annotated[
        ScoringHandler,
        Depends(get_scoring_handler),
    ],
) -> ScoreResponse:
    """Execute real-time scoring without fallback decisions."""
    evaluated_at = datetime.now(timezone.utc)

    if transaction.occurred_at > evaluated_at:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "INVALID_TRANSACTION_TIME",
                "message": "Transaction timestamp is in the future.",
            },
        )

    try:
        result = handler(transaction, evaluated_at)
    except HTTPException:
        raise
    except FeatureUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "FEATURE_UNAVAILABLE",
                "message": (
                    "Required online features are unavailable."
                ),
            },
        ) from exc
    except (
        ValueError,
        RuntimeError,
        OSError,
        RedisError,
        LightGBMError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "SCORING_UNAVAILABLE",
                "message": "Fraud scoring could not be completed.",
            },
        ) from exc

    observe_scoring_result(
        decision=result.decision.value,
        risk_score=result.risk_score,
    )

    return result


def get_feedback_store() -> SQLiteFeedbackStore:
    """Resolve the configured feedback database."""
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
