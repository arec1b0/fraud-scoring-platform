"""Construct the configured real-time scoring pipeline."""

import json
import math
import os
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TypeAlias

from redis import Redis

from fraud_scoring.api.contracts import ScoreRequest, ScoreResponse
from fraud_scoring.api.scoring_service import run_scoring_pipeline
from fraud_scoring.features.online import OnlineVelocityStore
from fraud_scoring.scoring.ml import LightGBMScorer


ScoringHandler: TypeAlias = Callable[
    [ScoreRequest, datetime],
    ScoreResponse,
]


def _required_env(name: str) -> str:
    """Read a required non-empty environment variable."""
    value = os.environ.get(name)

    if value is None or not value.strip():
        raise ValueError(f"Missing configuration: {name}")

    return value


def _threshold(name: str) -> float:
    """Read and validate one decision threshold."""
    try:
        value = float(_required_env(name))
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a number in [0, 1]"
        ) from exc

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be a finite number in [0, 1]"
        )

    return value


def build_scoring_handler() -> ScoringHandler:
    """Build a scoring handler using environment configuration.

    The handler owns one Redis client and one loaded LightGBM model.
    The HTTP dependency caches the handler per application process.
    """
    redis_url = _required_env("FRAUD_REDIS_URL")
    model_path = Path(_required_env("FRAUD_MODEL_PATH"))

    try:
        limits_data = json.loads(
            _required_env("FRAUD_REVIEW_LIMITS_JSON")
        )
        blocked_data = json.loads(
            _required_env("FRAUD_BLOCKED_MERCHANT_IDS_JSON")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Fraud policy configuration must contain valid JSON"
        ) from exc

    if not isinstance(limits_data, dict) or not limits_data:
        raise ValueError(
            "FRAUD_REVIEW_LIMITS_JSON must be a non-empty object"
        )

    review_limits: dict[str, Decimal] = {}

    for currency, raw_limit in limits_data.items():
        if (
            not isinstance(currency, str)
            or len(currency) != 3
            or not currency.isascii()
            or not currency.isalpha()
            or not currency.isupper()
            or not isinstance(raw_limit, str)
        ):
            raise ValueError(
                "Review limits require uppercase currency codes "
                "and decimal-string values"
            )

        try:
            limit = Decimal(raw_limit)
        except InvalidOperation as exc:
            raise ValueError(
                "Review limits must contain valid decimal strings"
            ) from exc

        if not limit.is_finite() or limit <= 0:
            raise ValueError(
                "Review limits must be finite and positive"
            )

        review_limits[currency] = limit

    if (
        not isinstance(blocked_data, list)
        or not all(
            isinstance(identifier, str)
            and 1 <= len(identifier) <= 64
            for identifier in blocked_data
        )
    ):
        raise ValueError(
            "FRAUD_BLOCKED_MERCHANT_IDS_JSON must be "
            "an array of merchant identifiers"
        )

    review_threshold = _threshold("FRAUD_REVIEW_THRESHOLD")
    decline_threshold = _threshold("FRAUD_DECLINE_THRESHOLD")

    if review_threshold >= decline_threshold:
        raise ValueError(
            "Review threshold must be below decline threshold"
        )

    try:
        max_age = int(
            _required_env("FRAUD_MAX_FEATURE_AGE_SECONDS")
        )
    except ValueError as exc:
        raise ValueError(
            "Maximum feature age must be a positive integer"
        ) from exc

    if max_age <= 0:
        raise ValueError(
            "Maximum feature age must be a positive integer"
        )

    model = LightGBMScorer(model_path)

    if set(model.feature_names) != {
        "amount",
        "transactions_5m",
    } or len(model.feature_names) != 2:
        raise ValueError(
            "Model feature schema is incompatible "
            "with the scoring pipeline"
        )

    redis_client = Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=0.05,
        socket_timeout=0.05,
    )

    feature_store = OnlineVelocityStore(redis_client)
    blocked_merchants = frozenset(blocked_data)

    def handle(
        transaction: ScoreRequest,
        evaluated_at: datetime,
    ) -> ScoreResponse:
        """Execute scoring with the initialized dependencies."""
        return run_scoring_pipeline(
            transaction,
            feature_store=feature_store,
            model=model,
            evaluated_at=evaluated_at,
            max_feature_age_seconds=max_age,
            blocked_merchant_ids=blocked_merchants,
            review_limits=review_limits,
            review_threshold=review_threshold,
            decline_threshold=decline_threshold,
        )

    return handle
