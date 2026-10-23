
"""Redis-backed storage for customer transaction velocity snapshots."""

import json
from dataclasses import dataclass
from datetime import datetime

from redis import Redis
from redis.exceptions import RedisError


class FeatureUnavailableError(RuntimeError):
    """A required online feature cannot be safely used."""


def _require_aware(value: datetime, name: str) -> None:
    """Require a datetime containing timezone information."""
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{name} must be timezone-aware")


def _require_positive_int(value: int, name: str) -> None:
    """Require a strictly positive integer, excluding booleans."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_customer_id(customer_id: str) -> None:
    """Enforce the customer identifier length contract."""
    if (
        not isinstance(customer_id, str)
        or not 1 <= len(customer_id) <= 64
    ):
        raise ValueError(
            "customer_id must contain between 1 and 64 characters"
        )


@dataclass(frozen=True, slots=True)
class VelocitySnapshot:
    """A precomputed customer transaction velocity snapshot."""

    transactions_5m: int
    computed_at: datetime

    def __post_init__(self) -> None:
        if (
            type(self.transactions_5m) is not int
            or self.transactions_5m < 0
        ):
            raise ValueError(
                "transactions_5m must be a non-negative integer"
            )

        _require_aware(self.computed_at, "computed_at")


class OnlineVelocityStore:
    """Read and write customer velocity snapshots using Redis."""

    KEY_PREFIX = "fraud:features:customer"

    def __init__(self, client: Redis) -> None:
        self._client = client

    @classmethod
    def _key(cls, customer_id: str) -> str:
        _validate_customer_id(customer_id)
        return f"{cls.KEY_PREFIX}:{customer_id}"

    def put(
        self,
        customer_id: str,
        snapshot: VelocitySnapshot,
        *,
        ttl_seconds: int,
    ) -> None:
        """Atomically store a snapshot with a Redis expiration."""
        key = self._key(customer_id)
        _require_positive_int(ttl_seconds, "ttl_seconds")

        payload = json.dumps(
            {
                "transactions_5m": snapshot.transactions_5m,
                "computed_at": snapshot.computed_at.isoformat(),
            },
            separators=(",", ":"),
        )

        try:
            acknowledged = self._client.set(
                key,
                payload,
                ex=ttl_seconds,
            )
        except RedisError as exc:
            raise FeatureUnavailableError(
                "Online feature write failed"
            ) from exc

        if acknowledged is not True:
            raise FeatureUnavailableError(
                "Online feature write was not acknowledged"
            )

    def get_fresh(
        self,
        customer_id: str,
        *,
        now: datetime,
        max_age_seconds: int,
    ) -> VelocitySnapshot:
        """Return a usable snapshot or fail closed."""
        key = self._key(customer_id)
        _require_aware(now, "now")
        _require_positive_int(max_age_seconds, "max_age_seconds")

        try:
            raw = self._client.get(key)
        except RedisError as exc:
            raise FeatureUnavailableError(
                "Online feature read failed"
            ) from exc

        if raw is None:
            raise FeatureUnavailableError(
                "Customer velocity snapshot is missing"
            )

        if not isinstance(raw, (str, bytes)):
            raise FeatureUnavailableError(
                "Customer velocity payload has an invalid type"
            )

        try:
            data = json.loads(raw)

            if (
                not isinstance(data, dict)
                or set(data) != {"transactions_5m", "computed_at"}
            ):
                raise ValueError("Unexpected feature schema")

            timestamp = data["computed_at"]

            if not isinstance(timestamp, str):
                raise ValueError("Invalid timestamp type")

            snapshot = VelocitySnapshot(
                transactions_5m=data["transactions_5m"],
                computed_at=datetime.fromisoformat(timestamp),
            )
        except (ValueError, TypeError, UnicodeError) as exc:
            raise FeatureUnavailableError(
                "Customer velocity payload is invalid"
            ) from exc

        age_seconds = (
            now - snapshot.computed_at
        ).total_seconds()

        if age_seconds < 0:
            raise FeatureUnavailableError(
                "Customer velocity timestamp is in the future"
            )

        if age_seconds > max_age_seconds:
            raise FeatureUnavailableError(
                "Customer velocity snapshot is stale"
            )

        return snapshot
