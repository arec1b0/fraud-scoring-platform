
"""Tests for Redis-backed customer velocity features."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from redis import Redis
from redis.exceptions import RedisError

from fraud_scoring.features import (
    FeatureUnavailableError,
    OnlineVelocityStore,
    VelocitySnapshot,
)


NOW = datetime(2026, 10, 23, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
def client() -> Mock:
    redis_client = Mock(spec=Redis)
    redis_client.set.return_value = True
    return redis_client


@pytest.fixture
def store(client: Mock) -> OnlineVelocityStore:
    return OnlineVelocityStore(client)


def payload(
    *,
    count: object = 3,
    computed_at: datetime = NOW,
) -> bytes:
    return json.dumps(
        {
            "transactions_5m": count,
            "computed_at": computed_at.isoformat(),
        }
    ).encode("utf-8")


def test_put_writes_snapshot_with_ttl(
    store: OnlineVelocityStore,
    client: Mock,
) -> None:
    snapshot = VelocitySnapshot(
        transactions_5m=3,
        computed_at=NOW,
    )

    store.put("customer-001", snapshot, ttl_seconds=120)

    client.set.assert_called_once()
    args, kwargs = client.set.call_args

    assert args[0] == (
        "fraud:features:customer:customer-001"
    )
    assert json.loads(args[1]) == {
        "transactions_5m": 3,
        "computed_at": NOW.isoformat(),
    }
    assert kwargs == {"ex": 120}


def test_reads_fresh_snapshot(
    store: OnlineVelocityStore,
    client: Mock,
) -> None:
    client.get.return_value = payload()

    result = store.get_fresh(
        "customer-001",
        now=NOW + timedelta(seconds=10),
        max_age_seconds=30,
    )

    assert result == VelocitySnapshot(3, NOW)


def test_accepts_exact_freshness_boundary(
    store: OnlineVelocityStore,
    client: Mock,
) -> None:
    client.get.return_value = payload()

    result = store.get_fresh(
        "customer-001",
        now=NOW + timedelta(seconds=30),
        max_age_seconds=30,
    )

    assert result.transactions_5m == 3


def test_rejects_stale_snapshot(
    store: OnlineVelocityStore,
    client: Mock,
) -> None:
    client.get.return_value = payload()

    with pytest.raises(
        FeatureUnavailableError,
        match="stale",
    ):
        store.get_fresh(
            "customer-001",
            now=NOW + timedelta(seconds=31),
            max_age_seconds=30,
        )


def test_rejects_missing_snapshot(
    store: OnlineVelocityStore,
    client: Mock,
) -> None:
    client.get.return_value = None

    with pytest.raises(
        FeatureUnavailableError,
        match="missing",
    ):
        store.get_fresh(
            "customer-001",
            now=NOW,
            max_age_seconds=30,
        )


def test_rejects_future_timestamp(
    store: OnlineVelocityStore,
    client: Mock,
) -> None:
    client.get.return_value = payload(
        computed_at=NOW + timedelta(seconds=5),
    )

    with pytest.raises(
        FeatureUnavailableError,
        match="future",
    ):
        store.get_fresh(
            "customer-001",
            now=NOW,
            max_age_seconds=30,
        )


@pytest.mark.parametrize(
    "raw",
    [
        b"not-json",
        b'{"transactions_5m":3}',
        b'{"transactions_5m":3,"computed_at":"invalid"}',
    ],
)
def test_rejects_invalid_payload(
    store: OnlineVelocityStore,
    client: Mock,
    raw: bytes,
) -> None:
    client.get.return_value = raw

    with pytest.raises(
        FeatureUnavailableError,
        match="payload is invalid",
    ):
        store.get_fresh(
            "customer-001",
            now=NOW,
            max_age_seconds=30,
        )


@pytest.mark.parametrize("count", [True, -1])
def test_rejects_invalid_velocity_count(
    count: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="non-negative integer",
    ):
        VelocitySnapshot(count, NOW)


def test_rejects_naive_snapshot_timestamp() -> None:
    naive = datetime(2026, 10, 23, 8, 0)

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        VelocitySnapshot(3, naive)


@pytest.mark.parametrize("ttl_seconds", [0, True])
def test_rejects_invalid_ttl(
    store: OnlineVelocityStore,
    ttl_seconds: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="ttl_seconds",
    ):
        store.put(
            "customer-001",
            VelocitySnapshot(3, NOW),
            ttl_seconds=ttl_seconds,
        )


@pytest.mark.parametrize(
    "customer_id",
    ["", "x" * 65],
)
def test_rejects_invalid_customer_id(
    store: OnlineVelocityStore,
    customer_id: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="customer_id",
    ):
        store.put(
            customer_id,
            VelocitySnapshot(3, NOW),
            ttl_seconds=120,
        )


def test_redis_read_error_fails_closed(
    store: OnlineVelocityStore,
    client: Mock,
) -> None:
    client.get.side_effect = RedisError("Connection failed")

    with pytest.raises(
        FeatureUnavailableError,
        match="Online feature read failed",
    ):
        store.get_fresh(
            "customer-001",
            now=NOW,
            max_age_seconds=30,
        )


def test_unacknowledged_write_fails_closed(
    store: OnlineVelocityStore,
    client: Mock,
) -> None:
    client.set.return_value = False

    with pytest.raises(
        FeatureUnavailableError,
        match="not acknowledged",
    ):
        store.put(
            "customer-001",
            VelocitySnapshot(3, NOW),
            ttl_seconds=120,
        )


def test_rejects_invalid_max_age(
    store: OnlineVelocityStore,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_age_seconds",
    ):
        store.get_fresh(
            "customer-001",
            now=NOW,
            max_age_seconds=0,
        )


def test_rejects_naive_current_time(
    store: OnlineVelocityStore,
) -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        store.get_fresh(
            "customer-001",
            now=datetime(2026, 10, 23, 8, 0),
            max_age_seconds=30,
        )
