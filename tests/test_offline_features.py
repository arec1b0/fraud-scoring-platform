
"""Tests for immutable Parquet-based offline feature storage."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fraud_scoring.features.offline as offline
from fraud_scoring.features import (
    OfflineVelocityRecord,
    VelocitySnapshot,
    read_as_of,
    write_history,
)


NOW = datetime(2026, 10, 23, 8, 0, tzinfo=timezone.utc)


def make_record(
    customer_id: str = "customer-001",
    count: int = 3,
    computed_at: datetime = NOW,
) -> OfflineVelocityRecord:
    return OfflineVelocityRecord(
        customer_id=customer_id,
        snapshot=VelocitySnapshot(
            transactions_5m=count,
            computed_at=computed_at,
        ),
    )


def test_reads_latest_snapshot_without_future_leakage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.parquet"

    earlier = NOW - timedelta(minutes=5)
    latest = NOW - timedelta(minutes=1)
    future = NOW + timedelta(minutes=1)

    write_history(
        path,
        [
            make_record(count=1, computed_at=earlier),
            make_record(count=4, computed_at=latest),
            make_record(count=9, computed_at=future),
        ],
    )

    result = read_as_of(
        path,
        customer_id="customer-001",
        at=NOW,
    )

    assert result == VelocitySnapshot(4, latest)


def test_accepts_exact_timestamp_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.parquet"
    write_history(path, [make_record()])

    assert read_as_of(
        path,
        customer_id="customer-001",
        at=NOW,
    ) == VelocitySnapshot(3, NOW)


def test_returns_none_when_all_snapshots_are_future(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.parquet"

    write_history(
        path,
        [
            make_record(
                computed_at=NOW + timedelta(seconds=1),
            )
        ],
    )

    assert read_as_of(
        path,
        customer_id="customer-001",
        at=NOW,
    ) is None


def test_does_not_mix_customer_histories(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.parquet"

    write_history(
        path,
        [
            make_record("customer-001", count=3),
            make_record("customer-002", count=8),
        ],
    )

    result = read_as_of(
        path,
        customer_id="customer-002",
        at=NOW,
    )

    assert result == VelocitySnapshot(8, NOW)

    assert read_as_of(
        path,
        customer_id="customer-003",
        at=NOW,
    ) is None


def test_normalizes_timezone_to_utc(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.parquet"
    plus_two = timezone(timedelta(hours=2))

    write_history(
        path,
        [
            make_record(
                computed_at=NOW.astimezone(plus_two),
            )
        ],
    )

    result = read_as_of(
        path,
        customer_id="customer-001",
        at=NOW,
    )

    assert result == VelocitySnapshot(3, NOW)


def test_existing_history_cannot_be_overwritten(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.parquet"

    write_history(path, [make_record(count=3)])

    with pytest.raises(FileExistsError):
        write_history(path, [make_record(count=99)])

    assert read_as_of(
        path,
        customer_id="customer-001",
        at=NOW,
    ) == VelocitySnapshot(3, NOW)


def test_rejects_empty_batch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.parquet"

    with pytest.raises(ValueError, match="at least one"):
        write_history(path, [])

    assert not path.exists()


def test_rejects_duplicate_customer_timestamp(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.parquet"
    plus_two = timezone(timedelta(hours=2))

    records = [
        make_record(count=3, computed_at=NOW),
        make_record(
            count=5,
            computed_at=NOW.astimezone(plus_two),
        ),
    ]

    with pytest.raises(ValueError, match="Duplicate"):
        write_history(path, records)

    assert not path.exists()


@pytest.mark.parametrize(
    "customer_id",
    ["", "x" * 65],
)
def test_rejects_invalid_customer_id(
    customer_id: str,
) -> None:
    with pytest.raises(ValueError, match="customer_id"):
        make_record(customer_id=customer_id)


def test_rejects_timezone_naive_cutoff(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        read_as_of(
            tmp_path / "missing.parquet",
            customer_id="customer-001",
            at=datetime(2026, 10, 23, 8, 0),
        )


def test_missing_parquet_file_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        read_as_of(
            tmp_path / "missing.parquet",
            customer_id="customer-001",
            at=NOW,
        )


def test_rejects_untyped_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.parquet"

    with pytest.raises(
        TypeError,
        match="OfflineVelocityRecord",
    ):
        write_history(path, [object()])

    assert not path.exists()


def test_rejects_incompatible_parquet_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.parquet"

    with path.open("wb") as stream:
        pq.write_table(
            pa.table({"unexpected_column": [1]}),
            stream,
        )

    with pytest.raises(
        ValueError,
        match="schema mismatch",
    ):
        read_as_of(
            path,
            customer_id="customer-001",
            at=NOW,
        )


def test_failed_write_removes_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "history.parquet"

    def fail_write(
        table: pa.Table,
        stream: object,
        **kwargs: object,
    ) -> None:
        stream.write(b"partial")
        raise OSError("Simulated write failure")

    monkeypatch.setattr(
        offline.pq,
        "write_table",
        fail_write,
    )

    with pytest.raises(
        OSError,
        match="Simulated write failure",
    ):
        write_history(path, [make_record()])

    assert not path.exists()


def test_rejects_velocity_outside_int64_range(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.parquet"

    with pytest.raises(
        ValueError,
        match="64-bit range",
    ):
        write_history(
            path,
            [make_record(count=2**63)],
        )

    assert not path.exists()


def test_rejects_invalid_snapshot_type() -> None:
    with pytest.raises(
        TypeError,
        match="VelocitySnapshot",
    ):
        OfflineVelocityRecord(
            customer_id="customer-001",
            snapshot=object(),
        )
