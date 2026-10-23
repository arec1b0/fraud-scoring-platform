
"""Immutable Parquet storage for historical customer velocity features."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fraud_scoring.features.online import VelocitySnapshot


_SCHEMA = pa.schema(
    [
        pa.field("customer_id", pa.string(), nullable=False),
        pa.field("transactions_5m", pa.int64(), nullable=False),
        pa.field(
            "computed_at",
            pa.timestamp("us", tz="UTC"),
            nullable=False,
        ),
    ]
)


def _validate_customer_id(customer_id: str) -> None:
    """Validate the identifier used by both feature stores."""
    if (
        not isinstance(customer_id, str)
        or not 1 <= len(customer_id) <= 64
    ):
        raise ValueError(
            "customer_id must contain between 1 and 64 characters"
        )


def _as_utc(value: datetime) -> datetime:
    """Normalize an aware timestamp to UTC."""
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError("Timestamp must be timezone-aware")

    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class OfflineVelocityRecord:
    """A customer identifier paired with a historical feature snapshot."""

    customer_id: str
    snapshot: VelocitySnapshot

    def __post_init__(self) -> None:
        _validate_customer_id(self.customer_id)

        if not isinstance(self.snapshot, VelocitySnapshot):
            raise TypeError("snapshot must be a VelocitySnapshot")


def write_history(
    path: Path,
    records: Iterable[OfflineVelocityRecord],
) -> None:
    """Write one immutable batch of historical velocity snapshots.

    Reject duplicate (customer_id, computed_at) keys and existing files.
    The destination parent directory must already exist.
    """
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, datetime]] = set()

    for record in records:
        if not isinstance(record, OfflineVelocityRecord):
            raise TypeError(
                "Each record must be an OfflineVelocityRecord"
            )

        timestamp = _as_utc(record.snapshot.computed_at)
        count = record.snapshot.transactions_5m
        key = (record.customer_id, timestamp)

        if key in seen:
            raise ValueError(
                "Duplicate customer snapshot timestamp"
            )

        if count > 2**63 - 1:
            raise ValueError(
                "transactions_5m exceeds signed 64-bit range"
            )

        seen.add(key)

        rows.append(
            {
                "customer_id": record.customer_id,
                "transactions_5m": count,
                "computed_at": timestamp,
            }
        )

    if not rows:
        raise ValueError(
            "History batch must contain at least one record"
        )

    table = pa.Table.from_pylist(rows, schema=_SCHEMA)

    created = False

    try:
        # Exclusive creation prevents accidental overwrites.
        with path.open("xb") as stream:
            created = True
            pq.write_table(
                table,
                stream,
                compression="snappy",
            )
    except Exception:
        # Remove a partial artifact after a failed write.
        # Never remove a file that already existed.
        if created:
            path.unlink(missing_ok=True)
        raise


def read_as_of(
    path: Path,
    *,
    customer_id: str,
    at: datetime,
) -> VelocitySnapshot | None:
    """Return the latest snapshot available at or before a timestamp.

    Future snapshots are excluded to prevent point-in-time leakage.
    Return None if no historical snapshot qualifies.
    """
    _validate_customer_id(customer_id)
    cutoff = _as_utc(at)

    with path.open("rb") as stream:
        table = pq.read_table(stream)

    if not table.schema.equals(_SCHEMA, check_metadata=False):
        raise ValueError("Offline feature schema mismatch")

    latest: VelocitySnapshot | None = None

    for row in table.to_pylist():
        if row["customer_id"] != customer_id:
            continue

        timestamp = row["computed_at"]

        if timestamp > cutoff:
            continue

        if latest is not None:
            previous_timestamp = latest.computed_at

            if timestamp == previous_timestamp:
                raise ValueError(
                    "Duplicate customer snapshot timestamp"
                )

            if timestamp < previous_timestamp:
                continue

        latest = VelocitySnapshot(
            transactions_5m=row["transactions_5m"],
            computed_at=timestamp,
        )

    return latest
