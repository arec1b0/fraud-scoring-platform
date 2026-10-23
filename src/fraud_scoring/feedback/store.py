"""Persistent, idempotent storage for delayed fraud labels."""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


class FeedbackConflictError(ValueError):
    """A transaction already has a different terminal feedback record."""


@dataclass(frozen=True, slots=True)
class FeedbackRecord:
    """An immutable fraud outcome associated with a transaction."""

    transaction_id: str
    outcome: Literal["FRAUD", "LEGITIMATE"]
    source: Literal["CHARGEBACK", "MANUAL_REVIEW"]
    transaction_at: datetime
    labeled_at: datetime

    def __post_init__(self) -> None:
        if (
            not isinstance(self.transaction_id, str)
            or not 1 <= len(self.transaction_id) <= 64
        ):
            raise ValueError(
                "transaction_id must contain 1 to 64 characters"
            )

        if self.outcome not in ("FRAUD", "LEGITIMATE"):
            raise ValueError("Invalid feedback outcome")

        if self.source not in ("CHARGEBACK", "MANUAL_REVIEW"):
            raise ValueError("Invalid feedback source")

        for name, timestamp in (
            ("transaction_at", self.transaction_at),
            ("labeled_at", self.labeled_at),
        ):
            if (
                not isinstance(timestamp, datetime)
                or timestamp.tzinfo is None
                or timestamp.utcoffset() is None
            ):
                raise ValueError(
                    f"{name} must be timezone-aware"
                )

        if self.labeled_at < self.transaction_at:
            raise ValueError(
                "labeled_at cannot precede transaction_at"
            )


class SQLiteFeedbackStore:
    """Store one immutable terminal feedback record per transaction."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = Path(database_path)

        with closing(
            sqlite3.connect(self._database_path, timeout=5.0)
        ) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feedback (
                        transaction_id TEXT PRIMARY KEY,
                        outcome TEXT NOT NULL
                            CHECK (outcome IN ('FRAUD', 'LEGITIMATE')),
                        source TEXT NOT NULL
                            CHECK (
                                source IN ('CHARGEBACK', 'MANUAL_REVIEW')
                            ),
                        transaction_at_utc TEXT NOT NULL,
                        labeled_at_utc TEXT NOT NULL,
                        CHECK (labeled_at_utc >= transaction_at_utc)
                    )
                    """
                )

    def record(self, feedback: FeedbackRecord) -> bool:
        """Persist feedback atomically.

        Return True for a new record and False for an exact replay.
        Raise FeedbackConflictError for contradictory feedback.

        All timestamps are normalized to UTC before comparison.
        """
        if not isinstance(feedback, FeedbackRecord):
            raise TypeError("feedback must be a FeedbackRecord")

        values = (
            feedback.transaction_id,
            feedback.outcome,
            feedback.source,
            feedback.transaction_at.astimezone(
                timezone.utc
            ).isoformat(timespec="microseconds"),
            feedback.labeled_at.astimezone(
                timezone.utc
            ).isoformat(timespec="microseconds"),
        )

        with closing(
            sqlite3.connect(self._database_path, timeout=5.0)
        ) as connection:
            with connection:
                # Serialize the read-before-insert transaction.
                connection.execute("BEGIN IMMEDIATE")

                existing = connection.execute(
                    """
                    SELECT
                        outcome,
                        source,
                        transaction_at_utc,
                        labeled_at_utc
                    FROM feedback
                    WHERE transaction_id = ?
                    """,
                    (feedback.transaction_id,),
                ).fetchone()

                if existing is not None:
                    if existing == values[1:]:
                        return False

                    raise FeedbackConflictError(
                        "Conflicting feedback for transaction_id "
                        f"{feedback.transaction_id!r}"
                    )

                connection.execute(
                    """
                    INSERT INTO feedback (
                        transaction_id,
                        outcome,
                        source,
                        transaction_at_utc,
                        labeled_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    values,
                )

        return True
