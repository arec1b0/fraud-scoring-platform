"""Tests for persistent and idempotent delayed fraud feedback."""

import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fraud_scoring.feedback import (
    FeedbackConflictError,
    FeedbackRecord,
    SQLiteFeedbackStore,
)


NOW = datetime(2026, 10, 23, 8, 0, tzinfo=timezone.utc)


@pytest.fixture
def feedback() -> FeedbackRecord:
    return FeedbackRecord(
        transaction_id="txn-001",
        outcome="FRAUD",
        source="CHARGEBACK",
        transaction_at=NOW,
        labeled_at=NOW + timedelta(days=2),
    )


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "feedback.sqlite3"


@pytest.fixture
def store(database_path: Path) -> SQLiteFeedbackStore:
    return SQLiteFeedbackStore(database_path)


def test_new_feedback_is_persisted(
    store: SQLiteFeedbackStore,
    database_path: Path,
    feedback: FeedbackRecord,
) -> None:
    assert store.record(feedback) is True

    with closing(sqlite3.connect(database_path)) as connection:
        row = connection.execute(
            """
            SELECT outcome, source, transaction_at_utc, labeled_at_utc
            FROM feedback
            WHERE transaction_id = ?
            """,
            ("txn-001",),
        ).fetchone()

    assert row == (
        "FRAUD",
        "CHARGEBACK",
        NOW.isoformat(timespec="microseconds"),
        (NOW + timedelta(days=2)).isoformat(
            timespec="microseconds"
        ),
    )


def test_exact_replay_is_idempotent(
    store: SQLiteFeedbackStore,
    database_path: Path,
    feedback: FeedbackRecord,
) -> None:
    assert store.record(feedback) is True
    assert store.record(feedback) is False

    with closing(sqlite3.connect(database_path)) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM feedback"
        ).fetchone()[0]

    assert count == 1


def test_feedback_survives_store_reopen(
    database_path: Path,
    feedback: FeedbackRecord,
) -> None:
    first = SQLiteFeedbackStore(database_path)
    assert first.record(feedback) is True

    second = SQLiteFeedbackStore(database_path)
    assert second.record(feedback) is False


def test_equivalent_timezone_offsets_are_idempotent(
    store: SQLiteFeedbackStore,
    feedback: FeedbackRecord,
) -> None:
    plus_two = timezone(timedelta(hours=2))

    equivalent = replace(
        feedback,
        transaction_at=feedback.transaction_at.astimezone(
            plus_two
        ),
        labeled_at=feedback.labeled_at.astimezone(plus_two),
    )

    assert store.record(feedback) is True
    assert store.record(equivalent) is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("outcome", "LEGITIMATE"),
        ("source", "MANUAL_REVIEW"),
        ("labeled_at", NOW + timedelta(days=3)),
    ],
)
def test_conflicting_feedback_is_rejected(
    store: SQLiteFeedbackStore,
    database_path: Path,
    feedback: FeedbackRecord,
    field: str,
    value: object,
) -> None:
    assert store.record(feedback) is True

    conflicting = replace(feedback, **{field: value})

    with pytest.raises(
        FeedbackConflictError,
        match="Conflicting feedback",
    ):
        store.record(conflicting)

    with closing(sqlite3.connect(database_path)) as connection:
        row = connection.execute(
            "SELECT outcome, source FROM feedback"
        ).fetchone()

    assert row == ("FRAUD", "CHARGEBACK")


@pytest.mark.parametrize("identifier", ["", "x" * 65])
def test_invalid_transaction_id_is_rejected(
    feedback: FeedbackRecord,
    identifier: str,
) -> None:
    with pytest.raises(ValueError, match="transaction_id"):
        replace(feedback, transaction_id=identifier)


@pytest.mark.parametrize("outcome", ["UNKNOWN", ""])
def test_invalid_outcome_is_rejected(
    feedback: FeedbackRecord,
    outcome: str,
) -> None:
    with pytest.raises(ValueError, match="outcome"):
        replace(feedback, outcome=outcome)


@pytest.mark.parametrize("source", ["AUTOMATED", ""])
def test_invalid_source_is_rejected(
    feedback: FeedbackRecord,
    source: str,
) -> None:
    with pytest.raises(ValueError, match="source"):
        replace(feedback, source=source)


@pytest.mark.parametrize(
    "field",
    ["transaction_at", "labeled_at"],
)
def test_timezone_naive_timestamp_is_rejected(
    feedback: FeedbackRecord,
    field: str,
) -> None:
    naive = datetime(2026, 10, 23, 8, 0)

    with pytest.raises(ValueError, match="timezone-aware"):
        replace(feedback, **{field: naive})


def test_label_cannot_precede_transaction(
    feedback: FeedbackRecord,
) -> None:
    with pytest.raises(ValueError, match="cannot precede"):
        replace(
            feedback,
            labeled_at=NOW - timedelta(seconds=1),
        )


def test_store_rejects_untyped_feedback(
    store: SQLiteFeedbackStore,
) -> None:
    with pytest.raises(TypeError, match="FeedbackRecord"):
        store.record(object())


def test_transaction_id_is_parameterized(
    store: SQLiteFeedbackStore,
    database_path: Path,
    feedback: FeedbackRecord,
) -> None:
    suspicious_id = "txn'); DROP TABLE feedback;--"
    event = replace(feedback, transaction_id=suspicious_id)

    assert store.record(event) is True

    with closing(sqlite3.connect(database_path)) as connection:
        row = connection.execute(
            "SELECT transaction_id FROM feedback"
        ).fetchone()

    assert row == (suspicious_id,)


def test_missing_database_directory_fails_explicitly(
    tmp_path: Path,
) -> None:
    database = tmp_path / "missing" / "feedback.sqlite3"

    with pytest.raises(sqlite3.OperationalError):
        SQLiteFeedbackStore(database)

    assert not database.exists()


def test_distinct_transactions_are_stored_separately(
    store: SQLiteFeedbackStore,
    database_path: Path,
    feedback: FeedbackRecord,
) -> None:
    second = replace(
        feedback,
        transaction_id="txn-002",
        outcome="LEGITIMATE",
        source="MANUAL_REVIEW",
    )

    assert store.record(feedback) is True
    assert store.record(second) is True

    with closing(sqlite3.connect(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT transaction_id, outcome
            FROM feedback
            ORDER BY transaction_id
            """
        ).fetchall()

    assert rows == [
        ("txn-001", "FRAUD"),
        ("txn-002", "LEGITIMATE"),
    ]
