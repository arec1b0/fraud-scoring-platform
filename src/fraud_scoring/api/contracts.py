"""Public HTTP request and response contracts."""

from decimal import Decimal
from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class Decision(StrEnum):
    """Possible outcomes of a completed scoring decision."""

    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    DECLINE = "DECLINE"


class ScoreRequest(BaseModel):
    """A transaction submitted for fraud scoring."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1, max_length=64)
    customer_id: str = Field(min_length=1, max_length=64)
    merchant_id: str = Field(min_length=1, max_length=64)
    amount: Decimal = Field(gt=0, max_digits=18, decimal_places=2)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    occurred_at: AwareDatetime


class ScoreResponse(BaseModel):
    """The response contract for a completed fraud assessment."""

    transaction_id: str
    risk_score: float = Field(ge=0, le=1)
    decision: Decision


class FeedbackRequest(BaseModel):
    """A terminal fraud label received after a transaction."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(
        min_length=1,
        max_length=64,
        strict=True,
    )
    outcome: Literal["FRAUD", "LEGITIMATE"]
    source: Literal["CHARGEBACK", "MANUAL_REVIEW"]
    transaction_at: AwareDatetime
    labeled_at: AwareDatetime

    @model_validator(mode="after")
    def validate_label_chronology(self) -> Self:
        """Reject labels dated before their transactions."""
        if self.labeled_at < self.transaction_at:
            raise ValueError(
                "labeled_at cannot precede transaction_at"
            )
        return self


class FeedbackResponse(BaseModel):
    """Result of feedback ingestion or idempotent replay."""

    transaction_id: str
    created: bool
