"""Public request and response contracts for transaction scoring."""

from decimal import Decimal
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


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