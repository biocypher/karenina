"""Unified failure taxonomy for verification results."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class FailureGroup(str, Enum):
    """Top-level grouping of failure categories for aggregation and UI."""

    CONTENT = "content"
    AUTOFAIL = "autofail"
    RETRY_EXHAUSTED = "retry"
    ABSTAINED = "abstained"
    SYSTEM = "system"


class FailureCategory(str, Enum):
    """Leaf category identifying the specific failure mode."""

    CONTENT = "content"
    RECURSION_LIMIT = "recursion_limit"
    TRACE_VALIDATION = "trace_validation"
    DEEP_JUDGMENT = "deep_judgment"
    DEEP_JUDGMENT_RUBRIC = "deep_judgment_rubric"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    ABSTENTION = "abstention"
    SUFFICIENCY = "sufficiency"
    TEMPLATE_VALIDATION = "template_validation"
    PARSING = "parsing"
    UNEXPECTED_ERROR = "unexpected_error"


CATEGORY_TO_GROUP: dict[FailureCategory, FailureGroup] = {
    FailureCategory.CONTENT: FailureGroup.CONTENT,
    FailureCategory.RECURSION_LIMIT: FailureGroup.AUTOFAIL,
    FailureCategory.TRACE_VALIDATION: FailureGroup.AUTOFAIL,
    FailureCategory.DEEP_JUDGMENT: FailureGroup.AUTOFAIL,
    FailureCategory.DEEP_JUDGMENT_RUBRIC: FailureGroup.AUTOFAIL,
    FailureCategory.TIMEOUT: FailureGroup.RETRY_EXHAUSTED,
    FailureCategory.CONNECTION: FailureGroup.RETRY_EXHAUSTED,
    FailureCategory.RATE_LIMIT: FailureGroup.RETRY_EXHAUSTED,
    FailureCategory.SERVER_ERROR: FailureGroup.RETRY_EXHAUSTED,
    FailureCategory.ABSTENTION: FailureGroup.ABSTAINED,
    FailureCategory.SUFFICIENCY: FailureGroup.ABSTAINED,
    FailureCategory.TEMPLATE_VALIDATION: FailureGroup.SYSTEM,
    FailureCategory.PARSING: FailureGroup.SYSTEM,
    FailureCategory.UNEXPECTED_ERROR: FailureGroup.SYSTEM,
}


class Failure(BaseModel):
    """A structured non-pass verdict for a verification run."""

    model_config = ConfigDict(extra="forbid")

    category: FailureCategory
    stage: str
    reason: str = Field(max_length=500)
    details: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_group(cls, data: Any) -> Any:
        """Silently drop any incoming ``group`` field; it is always computed."""
        if isinstance(data, dict) and "group" in data:
            data = {k: v for k, v in data.items() if k != "group"}
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def group(self) -> FailureGroup:
        return CATEGORY_TO_GROUP[self.category]


__all__ = [
    "CATEGORY_TO_GROUP",
    "Failure",
    "FailureCategory",
    "FailureGroup",
]
