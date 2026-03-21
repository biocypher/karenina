"""Question schema for Karenina.

This module defines the Question class, which represents a self-contained
benchmark question with its associated metadata, and QuestionRegistryEntry,
which tracks benchmark-level state for a question.
"""

import hashlib
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class Question(BaseModel):
    """Represents a self-contained benchmark question with its metadata.

    This class defines the structure and validation rules for questions
    in the benchmark, including unique identifiers, question text,
    categorization keywords, and intrinsic metadata.

    Backward compatibility: the legacy ``tags`` key is accepted during
    construction and automatically converted to ``keywords``.
    """

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(description="Question text", min_length=1)
    raw_answer: str = Field(description="Raw answer text", min_length=1)
    keywords: list[str] = Field(default_factory=list, description="Keywords for the question")
    few_shot_examples: list[dict[str, str]] | None = Field(
        default=None, description="Optional few-shot examples as question-answer pairs"
    )

    # Intrinsic metadata
    date_created: str = Field(default_factory=lambda: datetime.now().isoformat())
    date_modified: str = Field(default_factory=lambda: datetime.now().isoformat())
    answer_template: str | None = None
    answer_notes: str | None = Field(
        default=None, description="Free-text notes about how the answer should be interpreted"
    )
    author: dict[str, Any] | None = None
    sources: list[dict[str, Any]] | None = None
    custom_metadata: dict[str, Any] | None = None
    question_rubric: dict[str, Any] | None = None
    question_dynamic_rubric: dict[str, Any] | None = None
    workspace_path: str | None = Field(
        default=None,
        description=(
            "Relative path from workspace_root to this question's working "
            "directory. For coding benchmarks, this points to the pre-existing "
            "folder containing starter code, tests, or other artifacts for this "
            "task (e.g., 'task_01'). Resolved as workspace_root / workspace_path."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _convert_legacy_tags(cls, data: Any) -> Any:
        """Convert legacy ``tags`` key to ``keywords``."""
        if isinstance(data, dict) and "tags" in data and "keywords" not in data:
            tags = data.pop("tags")
            # Filter out None values from legacy tags list
            data["keywords"] = [t for t in (tags or []) if t is not None]
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def id(self) -> str:
        """Auto-generated MD5 hash of the question text."""
        return hashlib.md5(self.question.encode("utf-8")).hexdigest()


class QuestionRegistryEntry(BaseModel):
    """Tracks benchmark-level state for a question.

    This is separate from the Question model because ``finished`` status
    and benchmark-level timestamps are properties of the question's
    membership in a benchmark, not intrinsic to the question itself.
    """

    model_config = ConfigDict(extra="forbid")

    finished: bool = False
    date_added: str = Field(default_factory=lambda: datetime.now().isoformat())
    date_modified: str = Field(default_factory=lambda: datetime.now().isoformat())
