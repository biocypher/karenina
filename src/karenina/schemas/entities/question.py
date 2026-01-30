"""Question schema for Karenina.

This module defines the Question class, which represents a benchmark question
with its associated metadata, including unique identifiers, text content,
and categorization tags.
"""

import hashlib

from pydantic import BaseModel, Field, computed_field


class Question(BaseModel):
    """Represents a benchmark question with its metadata.

    This class defines the structure and validation rules for questions
    in the benchmark, including unique identifiers, question text, and
    associated metadata.
    """

    question: str = Field(description="Question text", min_length=1)
    raw_answer: str = Field(description="Raw answer text", min_length=1)
    tags: list[str | None] = Field(default_factory=list, description="Tags of the question")
    few_shot_examples: list[dict[str, str]] | None = Field(
        default=None, description="Optional few-shot examples as question-answer pairs"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def id(self) -> str:
        """Auto-generated MD5 hash of the question text."""
        return hashlib.md5(self.question.encode("utf-8")).hexdigest()
