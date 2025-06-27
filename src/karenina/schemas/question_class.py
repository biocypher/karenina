"""Question schema for Karenina.

This module defines the Question class, which represents a benchmark question
with its associated metadata, including unique identifiers, text content,
and categorization tags.
"""

from pydantic import BaseModel, Field


class Question(BaseModel):
    """Represents a benchmark question with its metadata.

    This class defines the structure and validation rules for questions
    in the benchmark, including unique identifiers, question text, and
    associated metadata.
    """

    id: str = Field(description="Hashed id of the question")
    question: str = Field(description="Question text", min_length=1)
    raw_answer: str = Field(description="Raw answer text", min_length=1)
    tags: list[str | None] = Field(description="Tags of the question")
