"""Multi-field extraction template with nested and complex types.

This template demonstrates handling multiple fields including nested structures,
optional fields, and different data types (str, int, list, dict).
"""

from pydantic import Field

from karenina.schemas.domain import BaseAnswer


class Citation(BaseAnswer):
    """Nested citation structure."""

    identifier: str = Field(description="Citation identifier like [1]")
    page: int | None = Field(default=None, description="Page number")
    url: str | None = Field(default=None, description="Source URL")


class Answer(BaseAnswer):
    """Multi-field extraction template with nested structures."""

    # Primary answer fields
    main_answer: str = Field(description="The primary answer text")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0)

    # List fields
    keywords: list[str] = Field(default_factory=list, description="List of relevant keywords")
    entities: list[str] = Field(default_factory=list, description="Named entities mentioned")

    # Nested structure
    citation: Citation | None = Field(default=None, description="Primary citation source")

    # Optional fields
    disclaimer: str | None = Field(default=None, description="Any disclaimer or caveat")

    def model_post_init(self, __context):
        """Set ground truth after model initialization."""
        self.correct = {
            "main_answer": "The mitochondria is the powerhouse of the cell",
            "confidence": 0.95,
            "keywords": ["mitochondria", "cell", "powerhouse", "organelle"],
            "entities": ["cell", "mitochondria"],
            "citation": {"identifier": "[1]", "page": 42, "url": None},
            "disclaimer": None,
        }

    def verify(self) -> bool:
        """Verify all fields match ground truth."""
        if self.main_answer != self.correct["main_answer"]:
            return False

        if self.confidence != self.correct["confidence"]:
            return False

        if set(self.keywords) != set(self.correct["keywords"]):
            return False

        if set(self.entities) != set(self.correct["entities"]):
            return False

        if self.citation is None:
            if self.correct["citation"] is not None:
                return False
        else:
            if self.citation.identifier != self.correct["citation"]["identifier"]:
                return False
            if self.citation.page != self.correct["citation"]["page"]:
                return False
            if self.citation.url != self.correct["citation"]["url"]:
                return False

        return self.disclaimer == self.correct["disclaimer"]
