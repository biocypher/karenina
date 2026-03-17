"""Simple extraction template with a single field.

This template demonstrates the most basic answer structure - a single
string field for extracting a simple answer.
"""

from pydantic import Field

from karenina.schemas.entities import BaseAnswer


class Answer(BaseAnswer):
    """Simple single-field extraction template."""

    value: str = Field(description="The extracted answer value")

    def model_post_init(self, __context):
        """Set ground truth after model initialization."""
        self.correct = {"value": "42"}

    def verify(self) -> bool:
        """Verify the extracted value matches ground truth."""
        return self.value.strip() == self.correct["value"]
