"""Template using ground_truth(self) instead of model_post_init.

This template demonstrates the new ground_truth alias for setting
correct values, which avoids the Pydantic-specific model_post_init signature.
"""

from pydantic import Field

from karenina.schemas.entities import BaseAnswer


class Answer(BaseAnswer):
    """Single-field extraction template using ground_truth style."""

    target: str = Field(description="The identified drug target")

    def ground_truth(self):
        """Set ground truth values."""
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        """Verify the extracted target matches ground truth."""
        return self.target.strip().upper() == self.correct["target"].upper()
