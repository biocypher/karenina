"""Base answer class for Karenina.

This module defines the BaseAnswer class, which serves as the foundation for
all answer templates in the benchmark. It provides common functionality and
validation for answer structures.
"""


from pydantic import BaseModel, ConfigDict


class BaseAnswer(BaseModel):
    """Base class for all answer templates in Karenina.

    This class provides common functionality and configuration for answer
    validation and processing.
    """

    model_config = ConfigDict(extra="allow")

    # Question ID will be set programmatically after class instantiation
    id: str | None = None

    def set_question_id(self, question_id: str) -> None:
        """Set the question ID programmatically.

        Args:
            question_id: The unique identifier for the question this answer relates to.
        """
        self.id = question_id
