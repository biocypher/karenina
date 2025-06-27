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
