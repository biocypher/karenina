"""Schema definitions for Karenina.

This module contains Pydantic models for data validation and serialization.
"""

from .answer_class import BaseAnswer
from .question_class import Question

__all__ = ["BaseAnswer", "Question"]