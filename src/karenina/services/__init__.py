"""Domain services for Karenina.

This package contains domain-specific operations for working with
benchmark questions and answer templates.
"""

from .answers import generate_answer_template
from .questions import read_questions_from_file

__all__ = [
    "generate_answer_template",
    "read_questions_from_file",
]
