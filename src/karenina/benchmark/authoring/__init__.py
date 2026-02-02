"""Authoring tools for Karenina benchmarks.

This package provides tools for creating benchmark content:
- Question extraction from files (CSV, Excel, TSV)
- Answer template generation using LLMs
- Template building utilities
"""

from .answers import generate_answer_template
from .questions import read_questions_from_file

__all__ = [
    "generate_answer_template",
    "read_questions_from_file",
]
