"""Answer generation and processing functionality for Karenina.

This package provides tools for generating and validating answer templates
for benchmark questions, including LLM-based answer generation and template
validation.
"""

from .generator import generate_answer_template

__all__ = ["generate_answer_template"]
