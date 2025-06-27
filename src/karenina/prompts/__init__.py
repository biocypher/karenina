"""Prompt templates for Karenina.

This package contains prompt templates used for various LLM interactions,
including answer generation, question processing, and other benchmark-related
tasks.
"""

from .answer_generation import ANSWER_GENERATION_SYS, ANSWER_GENERATION_USER

__all__ = ["ANSWER_GENERATION_SYS", "ANSWER_GENERATION_USER"]
