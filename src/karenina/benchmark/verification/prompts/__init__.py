"""Centralized prompt management for the verification pipeline.

This package provides the PromptTask enum identifying every distinct LLM call
in the verification pipeline, and the PromptAssembler that combines task,
adapter, and user instructions into prompt messages.
"""

from karenina.benchmark.verification.prompts.assembler import PromptAssembler
from karenina.benchmark.verification.prompts.task_types import PromptTask

__all__ = ["PromptAssembler", "PromptTask"]
