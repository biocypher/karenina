"""Centralized prompt management for the verification pipeline.

This package provides the PromptTask enum identifying every distinct LLM call
in the verification pipeline, and will host the PromptAssembler and centralized
prompt definitions as the prompt refactor progresses.
"""

from karenina.benchmark.verification.prompts.task_types import PromptTask

__all__ = ["PromptTask"]
