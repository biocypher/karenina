"""Utility functions and helpers for Karenina.

This package contains various utility functions and helper classes used
throughout the Karenina framework, including code parsing, async execution,
answer caching, checkpoint conversion, and other common operations.
"""

from .answer_cache import AnswerTraceCache, CacheEntry
from .async_utils import AsyncConfig, execute_with_config, run_async_chunked, run_sync_with_progress
from .checkpoint import (
    BenchmarkConversionError,
    add_global_rubric_to_benchmark,
    add_question_to_benchmark,
    convert_rating_to_rubric_trait,
    convert_rubric_trait_to_rating,
    create_jsonld_benchmark,
    extract_global_rubric_from_benchmark,
    extract_questions_from_benchmark,
    generate_question_id,
    generate_template_id,
    validate_jsonld_benchmark,
)
from .code import extract_and_combine_codeblocks

__all__ = [
    # Answer caching
    "AnswerTraceCache",
    "CacheEntry",
    # Async utilities
    "AsyncConfig",
    "execute_with_config",
    "run_async_chunked",
    "run_sync_with_progress",
    # Checkpoint utilities
    "BenchmarkConversionError",
    "generate_template_id",
    "generate_question_id",
    "add_question_to_benchmark",
    "add_global_rubric_to_benchmark",
    "extract_global_rubric_from_benchmark",
    "extract_questions_from_benchmark",
    "create_jsonld_benchmark",
    "validate_jsonld_benchmark",
    "convert_rubric_trait_to_rating",
    "convert_rating_to_rubric_trait",
    # Code parsing
    "extract_and_combine_codeblocks",
]
