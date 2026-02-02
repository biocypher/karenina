"""Utility functions and helpers for Karenina.

This package contains various utility functions and helper classes used
throughout the Karenina framework, including code parsing, answer caching,
checkpoint conversion, error handling, retry logic, and other common operations.
"""

from .answer_cache import AnswerTraceCache, CacheEntry
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
    strip_deep_judgment_config_from_checkpoint,
    validate_jsonld_benchmark,
)
from .code import extract_and_combine_codeblocks
from .errors import is_retryable_error
from .json_extraction import extract_json_from_response
from .messages import append_error_feedback
from .retry import TRANSIENT_RETRY, create_transient_retry, log_retry
from .testing import FixtureBackedLLMClient, MockResponse, MockUsage

__all__ = [
    # Answer caching
    "AnswerTraceCache",
    "CacheEntry",
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
    "strip_deep_judgment_config_from_checkpoint",
    # Code parsing
    "extract_and_combine_codeblocks",
    # Error handling
    "is_retryable_error",
    # JSON extraction
    "extract_json_from_response",
    # Message utilities
    "append_error_feedback",
    # Retry utilities
    "TRANSIENT_RETRY",
    "create_transient_retry",
    "log_retry",
    # Testing utilities
    "FixtureBackedLLMClient",
    "MockResponse",
    "MockUsage",
]
