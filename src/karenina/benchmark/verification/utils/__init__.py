"""Utility functions for verification operations."""

from .error_helpers import is_retryable_error
from .json_helpers import (
    extract_balanced_braces,
    extract_json_from_text,
    strip_markdown_fences,
)
from .llm_detection import is_openai_endpoint_llm
from .llm_judge_helpers import extract_judge_result, fallback_json_parse
from .prompts import (
    ABSTENTION_DETECTION_SYS,
    ABSTENTION_DETECTION_USER,
    ANSWER_EVALUATION_SYS,
    ANSWER_EVALUATION_USER,
)
from .search_helpers import parse_tool_output
from .template_parsing_helpers import (
    _extract_attribute_descriptions,
    _extract_attribute_names_from_class,
    _extract_text_from_search_results,
    _format_search_results_for_llm,
    create_test_instance_from_answer_class,
    extract_ground_truth_from_template_code,
    extract_rubric_traits_from_template,
    format_excerpts_for_reasoning,
    format_reasoning_for_parsing,
)
from .template_validation import validate_answer_template
from .trace_agent_metrics import (
    TOOL_FAILURE_PATTERNS,
    extract_agent_metrics,
    extract_middleware_metrics,
)
from .trace_parsing import extract_final_ai_message
from .trace_usage_tracker import UsageMetadata, UsageTracker

__all__ = [
    # Error helpers
    "is_retryable_error",
    # JSON helpers
    "strip_markdown_fences",
    "extract_json_from_text",
    "extract_balanced_braces",
    # LLM detection
    "is_openai_endpoint_llm",
    # Search helpers
    "parse_tool_output",
    # Agent metrics
    "TOOL_FAILURE_PATTERNS",
    "extract_agent_metrics",
    "extract_middleware_metrics",
    # Parsing utilities
    "_extract_attribute_names_from_class",
    "_extract_attribute_descriptions",
    "_extract_text_from_search_results",
    "_format_search_results_for_llm",
    "create_test_instance_from_answer_class",
    "extract_ground_truth_from_template_code",
    "extract_rubric_traits_from_template",
    "format_excerpts_for_reasoning",
    "format_reasoning_for_parsing",
    # Prompts
    "ABSTENTION_DETECTION_SYS",
    "ABSTENTION_DETECTION_USER",
    "ANSWER_EVALUATION_SYS",
    "ANSWER_EVALUATION_USER",
    # Validation
    "validate_answer_template",
    # Usage tracking
    "UsageMetadata",
    "UsageTracker",
    # LLM judge helpers
    "extract_judge_result",
    "fallback_json_parse",
    # Trace parsing
    "extract_final_ai_message",
]
