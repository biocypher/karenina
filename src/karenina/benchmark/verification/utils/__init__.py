"""Utility functions for verification operations."""

from .agent_metrics import (
    TOOL_FAILURE_PATTERNS,
    extract_agent_metrics,
    extract_middleware_metrics,
)
from .parsing import (
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
from .prompts import (
    ABSTENTION_DETECTION_SYS,
    ABSTENTION_DETECTION_USER,
    ANSWER_EVALUATION_SYS,
    ANSWER_EVALUATION_USER,
)
from .shared import parse_tool_output as _parse_tool_output
from .shared import strip_markdown_fences as _strip_markdown_fences
from .usage_tracker import UsageMetadata, UsageTracker
from .validation import validate_answer_template

__all__ = [
    # Agent metrics
    "TOOL_FAILURE_PATTERNS",
    "extract_agent_metrics",
    "extract_middleware_metrics",
    # Parsing utilities
    "_extract_attribute_names_from_class",
    "_strip_markdown_fences",
    "_extract_attribute_descriptions",
    "_extract_text_from_search_results",
    "_format_search_results_for_llm",
    "_parse_tool_output",
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
]
