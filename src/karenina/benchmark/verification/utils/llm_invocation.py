"""LLM invocation utilities for verification operations.

Shared utilities used by verification stages, runner, and evaluators.
These functions handle common tasks like response parsing and prompt construction.

Functions provided:
- Response parsing: _split_parsed_response
- Prompt construction: _construct_few_shot_prompt
- Agent metrics: _extract_agent_metrics, _extract_middleware_metrics (from trace_agent_metrics)

Note: LLM invocation is now handled by the port/adapter layer:
- LLMPort.invoke() for simple LLM calls
- AgentPort.run_sync() for agent invocations
- ParserPort.parse_to_pydantic() for structured parsing with retry logic

Note: Template-specific parsing and prompt construction is handled by
TemplateEvaluator (evaluators/template/evaluator.py) which encapsulates
all template evaluation logic following the evaluator pattern.
"""

import logging
from typing import Any

from .trace_agent_metrics import (
    extract_agent_metrics,
    extract_middleware_metrics,
)

# Set up logger
logger = logging.getLogger(__name__)

# Re-export agent metrics functions with underscore prefix for backward compatibility
# These are the internal names used throughout the codebase
_extract_agent_metrics = extract_agent_metrics
_extract_middleware_metrics = extract_middleware_metrics


# ============================================================================
# Response Parsing and Validation
# ============================================================================


def _split_parsed_response(parsed_answer: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Split parsed answer into ground truth and LLM response components.

    Args:
        parsed_answer: The parsed answer object from LLM

    Returns:
        Tuple of (parsed_gt_response, parsed_llm_response)
        - parsed_gt_response: The 'correct' field content (ground truth)
        - parsed_llm_response: All other fields except 'id' and 'correct'
    """
    if not parsed_answer or not hasattr(parsed_answer, "model_dump"):
        return None, None

    parsed_dict = parsed_answer.model_dump()

    # Extract ground truth from 'correct' field
    parsed_gt_response = parsed_dict.get("correct")

    # Create LLM response by excluding 'id', 'correct', and 'regex' (configuration fields)
    parsed_llm_response = {k: v for k, v in parsed_dict.items() if k not in ("id", "correct", "regex")}

    return parsed_gt_response, parsed_llm_response


# ============================================================================
# Prompt Construction
# ============================================================================


def _construct_few_shot_prompt(
    question_text: str, few_shot_examples: list[dict[str, str]] | None, few_shot_enabled: bool
) -> str:
    """
    Construct a prompt with few-shot examples if enabled.

    Args:
        question_text: The main question to ask
        few_shot_examples: List of question-answer pairs for few-shot prompting
        few_shot_enabled: Whether few-shot prompting is enabled

    Returns:
        The constructed prompt with optional few-shot examples
    """
    if not few_shot_enabled or not few_shot_examples:
        return question_text

    # Build the prompt with examples
    prompt_parts = []

    for example in few_shot_examples:
        if "question" in example and "answer" in example:
            prompt_parts.append(f"Question: {example['question']}")
            prompt_parts.append(f"Answer: {example['answer']}")
            prompt_parts.append("")  # Empty line for separation

    # Add the actual question
    prompt_parts.append(f"Question: {question_text}")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)
