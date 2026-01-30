"""Task expansion and preview helper functions for verification.

This module provides helpers for task queue generation, rubric merging,
few-shot resolution, and preview result creation.
"""

import logging
from typing import Any

from ....schemas.domain import Rubric
from ....schemas.workflow import (
    FinishedTemplate,
    VerificationConfig,
    VerificationResult,
    VerificationResultMetadata,
)

logger = logging.getLogger(__name__)


def merge_rubrics_for_task(
    global_rubric: Rubric | None,
    template: FinishedTemplate,
    config: VerificationConfig,
) -> Rubric | None:
    """Merge global and question-specific rubrics for a task.

    This is an adapter function that bridges the verification workflow layer
    with the schema layer's pure rubric operations, handling config checking
    and error logging.

    Args:
        global_rubric: Optional global rubric applied to all questions
        template: The finished template containing question-specific rubric
        config: Verification configuration with rubric settings

    Returns:
        Merged rubric or None if rubrics are disabled
    """
    if not getattr(config, "rubric_enabled", False):
        return None

    question_rubric = None
    if template.question_rubric:
        try:
            question_rubric = Rubric.model_validate(template.question_rubric)
        except Exception as e:
            logger.warning(f"Failed to parse question rubric for {template.question_id}: {e}")

    try:
        from ....schemas import merge_rubrics

        return merge_rubrics(global_rubric, question_rubric)
    except ValueError as e:
        logger.error(f"Error merging rubrics for {template.question_id}: {e}")
        return global_rubric


def resolve_few_shot_for_task(
    template: FinishedTemplate,
    config: VerificationConfig,
) -> list[dict[str, str]] | None:
    """Resolve few-shot examples for a task.

    This is an adapter function that bridges the verification workflow layer
    with FewShotConfig's resolution logic, handling null checking.

    Args:
        template: The finished template containing few-shot examples
        config: Verification configuration with few-shot settings

    Returns:
        List of few-shot examples or None if disabled/unavailable
    """
    few_shot_config = config.get_few_shot_config()
    if not few_shot_config or not few_shot_config.enabled:
        return None

    return few_shot_config.resolve_examples_for_question(
        question_id=template.question_id,
        available_examples=template.few_shot_examples,
        question_text=template.question_text,
    )


def create_preview_result(task: dict[str, Any]) -> VerificationResult:
    """Create a preview VerificationResult for progress tracking.

    This creates a minimal result object with empty timestamp to indicate
    that the task is "starting" (not yet completed).

    Args:
        task: Task dictionary with question_id, question_text, and model info

    Returns:
        VerificationResult with preview metadata
    """
    preview_result_id = VerificationResultMetadata.compute_result_id(
        question_id=task["question_id"],
        answering_model=task["answering_model"].id,
        parsing_model=task["parsing_model"].id,
        timestamp="",  # Empty timestamp indicates "starting" event
    )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=task["question_id"],
            template_id="no_template",
            completed_without_errors=False,
            question_text=task["question_text"],
            answering_model=task["answering_model"].id,
            parsing_model=task["parsing_model"].id,
            execution_time=0.0,
            timestamp="",  # Empty timestamp indicates "starting" event
            result_id=preview_result_id,
        )
    )


def extract_feature_flags(config: VerificationConfig) -> dict[str, Any]:
    """Extract feature flags from verification config.

    Args:
        config: Verification configuration

    Returns:
        Dictionary of feature flags for task execution
    """
    return {
        "few_shot_enabled": config.is_few_shot_enabled(),
        "abstention_enabled": getattr(config, "abstention_enabled", False),
        "sufficiency_enabled": getattr(config, "sufficiency_enabled", False),
        "deep_judgment_enabled": getattr(config, "deep_judgment_enabled", False),
        "evaluation_mode": getattr(config, "evaluation_mode", "template_only"),
        "rubric_evaluation_strategy": getattr(config, "rubric_evaluation_strategy", "batch"),
        "deep_judgment_max_excerpts_per_attribute": getattr(config, "deep_judgment_max_excerpts_per_attribute", 3),
        "deep_judgment_fuzzy_match_threshold": getattr(config, "deep_judgment_fuzzy_match_threshold", 0.80),
        "deep_judgment_excerpt_retry_attempts": getattr(config, "deep_judgment_excerpt_retry_attempts", 2),
        "deep_judgment_search_enabled": getattr(config, "deep_judgment_search_enabled", False),
        "deep_judgment_search_tool": getattr(config, "deep_judgment_search_tool", "tavily"),
        # Deep-judgment rubric configuration (NEW)
        "deep_judgment_rubric_mode": getattr(config, "deep_judgment_rubric_mode", "disabled"),
        "deep_judgment_rubric_global_excerpts": getattr(config, "deep_judgment_rubric_global_excerpts", True),
        "deep_judgment_rubric_config": getattr(config, "deep_judgment_rubric_config", None),
        "deep_judgment_rubric_max_excerpts_default": getattr(config, "deep_judgment_rubric_max_excerpts_default", 7),
        "deep_judgment_rubric_fuzzy_match_threshold_default": getattr(
            config, "deep_judgment_rubric_fuzzy_match_threshold_default", 0.80
        ),
        "deep_judgment_rubric_excerpt_retry_attempts_default": getattr(
            config, "deep_judgment_rubric_excerpt_retry_attempts_default", 2
        ),
        "deep_judgment_rubric_search_enabled": getattr(config, "deep_judgment_rubric_search_enabled", False),
        "deep_judgment_rubric_search_tool": getattr(config, "deep_judgment_rubric_search_tool", "tavily"),
        # Prompt configuration
        "prompt_config": getattr(config, "prompt_config", None),
    }
