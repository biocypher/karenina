"""Task expansion and preview helper functions for verification.

This module provides helpers for task queue generation, rubric merging,
few-shot resolution, and preview result creation.
"""

import logging
from typing import Any

from karenina.schemas.entities import Rubric
from karenina.schemas.entities.rubric import DynamicRubric, merge_dynamic_rubrics
from karenina.schemas.verification import (
    FinishedTemplate,
    VerificationConfig,
    VerificationResult,
    VerificationResultMetadata,
)

logger = logging.getLogger(__name__)


def model_sort_key(model: Any) -> str:
    """Return a stable string key for sorting models by identity.

    Uses the model's ``id`` if available, falling back to ``model_name``,
    then to an empty string. This groups tasks by answering model so that
    prefix caches (KV caches) can be reused across consecutive requests.

    Args:
        model: A ModelConfig instance (or any object with ``id`` / ``model_name``).

    Returns:
        A string suitable for use as a sort key.
    """
    return getattr(model, "id", None) or getattr(model, "model_name", None) or ""


def stamp_agentic_trait_overrides(
    rubric: Rubric | None,
    config: VerificationConfig,
) -> Rubric | None:
    """Stamp pipeline-level retry policy and request timeout onto agentic trait overrides.

    Walks ``rubric.agentic_traits`` and, for any trait whose ``model_override``
    is set, returns a copy with ``request_timeout`` and ``retry_policy`` filled
    in from ``config`` (mirroring how the top-level answering and parsing
    models are stamped). Existing values on the override are preserved: only
    ``None`` fields are populated, matching the ``is None`` guard used by
    :func:`_apply_request_timeout` and :func:`_apply_retry_config` in
    :mod:`karenina.benchmark.verification.batch_runner`.

    The original rubric instance is returned unchanged when no traits need
    stamping (no agentic traits, none with ``model_override``, or all override
    fields already populated). This avoids rebuilding frozen rubric and trait
    instances in the common case.

    Args:
        rubric: The rubric whose agentic traits should be inspected; ``None``
            short-circuits to ``None``.
        config: Verification configuration providing the pipeline-level
            ``request_timeout`` and ``retry_policy`` defaults.

    Returns:
        The original rubric (unchanged) when no stamping was required, or a
        new ``Rubric`` instance with stamped agentic traits.
    """
    if rubric is None:
        return None
    if not rubric.agentic_traits:
        return rubric

    new_traits: list[Any] | None = None
    for idx, trait in enumerate(rubric.agentic_traits):
        override = trait.model_override
        if override is None:
            continue
        updates: dict[str, Any] = {}
        if config.request_timeout is not None and override.request_timeout is None:
            updates["request_timeout"] = config.request_timeout
        if config.retry_policy is not None and override.retry_policy is None:
            updates["retry_policy"] = config.retry_policy
        if not updates:
            continue
        stamped_override = override.model_copy(update=updates)
        if new_traits is None:
            new_traits = list(rubric.agentic_traits)
        new_traits[idx] = trait.model_copy(update={"model_override": stamped_override})

    if new_traits is None:
        return rubric
    return rubric.model_copy(update={"agentic_traits": new_traits})


def merge_rubrics_for_task(
    global_rubric: Rubric | None,
    template: FinishedTemplate,
    config: VerificationConfig,
) -> tuple[Rubric | None, dict[str, str] | None]:
    """Merge global and question-specific rubrics for a task.

    This is an adapter function that bridges the verification workflow layer
    with the schema layer's pure rubric operations, handling config checking
    and error logging. After the merge, any agentic traits whose
    ``model_override`` lacks ``request_timeout`` or ``retry_policy`` are
    stamped from the pipeline-level configuration via
    :func:`stamp_agentic_trait_overrides`, mirroring the behavior applied to
    the top-level answering and parsing models in
    :mod:`karenina.benchmark.verification.batch_runner`.

    Args:
        global_rubric: Optional global rubric applied to all questions
        template: The finished template containing question-specific rubric
        config: Verification configuration with rubric settings

    Returns:
        Tuple of (merged_rubric, provenance) where provenance maps each
        trait name to its source ("global" or "question_specific"). Both
        elements are None when rubrics are disabled or both inputs are None.
    """
    if not config.rubric_enabled:
        return None, None

    question_rubric = None
    if template.question_rubric:
        try:
            question_rubric = Rubric.model_validate(template.question_rubric)
        except Exception as e:
            logger.warning("Failed to parse question rubric for %s: %s", template.question_id, e)

    from karenina.schemas import merge_rubrics

    merged, provenance = merge_rubrics(global_rubric, question_rubric)
    stamped = stamp_agentic_trait_overrides(merged, config)
    return stamped, provenance


def merge_dynamic_rubrics_for_task(
    global_dynamic_rubric: DynamicRubric | None,
    template: FinishedTemplate,
    config: VerificationConfig,
) -> DynamicRubric | None:
    """Merge global and question-specific dynamic rubrics for a task.

    Mirrors :func:`merge_rubrics_for_task` for the dynamic rubric variant.
    Deserializes the question-level dict into a DynamicRubric, then delegates
    to :func:`merge_dynamic_rubrics` for the actual merge.

    Args:
        global_dynamic_rubric: Optional global dynamic rubric applied to all questions.
        template: The finished template containing question-specific dynamic rubric.
        config: Verification configuration with rubric settings.

    Returns:
        Merged DynamicRubric or None if rubrics are disabled or absent.
    """
    if not config.rubric_enabled:
        return None

    question_dynamic_rubric = None
    if template.question_dynamic_rubric:
        try:
            question_dynamic_rubric = DynamicRubric.model_validate(template.question_dynamic_rubric)
        except Exception as e:
            logger.warning(
                "Failed to parse question dynamic rubric for %s: %s",
                template.question_id,
                e,
            )

    return merge_dynamic_rubrics(global_dynamic_rubric, question_dynamic_rubric)


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
    if not few_shot_config or few_shot_config.source == "disabled":
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
        task: Task dictionary with question_id, question_text, model info, and optional replicate

    Returns:
        VerificationResult with preview metadata including replicate info
    """
    from karenina.schemas.verification.model_identity import ModelIdentity

    answering_identity = ModelIdentity.from_model_config(task["answering_model"], role="answering")
    parsing_identity = ModelIdentity.from_model_config(task["parsing_model"], role="parsing")
    replicate = task.get("replicate")  # May be None for single-replicate runs

    preview_result_id = VerificationResultMetadata.compute_result_id(
        question_id=task["question_id"],
        answering=answering_identity,
        parsing=parsing_identity,
        timestamp="",  # Empty timestamp indicates "starting" event
        replicate=replicate,
    )
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=task["question_id"],
            template_id="no_template",
            completed_without_errors=False,
            question_text=task["question_text"],
            answering=answering_identity,
            parsing=parsing_identity,
            execution_time=0.0,
            timestamp="",  # Empty timestamp indicates "starting" event
            result_id=preview_result_id,
            replicate=replicate,
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
        "deep_judgment_mode": getattr(config, "deep_judgment_mode", "disabled"),
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
        # Embedding check configuration
        "embedding_check_enabled": getattr(config, "embedding_check_enabled", False),
        "embedding_check_model": getattr(config, "embedding_check_model", None),
        "embedding_check_threshold": getattr(config, "embedding_check_threshold", None),
        # Prompt configuration
        "prompt_config": getattr(config, "prompt_config", None),
        # Agentic parsing configuration
        "agentic_parsing": getattr(config, "agentic_parsing", False),
        "agentic_judge_context": getattr(config, "agentic_judge_context", "workspace_only"),
        "agentic_parsing_max_turns": getattr(config, "agentic_parsing_max_turns", 15),
        "agentic_parsing_timeout": getattr(config, "agentic_parsing_timeout", 120.0),
        "agentic_parsing_materialize_trace": getattr(config, "agentic_parsing_materialize_trace", False),
        "agentic_parsing_persist_trace": getattr(config, "agentic_parsing_persist_trace", False),
        "workspace_copy": getattr(config, "workspace_copy", True),
        "workspace_cleanup": getattr(config, "workspace_cleanup", True),
        # Agentic rubric evaluation configuration
        "agentic_rubric_strategy": getattr(config, "agentic_rubric_strategy", "individual"),
        "agentic_rubric_parallel": getattr(config, "agentic_rubric_parallel", False),
        # Error classification
        "custom_error_patterns": getattr(config, "custom_error_patterns", []),
    }


def replicate_range(count: int) -> list[int | None]:
    """Replicate iteration matching the task queue convention.

    Returns [None] for count <= 1 (no replicate numbering),
    list[1..count] otherwise.
    """
    if count <= 1:
        return [None]
    return list(range(1, count + 1))
