"""Batch verification runner with combinatorial task expansion.

This module provides the core verification orchestration logic that can be used
standalone (without karenina-server) to run combinatorial verification tests.
"""

import logging
import os
import time
from collections.abc import Callable
from typing import Any

from ...schemas.domain import Rubric
from ...schemas.workflow import (
    FinishedTemplate,
    VerificationConfig,
    VerificationResult,
    VerificationResultSet,
)
from ...utils.answer_cache import AnswerTraceCache
from .utils.cache_helpers import (
    extract_answer_data_from_result,
    generate_answer_cache_key,
)
from .utils.resource_helpers import cleanup_resources
from .utils.storage_helpers import auto_save_results
from .utils.task_helpers import (
    extract_feature_flags,
    merge_rubrics_for_task,
    resolve_few_shot_for_task,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Task Queue Generation
# ============================================================================


def generate_task_queue(
    templates: list[FinishedTemplate],
    config: VerificationConfig,
    global_rubric: Rubric | None = None,
    run_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Generate complete task queue via combinatorial expansion.

    Expansion formula:
      Templates × Answering Models × Parsing Models × Replicates = Total Tasks

    Args:
        templates: List of finished templates to verify
        config: Verification configuration with models and settings
        global_rubric: Optional global rubric for evaluation
        run_name: Optional name for this verification run

    Returns:
        List of task dictionaries with all arguments for verification
    """
    tasks = []

    for template in templates:
        # Prepare rubric for this question
        rubric = merge_rubrics_for_task(global_rubric, template, config)

        # Resolve few-shot examples
        few_shot = resolve_few_shot_for_task(template, config)

        # Expand over model combinations
        for ans_model in config.answering_models:
            for parse_model in config.parsing_models:
                # Expand over replicates
                for rep in range(1, config.replicate_count + 1):
                    # For single replicate, don't include replicate numbers
                    replicate = None if config.replicate_count == 1 else rep

                    tasks.append(
                        {
                            # Core
                            "question_id": template.question_id,
                            "question_text": template.question_text,
                            "raw_answer": template.raw_answer,
                            "template_code": template.template_code,
                            # Models
                            "answering_model": ans_model,
                            "parsing_model": parse_model,
                            # Metadata
                            "run_name": run_name,
                            "replicate": replicate,
                            # Context
                            "rubric": rubric,
                            "keywords": template.keywords,
                            "few_shot_examples": few_shot,
                            # Feature flags (from config)
                            **extract_feature_flags(config),
                        }
                    )

    return tasks


# ============================================================================
# Task Execution
# ============================================================================


def execute_task(
    task: dict[str, Any],
    answer_cache: AnswerTraceCache | None = None,
    cache_status: str | None = None,
    cached_answer_data: dict[str, Any] | None = None,
) -> tuple[str, VerificationResult]:
    """
    Execute verification task and return unique key + result.

    This function supports answer caching to avoid regenerating answers
    when multiple judges evaluate the same answering model output.

    Key format: {question_id}_{answering}_{parsing}_rep{N}_{timestamp}

    Args:
        task: Task dictionary with all verification parameters
        answer_cache: Optional answer cache for sharing traces across judges
        cache_status: Optional pre-checked cache status ("MISS" or "HIT") to avoid double-checking
        cached_answer_data: Optional pre-fetched cached answer data (when cache_status="HIT")

    Returns:
        Tuple of (result_key, verification_result)
    """
    from .runner import run_single_model_verification

    # Generate unique result key
    key_parts = [
        task["question_id"],
        task["answering_model"].id,
        task["parsing_model"].id,
    ]

    if task["replicate"] is not None:
        key_parts.append(f"rep{task['replicate']}")

    key_parts.append(str(int(time.time() * 1000)))  # Timestamp in ms

    result_key = "_".join(key_parts)

    # Check answer cache if available (unless already checked by caller)
    should_generate = True
    cache_key = None

    if answer_cache and cache_status is None:
        # Cache not pre-checked by caller, check it now
        cache_key = generate_answer_cache_key(task)
        status, cached_answer_data = answer_cache.get_or_reserve(cache_key)

        # With new non-blocking cache, IN_PROGRESS should be handled by caller
        # This function should only be called when ready to execute
        if status == "IN_PROGRESS":
            raise RuntimeError(
                f"execute_task() called with IN_PROGRESS cache status for {cache_key}. "
                "Caller should handle requeuing for IN_PROGRESS tasks."
            )

        should_generate = status == "MISS"
    elif answer_cache and cache_status is not None:
        # Cache pre-checked by caller, use provided status
        cache_key = generate_answer_cache_key(task)
        should_generate = cache_status == "MISS"

    # Execute verification
    try:
        result = run_single_model_verification(
            question_id=task["question_id"],
            question_text=task["question_text"],
            template_code=task["template_code"],
            answering_model=task["answering_model"],
            parsing_model=task["parsing_model"],
            run_name=task.get("run_name"),
            replicate=task["replicate"],
            rubric=task["rubric"],
            keywords=task.get("keywords"),
            raw_answer=task.get("raw_answer"),
            few_shot_examples=task.get("few_shot_examples"),
            few_shot_enabled=task.get("few_shot_enabled", False),
            abstention_enabled=task.get("abstention_enabled", False),
            sufficiency_enabled=task.get("sufficiency_enabled", False),
            deep_judgment_enabled=task.get("deep_judgment_enabled", False),
            evaluation_mode=task.get("evaluation_mode", "template_only"),
            rubric_evaluation_strategy=task.get("rubric_evaluation_strategy", "batch"),
            deep_judgment_max_excerpts_per_attribute=task.get("deep_judgment_max_excerpts_per_attribute", 3),
            deep_judgment_fuzzy_match_threshold=task.get("deep_judgment_fuzzy_match_threshold", 0.80),
            deep_judgment_excerpt_retry_attempts=task.get("deep_judgment_excerpt_retry_attempts", 2),
            deep_judgment_search_enabled=task.get("deep_judgment_search_enabled", False),
            deep_judgment_search_tool=task.get("deep_judgment_search_tool", "tavily"),
            # Deep-judgment rubric configuration (NEW)
            deep_judgment_rubric_mode=task.get("deep_judgment_rubric_mode", "disabled"),
            deep_judgment_rubric_global_excerpts=task.get("deep_judgment_rubric_global_excerpts", True),
            deep_judgment_rubric_config=task.get("deep_judgment_rubric_config"),
            deep_judgment_rubric_max_excerpts_default=task.get("deep_judgment_rubric_max_excerpts_default", 7),
            deep_judgment_rubric_fuzzy_match_threshold_default=task.get(
                "deep_judgment_rubric_fuzzy_match_threshold_default", 0.80
            ),
            deep_judgment_rubric_excerpt_retry_attempts_default=task.get(
                "deep_judgment_rubric_excerpt_retry_attempts_default", 2
            ),
            deep_judgment_rubric_search_enabled=task.get("deep_judgment_rubric_search_enabled", False),
            deep_judgment_rubric_search_tool=task.get("deep_judgment_rubric_search_tool", "tavily"),
            # Prompt configuration
            prompt_config=task.get("prompt_config"),
            cached_answer_data=cached_answer_data,
        )

        # If we generated a new answer, cache it for other tasks
        if answer_cache and should_generate and cache_key:
            answer_data = extract_answer_data_from_result(result)
            answer_cache.complete(cache_key, answer_data, error=None)

        return result_key, result

    except Exception as e:
        # If answer generation failed and we reserved the slot, mark it as failed
        if answer_cache and should_generate and cache_key:
            answer_cache.complete(cache_key, None, error=e)
        raise


# ============================================================================
# High-Level Batch Runner
# ============================================================================


def run_verification_batch(
    templates: list[FinishedTemplate],
    config: VerificationConfig,
    run_name: str | None = None,
    global_rubric: Rubric | None = None,
    async_enabled: bool | None = None,
    max_workers: int | None = None,
    storage_url: str | None = None,
    benchmark_name: str | None = None,
    progress_callback: Callable[[int, int, VerificationResult | None], None] | None = None,
) -> VerificationResultSet:
    """
    Run batch verification with combinatorial expansion.

    This is the main entry point for standalone verification runs.

    Args:
        templates: List of finished templates to verify
        config: Verification configuration with models and settings
        run_name: Optional name for this run (auto-generated if not provided)
        global_rubric: Optional global rubric for evaluation
        async_enabled: Whether to run in parallel (defaults to KARENINA_ASYNC_ENABLED env var)
        max_workers: Maximum parallel workers (defaults to KARENINA_ASYNC_MAX_WORKERS env var)
        storage_url: Optional database URL for auto-save
        benchmark_name: Optional benchmark name for auto-save
        progress_callback: Optional callback(current, total, result | None) for progress updates
                          Called before starting each task with preview result

    Returns:
        VerificationResultSet containing all verification results
    """
    # Generate run name if not provided
    if run_name is None:
        import uuid

        run_name = f"run_{uuid.uuid4().hex[:8]}"

    # Determine async mode
    if async_enabled is None:
        async_enabled = os.getenv("KARENINA_ASYNC_ENABLED", "true").lower() == "true"

    # Determine max workers: explicit arg > config > env var > default
    if max_workers is None:
        max_workers = config.async_max_workers  # Uses env var fallback internally

    # Generate task queue
    logger.info(f"Generating task queue for {len(templates)} templates...")
    task_queue = generate_task_queue(
        templates=templates,
        config=config,
        global_rubric=global_rubric,
        run_name=run_name,
    )

    # Log execution plan
    logger.info(f"Starting verification: {len(task_queue)} tasks ({'parallel' if async_enabled else 'sequential'})")

    # Execute tasks using the VerificationExecutor
    from .executor import ExecutorConfig, VerificationExecutor

    executor = VerificationExecutor(
        parallel=async_enabled,
        config=ExecutorConfig(max_workers=max_workers),
    )
    results = executor.run_batch(task_queue, progress_callback)

    # Auto-save if configured
    autosave_enabled = os.getenv("AUTOSAVE_DATABASE", "true").lower() in ("true", "1", "yes")
    if autosave_enabled and storage_url and benchmark_name:
        # Use mode='json' to serialize SecretStr and other non-JSON types properly
        config_dict = config.model_dump(mode="json")
        auto_save_results(
            results=results,
            templates=templates,
            storage_url=storage_url,
            benchmark_name=benchmark_name,
            run_name=run_name,
            config_dict=config_dict,
            run_id=run_name,
        )

    # Convert dict to VerificationResultSet
    result_list = list(results.values())
    result_set = VerificationResultSet(results=result_list)

    logger.info(f"Verification complete: {len(result_set)} results")

    # Clean up lingering resources (HTTP clients, event loops, etc.)
    cleanup_resources()

    return result_set
