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
from ...schemas.workflow import FinishedTemplate, VerificationConfig, VerificationResult
from ...utils.async_utils import AsyncConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


def _merge_rubrics(
    global_rubric: Rubric | None,
    template: FinishedTemplate,
    config: VerificationConfig,
) -> Rubric | None:
    """Merge global and question-specific rubrics."""
    if not getattr(config, "rubric_enabled", False):
        return None

    question_rubric = None
    if template.question_rubric:
        try:
            question_rubric = Rubric.model_validate(template.question_rubric)
        except Exception as e:
            logger.warning(f"Failed to parse question rubric for {template.question_id}: {e}")

    try:
        from ...schemas import merge_rubrics

        return merge_rubrics(global_rubric, question_rubric)
    except ValueError as e:
        logger.error(f"Error merging rubrics for {template.question_id}: {e}")
        return global_rubric


def _resolve_few_shot(
    template: FinishedTemplate,
    config: VerificationConfig,
) -> list[dict[str, str]] | None:
    """Resolve few-shot examples for this question."""
    few_shot_config = config.get_few_shot_config()
    if not few_shot_config or not few_shot_config.enabled:
        return None

    return few_shot_config.resolve_examples_for_question(
        question_id=template.question_id,
        available_examples=template.few_shot_examples,
        question_text=template.question_text,
    )


def _extract_feature_flags(config: VerificationConfig) -> dict[str, Any]:
    """Extract feature flags from config."""
    return {
        "few_shot_enabled": config.is_few_shot_enabled(),
        "abstention_enabled": getattr(config, "abstention_enabled", False),
        "deep_judgment_enabled": getattr(config, "deep_judgment_enabled", False),
        "deep_judgment_max_excerpts_per_attribute": getattr(config, "deep_judgment_max_excerpts_per_attribute", 3),
        "deep_judgment_fuzzy_match_threshold": getattr(config, "deep_judgment_fuzzy_match_threshold", 0.80),
        "deep_judgment_excerpt_retry_attempts": getattr(config, "deep_judgment_excerpt_retry_attempts", 2),
        "deep_judgment_search_enabled": getattr(config, "deep_judgment_search_enabled", False),
        "deep_judgment_search_tool": getattr(config, "deep_judgment_search_tool", "tavily"),
    }


# ============================================================================
# Task Queue Generation
# ============================================================================


def generate_task_queue(
    templates: list[FinishedTemplate],
    config: VerificationConfig,
    global_rubric: Rubric | None = None,
    run_name: str | None = None,
    job_id: str | None = None,
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
        job_id: Optional job identifier

    Returns:
        List of task dictionaries with all arguments for verification
    """
    tasks = []

    for template in templates:
        # Prepare rubric for this question
        rubric = _merge_rubrics(global_rubric, template, config)

        # Resolve few-shot examples
        few_shot = _resolve_few_shot(template, config)

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
                            "template_code": template.template_code,
                            # Models
                            "answering_model": ans_model,
                            "parsing_model": parse_model,
                            # Metadata
                            "run_name": run_name,
                            "job_id": job_id,
                            "replicate": replicate,
                            # Context
                            "rubric": rubric,
                            "keywords": template.keywords,
                            "few_shot_examples": few_shot,
                            # Feature flags (from config)
                            **_extract_feature_flags(config),
                        }
                    )

    return tasks


# ============================================================================
# Task Execution
# ============================================================================


def execute_task(task: dict[str, Any]) -> tuple[str, VerificationResult]:
    """
    Execute verification task and return unique key + result.

    Key format: {question_id}_{answering}_{parsing}_rep{N}_{job_id}_{timestamp}

    Args:
        task: Task dictionary with all verification parameters

    Returns:
        Tuple of (result_key, verification_result)
    """
    from .runner import run_single_model_verification

    # Generate unique key
    key_parts = [
        task["question_id"],
        task["answering_model"].id,
        task["parsing_model"].id,
    ]

    if task["replicate"] is not None:
        key_parts.append(f"rep{task['replicate']}")

    if task.get("job_id"):
        key_parts.append(task["job_id"][:8])  # Short job ID

    key_parts.append(str(int(time.time() * 1000)))  # Timestamp in ms

    result_key = "_".join(key_parts)

    # Execute verification
    result = run_single_model_verification(
        question_id=task["question_id"],
        question_text=task["question_text"],
        template_code=task["template_code"],
        answering_model=task["answering_model"],
        parsing_model=task["parsing_model"],
        run_name=task.get("run_name"),
        job_id=task.get("job_id"),
        answering_replicate=task["replicate"],
        parsing_replicate=task["replicate"],
        rubric=task["rubric"],
        keywords=task.get("keywords"),
        few_shot_examples=task.get("few_shot_examples"),
        few_shot_enabled=task.get("few_shot_enabled", False),
        abstention_enabled=task.get("abstention_enabled", False),
        deep_judgment_enabled=task.get("deep_judgment_enabled", False),
        deep_judgment_max_excerpts_per_attribute=task.get("deep_judgment_max_excerpts_per_attribute", 3),
        deep_judgment_fuzzy_match_threshold=task.get("deep_judgment_fuzzy_match_threshold", 0.80),
        deep_judgment_excerpt_retry_attempts=task.get("deep_judgment_excerpt_retry_attempts", 2),
        deep_judgment_search_enabled=task.get("deep_judgment_search_enabled", False),
        deep_judgment_search_tool=task.get("deep_judgment_search_tool", "tavily"),
    )

    return result_key, result


def execute_sequential(
    tasks: list[dict[str, Any]],
    progress_callback: Callable[[int, int, VerificationResult], None] | None = None,
) -> dict[str, VerificationResult]:
    """
    Execute tasks one at a time.

    Args:
        tasks: List of task dictionaries
        progress_callback: Optional callback(current, total, result) for progress updates

    Returns:
        Dictionary mapping result keys to verification results
    """
    results = {}
    total = len(tasks)

    for idx, task in enumerate(tasks, 1):
        result_key, result = execute_task(task)
        results[result_key] = result

        if progress_callback:
            progress_callback(idx, total, result)

    return results


def execute_parallel(
    tasks: list[dict[str, Any]],
    max_workers: int | None = None,
) -> dict[str, VerificationResult]:
    """
    Execute tasks in parallel chunks.

    Args:
        tasks: List of task dictionaries
        max_workers: Optional maximum number of parallel workers (defaults to env var or 2)

    Returns:
        Dictionary mapping result keys to verification results
    """
    import asyncio

    if max_workers is None:
        max_workers = int(os.getenv("KARENINA_ASYNC_MAX_WORKERS", "2"))

    config = AsyncConfig(enabled=True, max_workers=max_workers)

    # Execute all tasks
    from ...utils.async_utils import execute_with_config

    task_results = asyncio.run(
        execute_with_config(
            items=tasks,
            sync_function=execute_task,
            config=config,
        )
    )

    # Convert to dictionary
    results = {}
    for result in task_results:
        if isinstance(result, Exception):
            logger.error(f"Task execution failed: {result}")
            continue

        result_key, verification_result = result
        results[result_key] = verification_result

    return results


# ============================================================================
# Database Auto-Save
# ============================================================================


def auto_save_results(
    results: dict[str, VerificationResult],
    templates: list[FinishedTemplate],
    storage_url: str,
    benchmark_name: str,
    run_name: str,
    config_dict: dict[str, Any],
    run_id: str,
) -> None:
    """
    Auto-save verification results to database.

    Args:
        results: Dictionary of verification results
        templates: List of templates that were verified
        storage_url: Database URL for storage
        benchmark_name: Name of the benchmark
        run_name: Name of this verification run
        config_dict: Configuration dictionary for this run
        run_id: Unique identifier for this run
    """
    try:
        from ...benchmark import Benchmark
        from ...storage import DBConfig, get_benchmark_summary, save_benchmark, save_verification_results

        # Create database config
        db_config = DBConfig(storage_url=storage_url)

        # Check if benchmark already exists
        existing_benchmarks = get_benchmark_summary(db_config, benchmark_name=benchmark_name)

        if not existing_benchmarks:
            # Benchmark doesn't exist, create it
            logger.info(f"Creating new benchmark '{benchmark_name}' in database")
            benchmark = Benchmark.create(
                name=benchmark_name,
                description=f"Auto-created for verification run: {run_name}",
                version="1.0.0",
            )

            # Add questions from templates
            for template in templates:
                # Add question using text format to ensure question_id is preserved
                benchmark.add_question(
                    question=template.question_text,
                    raw_answer="[Placeholder - see template]",
                    answer_template=template.template_code,
                    question_id=template.question_id,  # Explicitly set question_id to match template
                )

            # Save benchmark to database
            save_benchmark(benchmark, db_config)

        # Save verification results
        save_verification_results(
            results=results,
            db_config=db_config,
            run_id=run_id,
            benchmark_name=benchmark_name,
            run_name=run_name,
            config=config_dict,
        )

        logger.info(f"Auto-saved {len(results)} results to {storage_url} (benchmark: {benchmark_name})")

    except Exception as e:
        logger.error(f"Auto-save failed: {e}")
        # Don't raise - auto-save failure shouldn't fail the verification


# ============================================================================
# High-Level Batch Runner
# ============================================================================


def run_verification_batch(
    templates: list[FinishedTemplate],
    config: VerificationConfig,
    run_name: str | None = None,
    job_id: str | None = None,
    global_rubric: Rubric | None = None,
    async_enabled: bool | None = None,
    max_workers: int | None = None,
    storage_url: str | None = None,
    benchmark_name: str | None = None,
    progress_callback: Callable[[int, int, VerificationResult], None] | None = None,
) -> dict[str, VerificationResult]:
    """
    Run batch verification with combinatorial expansion.

    This is the main entry point for standalone verification runs.

    Args:
        templates: List of finished templates to verify
        config: Verification configuration with models and settings
        run_name: Optional name for this run (auto-generated if not provided)
        job_id: Optional job identifier
        global_rubric: Optional global rubric for evaluation
        async_enabled: Whether to run in parallel (defaults to KARENINA_ASYNC_ENABLED env var)
        max_workers: Maximum parallel workers (defaults to KARENINA_ASYNC_MAX_WORKERS env var)
        storage_url: Optional database URL for auto-save
        benchmark_name: Optional benchmark name for auto-save
        progress_callback: Optional callback(current, total, result) for progress updates

    Returns:
        Dictionary mapping result keys to verification results
    """
    # Generate run name if not provided
    if run_name is None:
        import uuid

        run_name = f"run_{uuid.uuid4().hex[:8]}"

    # Determine async mode
    if async_enabled is None:
        async_enabled = os.getenv("KARENINA_ASYNC_ENABLED", "true").lower() == "true"

    # Generate task queue
    logger.info(f"Generating task queue for {len(templates)} templates...")
    task_queue = generate_task_queue(
        templates=templates,
        config=config,
        global_rubric=global_rubric,
        run_name=run_name,
        job_id=job_id,
    )

    # Log execution plan
    logger.info(f"Starting verification: {len(task_queue)} tasks ({'parallel' if async_enabled else 'sequential'})")

    # Execute tasks
    if async_enabled:
        results = execute_parallel(task_queue, max_workers=max_workers)
    else:
        results = execute_sequential(task_queue, progress_callback=progress_callback)

    # Auto-save if configured
    autosave_enabled = os.getenv("AUTOSAVE_DATABASE", "true").lower() in ("true", "1", "yes")
    if autosave_enabled and storage_url and benchmark_name:
        auto_save_results(
            results=results,
            templates=templates,
            storage_url=storage_url,
            benchmark_name=benchmark_name,
            run_name=run_name,
            config_dict=config.model_dump(),
            run_id=job_id if job_id else run_name,
        )

    logger.info(f"Verification complete: {len(results)} results")
    return results
