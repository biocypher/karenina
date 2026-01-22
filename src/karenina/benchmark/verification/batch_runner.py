"""Batch verification runner with combinatorial task expansion.

This module provides the core verification orchestration logic that can be used
standalone (without karenina-server) to run combinatorial verification tests.
"""

import logging
import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ...schemas.domain import Rubric
from ...schemas.workflow import (
    FinishedTemplate,
    VerificationConfig,
    VerificationResult,
    VerificationResultMetadata,
    VerificationResultSet,
)
from ...utils.answer_cache import AnswerTraceCache

if TYPE_CHECKING:
    from anyio.from_thread import BlockingPortal

logger = logging.getLogger(__name__)

# Module-level portal for async operations from worker threads
# This allows LLM invocation code to access the portal without changing all function signatures
_async_portal: "BlockingPortal | None" = None


def get_async_portal() -> "BlockingPortal | None":
    """Get the current async portal for running async code from threads.

    Returns:
        The BlockingPortal if one is active, None otherwise
    """
    return _async_portal


def set_async_portal(portal: "BlockingPortal | None") -> None:
    """Set the async portal for the current execution context.

    Args:
        portal: The BlockingPortal to use, or None to clear
    """
    global _async_portal
    _async_portal = portal


# ============================================================================
# Helper Functions
# ============================================================================


def _generate_answer_cache_key(task: dict[str, Any]) -> str:
    """Generate cache key for answer traces.

    Cache key format: {question_id}_{answering_model_id}_{replicate}

    Args:
        task: Task dictionary

    Returns:
        Cache key string
    """
    question_id = task["question_id"]
    answering_model_id = task["answering_model"].id
    replicate = task.get("replicate")

    if replicate is None:
        return f"{question_id}_{answering_model_id}"
    else:
        return f"{question_id}_{answering_model_id}_rep{replicate}"


def _extract_answer_data_from_result(result: VerificationResult) -> dict[str, Any]:
    """Extract answer data from verification result for caching.

    Args:
        result: Verification result

    Returns:
        Dictionary with answer data to cache
    """
    # Extract usage metadata for answer generation stage
    # Convert from stage summary format to callback metadata format
    # Stage summary: {"input_tokens": 123, "model": "gpt-4", ...}
    # Callback format: {"gpt-4": {"input_tokens": 123, ...}}
    usage_metadata = None
    if result.usage_metadata and "answer_generation" in result.usage_metadata:
        stage_summary = result.usage_metadata["answer_generation"]
        if stage_summary and isinstance(stage_summary, dict):
            # Extract model name and create callback-style nested dict
            model_name = stage_summary.get("model", "unknown")
            usage_metadata = {
                model_name: {
                    "input_tokens": stage_summary.get("input_tokens", 0),
                    "output_tokens": stage_summary.get("output_tokens", 0),
                    "total_tokens": stage_summary.get("total_tokens", 0),
                    "input_token_details": stage_summary.get("input_token_details", {}),
                    "output_token_details": stage_summary.get("output_token_details", {}),
                }
            }

    return {
        "raw_llm_response": result.raw_llm_response,
        "usage_metadata": usage_metadata,
        "agent_metrics": result.agent_metrics,
        "recursion_limit_reached": result.recursion_limit_reached,
        "answering_mcp_servers": result.answering_mcp_servers,
    }


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


def _create_preview_result(task: dict[str, Any]) -> VerificationResult:
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


def _log_cache_stats(answer_cache: AnswerTraceCache, mode: str = "sequential") -> None:
    """Log answer cache statistics if there were cache interactions.

    Args:
        answer_cache: The answer trace cache instance
        mode: Execution mode for logging ("sequential" or "parallel mode")
    """
    stats = answer_cache.get_stats()
    if stats["hits"] > 0 or stats["waits"] > 0:
        logger.info(
            f"Answer cache statistics ({mode}): {stats['hits']} hits, {stats['misses']} misses, "
            f"{stats['waits']} {'IN_PROGRESS encounters' if mode == 'parallel mode' else 'waits'}, "
            f"{stats['timeouts']} timeouts"
        )


def _extract_feature_flags(config: VerificationConfig) -> dict[str, Any]:
    """Extract feature flags from config."""
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
    }


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
                            **_extract_feature_flags(config),
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
        cache_key = _generate_answer_cache_key(task)
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
        cache_key = _generate_answer_cache_key(task)
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
            cached_answer_data=cached_answer_data,
        )

        # If we generated a new answer, cache it for other tasks
        if answer_cache and should_generate and cache_key:
            answer_data = _extract_answer_data_from_result(result)
            answer_cache.complete(cache_key, answer_data, error=None)

        return result_key, result

    except Exception as e:
        # If answer generation failed and we reserved the slot, mark it as failed
        if answer_cache and should_generate and cache_key:
            answer_cache.complete(cache_key, None, error=e)
        raise


def execute_sequential(
    tasks: list[dict[str, Any]],
    progress_callback: Callable[[int, int, VerificationResult | None], None] | None = None,
) -> dict[str, VerificationResult]:
    """
    Execute tasks one at a time.

    This function creates an answer cache to share answers across multiple
    judges evaluating the same answering model output.

    Args:
        tasks: List of task dictionaries
        progress_callback: Optional callback(current, total, result | None) for progress updates
                          Called twice per task: before (with None) and after (with result)

    Returns:
        Dictionary mapping result keys to verification results
    """
    # Create answer cache for sharing traces across judges
    answer_cache = AnswerTraceCache()

    results = {}
    total = len(tasks)

    for idx, task in enumerate(tasks, 1):
        # Call progress callback BEFORE starting task (with preview result)
        if progress_callback:
            preview_result = _create_preview_result(task)
            progress_callback(idx, total, preview_result)

        # Execute the task with answer cache
        result_key, result = execute_task(task, answer_cache)
        results[result_key] = result

        # Call progress callback AFTER completion (with actual result)
        # This allows the service to remove from in_progress_questions and update counts
        if progress_callback:
            progress_callback(idx, total, result)

    # Log cache statistics
    _log_cache_stats(answer_cache, mode="sequential")

    return results


def execute_parallel(
    tasks: list[dict[str, Any]],
    max_workers: int | None = None,
    progress_callback: Callable[[int, int, VerificationResult | None], None] | None = None,
) -> dict[str, VerificationResult]:
    """
    Execute tasks in parallel with intelligent retry and answer cache optimization.

    This function creates a thread-safe answer cache and uses task shuffling and
    progressive retry to maximize cache hits while avoiding blocking.

    Uses AnyIO BlockingPortal to properly manage async event loops across worker
    threads, preventing connection pool degradation that causes performance tapering.

    Args:
        tasks: List of task dictionaries
        max_workers: Optional maximum number of parallel workers (defaults to env var or 2)
        progress_callback: Optional callback(current, total, result | None) for progress updates
                          Called when task starts (with preview) and completes (with actual result)

    Returns:
        Dictionary mapping result keys to verification results (in original task order)
    """
    import queue
    import random
    import threading

    from anyio.from_thread import start_blocking_portal

    if max_workers is None:
        max_workers = int(os.getenv("KARENINA_ASYNC_MAX_WORKERS", "2"))

    logger.info(f"Parallel execution: {len(tasks)} tasks with {max_workers} workers")

    # Create thread-safe answer cache for sharing traces across judges
    answer_cache = AnswerTraceCache()

    # Preserve original order and shuffle for better cache distribution
    indexed_tasks = [(idx, task) for idx, task in enumerate(tasks)]
    random.shuffle(indexed_tasks)

    # Create work queue with (original_index, task, retry_count)
    work_queue: queue.Queue[tuple[int, dict[str, Any], int] | None] = queue.Queue()
    for idx, task in indexed_tasks:
        work_queue.put((idx, task, 0))

    # Thread-safe storage for results (indexed by original position)
    results_lock = threading.Lock()
    results_by_index: dict[int, tuple[str, VerificationResult]] = {}

    # Thread-safe progress tracking
    progress_lock = threading.Lock()
    completed_count = [0]
    total = len(tasks)

    # Retry wait time for IN_PROGRESS cache entries (reduced from 30s)
    RETRY_WAIT_SECONDS = 5

    # Completion tracking
    tasks_completed_event = threading.Event()

    def worker() -> None:
        """Worker function that processes tasks from queue with retry logic."""
        while True:
            try:
                # Get next task (non-blocking to allow checking for shutdown)
                try:
                    item = work_queue.get(timeout=1.0)
                except queue.Empty:
                    # Check if all tasks are completed
                    with results_lock:
                        if len(results_by_index) == total:
                            tasks_completed_event.set()
                    continue

                # Check for shutdown signal
                if item is None:
                    work_queue.task_done()
                    break

                original_index, task, retry_count = item

                # Check answer cache first
                cache_key = _generate_answer_cache_key(task)
                status, cached_answer_data = answer_cache.get_or_reserve(cache_key)

                if status == "IN_PROGRESS":
                    # Another worker is generating this answer
                    # Always call task_done() immediately after get()
                    work_queue.task_done()

                    if retry_count == 0:
                        # First encounter: immediately requeue and move to next task
                        logger.debug(f"Task {cache_key} in progress (retry={retry_count}), requeueing immediately")
                        work_queue.put((original_index, task, retry_count + 1))
                        continue
                    else:
                        # Subsequent encounter: use cache event-based waiting
                        # Wait for completion with timeout, then requeue if still not ready
                        completed = answer_cache.wait_for_completion(cache_key, timeout=float(RETRY_WAIT_SECONDS))
                        if not completed:
                            logger.debug(f"Task {cache_key} still in progress after {RETRY_WAIT_SECONDS}s, requeueing")
                        work_queue.put((original_index, task, retry_count + 1))
                        continue

                # Status is MISS or HIT - ready to execute
                # Call preview progress callback
                if progress_callback:
                    preview_result = _create_preview_result(task)
                    with progress_lock:
                        progress_callback(completed_count[0] + 1, total, preview_result)

                # Execute the task with pre-checked cache status to avoid double-checking
                try:
                    result_key, verification_result = execute_task(
                        task, answer_cache, cache_status=status, cached_answer_data=cached_answer_data
                    )

                    # Store result at original index
                    with results_lock:
                        results_by_index[original_index] = (result_key, verification_result)
                        # Check if all tasks are now complete
                        if len(results_by_index) == total:
                            tasks_completed_event.set()

                    # Call completion progress callback
                    if progress_callback:
                        with progress_lock:
                            completed_count[0] += 1
                            progress_callback(completed_count[0], total, verification_result)

                except Exception as e:
                    logger.error(f"Task execution failed for {cache_key}: {e}")
                    # Don't raise - log and continue

                finally:
                    work_queue.task_done()

            except Exception as e:
                logger.error(f"Worker error: {e}")
                work_queue.task_done()

    # Use BlockingPortal to properly manage async event loop for all worker threads
    # This prevents "Event loop is closed" errors and connection pool degradation
    with start_blocking_portal(backend="asyncio") as portal:
        # Set the global portal so LLM invocation code can use it
        set_async_portal(portal)

        try:
            # Start worker threads
            workers = []
            for _ in range(max_workers):
                t = threading.Thread(target=worker, daemon=True)
                t.start()
                workers.append(t)

            # Wait for all tasks to actually complete (not just queue empty)
            tasks_completed_event.wait()

            # Send shutdown signal to workers
            for _ in range(max_workers):
                work_queue.put(None)

            # Wait for workers to finish
            for t in workers:
                t.join(timeout=5.0)

        finally:
            # Clear the portal when done
            set_async_portal(None)

    # Restore original order and convert to dictionary
    results = {}
    for idx in sorted(results_by_index.keys()):
        result_key, verification_result = results_by_index[idx]
        results[result_key] = verification_result

    # Log cache statistics
    _log_cache_stats(answer_cache, mode="parallel mode")

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
# Resource Cleanup
# ============================================================================


def cleanup_resources() -> None:
    """
    Clean up lingering resources that may prevent process exit.

    This function attempts to close HTTP connection pools, MCP clients,
    and other resources that might keep background threads alive after
    verification completes.
    """
    import gc

    # Force garbage collection to close unused objects
    gc.collect()

    # Try to close httpx default client (used by many LangChain integrations)
    try:
        import httpx

        # Close the singleton/default client if it exists
        if hasattr(httpx, "_client") and httpx._client:
            httpx._client.close()  # type: ignore[attr-defined]
    except (ImportError, AttributeError, Exception):
        pass

    # Try to close aiohttp default client session
    try:
        import aiohttp

        # Close any default ClientSession instances
        if hasattr(aiohttp, "_default_session") and aiohttp._default_session:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(aiohttp._default_session.close())  # type: ignore[attr-defined]
            except RuntimeError:
                # No event loop running
                pass
    except (ImportError, AttributeError, Exception):
        pass

    # Final garbage collection
    gc.collect()

    logger.debug("Resource cleanup completed")


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

    # Execute tasks
    if async_enabled:
        results = execute_parallel(task_queue, max_workers=max_workers, progress_callback=progress_callback)
    else:
        results = execute_sequential(task_queue, progress_callback=progress_callback)

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
