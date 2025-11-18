"""Orchestration logic for multi-model verification."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ...schemas.domain import Rubric
from ...schemas.workflow import ModelConfig, VerificationConfig, VerificationResult
from .runner import run_single_model_verification
from .tools.embedding_check import clear_embedding_model_cache, preload_embedding_model


def _create_verification_task(
    question_id: str,
    question_text: str,
    template_code: str,
    answering_model: ModelConfig,
    parsing_model: ModelConfig,
    run_name: str | None,
    job_id: str | None,
    answering_replicate: int | None,
    parsing_replicate: int | None,
    rubric: Rubric | None,
    keywords: list[str] | None = None,
    few_shot_examples: list[dict[str, str]] | None = None,
    few_shot_enabled: bool = False,
    abstention_enabled: bool = False,
    deep_judgment_enabled: bool = False,
    rubric_evaluation_strategy: str = "batch",
    deep_judgment_max_excerpts_per_attribute: int = 3,
    deep_judgment_fuzzy_match_threshold: float = 0.80,
    deep_judgment_excerpt_retry_attempts: int = 2,
    deep_judgment_search_enabled: bool = False,
    deep_judgment_search_tool: str | Any = "tavily",
) -> dict[str, Any]:
    """Create a verification task dictionary from parameters."""
    return {
        "question_id": question_id,
        "question_text": question_text,
        "template_code": template_code,
        "answering_model": answering_model,
        "parsing_model": parsing_model,
        "run_name": run_name,
        "job_id": job_id,
        "answering_replicate": answering_replicate,
        "parsing_replicate": parsing_replicate,
        "rubric": rubric,
        "keywords": keywords,
        "few_shot_examples": few_shot_examples,
        "few_shot_enabled": few_shot_enabled,
        "abstention_enabled": abstention_enabled,
        "deep_judgment_enabled": deep_judgment_enabled,
        "rubric_evaluation_strategy": rubric_evaluation_strategy,
        "deep_judgment_max_excerpts_per_attribute": deep_judgment_max_excerpts_per_attribute,
        "deep_judgment_fuzzy_match_threshold": deep_judgment_fuzzy_match_threshold,
        "deep_judgment_excerpt_retry_attempts": deep_judgment_excerpt_retry_attempts,
        "deep_judgment_search_enabled": deep_judgment_search_enabled,
        "deep_judgment_search_tool": deep_judgment_search_tool,
    }


def _execute_verification_task(task: dict[str, Any]) -> tuple[str, VerificationResult]:
    """Execute a single verification task and return result with its key."""
    # Extract parameters from task
    question_id = task["question_id"]
    answering_model = task["answering_model"]
    parsing_model = task["parsing_model"]
    answering_replicate = task["answering_replicate"]

    # Generate result key
    if answering_replicate is not None:
        result_key = f"{question_id}_{answering_model.id}_{parsing_model.id}_rep{answering_replicate}"
    else:
        result_key = f"{question_id}_{answering_model.id}_{parsing_model.id}"

    # Execute verification
    result = run_single_model_verification(
        question_id=task["question_id"],
        question_text=task["question_text"],
        template_code=task["template_code"],
        answering_model=task["answering_model"],
        parsing_model=task["parsing_model"],
        run_name=task["run_name"],
        job_id=task["job_id"],
        answering_replicate=task["answering_replicate"],
        parsing_replicate=task["parsing_replicate"],
        rubric=task["rubric"],
        keywords=task.get("keywords"),
        few_shot_examples=task.get("few_shot_examples"),
        few_shot_enabled=task.get("few_shot_enabled", False),
        abstention_enabled=task.get("abstention_enabled", False),
        deep_judgment_enabled=task.get("deep_judgment_enabled", False),
        rubric_evaluation_strategy=task.get("rubric_evaluation_strategy", "batch"),
        deep_judgment_max_excerpts_per_attribute=task.get("deep_judgment_max_excerpts_per_attribute", 3),
        deep_judgment_fuzzy_match_threshold=task.get("deep_judgment_fuzzy_match_threshold", 0.80),
        deep_judgment_excerpt_retry_attempts=task.get("deep_judgment_excerpt_retry_attempts", 2),
        deep_judgment_search_enabled=task.get("deep_judgment_search_enabled", False),
        deep_judgment_search_tool=task.get("deep_judgment_search_tool", "tavily"),
    )

    return result_key, result


def run_question_verification(
    question_id: str,
    question_text: str,
    template_code: str,
    config: VerificationConfig,
    rubric: Rubric | None = None,
    async_enabled: bool | None = None,
    max_workers: int | None = None,
    keywords: list[str] | None = None,
    few_shot_examples: list[dict[str, str]] | None = None,
) -> dict[str, VerificationResult]:
    """
    Run verification for a single question with all model combinations.

    Args:
        question_id: Unique identifier for the question
        question_text: The question to ask the LLM
        template_code: Python code defining the Answer class
        config: Verification configuration with multiple models
        rubric: Optional rubric for qualitative evaluation
        async_enabled: Enable parallel execution (uses environment default if not provided)
        max_workers: Maximum number of worker threads (uses environment default if not provided)
        keywords: Optional keywords for verification
        few_shot_examples: Optional few-shot examples

    Returns:
        Dictionary of VerificationResult keyed by combination ID
    """
    # Read async settings from parameters or environment
    if async_enabled is None:
        async_enabled = os.getenv("KARENINA_ASYNC_ENABLED", "true").lower() == "true"

    if max_workers is None:
        max_workers_env = os.getenv("KARENINA_ASYNC_MAX_WORKERS")
        max_workers = int(max_workers_env) if max_workers_env else 2

    # Preload embedding model if embedding check is enabled
    try:
        from .tools.embedding_check import _should_use_embedding_check

        if _should_use_embedding_check():
            preload_embedding_model()
    except Exception:
        # If preloading fails, embedding check will handle it gracefully per question
        pass

    # Build list of all verification tasks via combinatorial expansion
    verification_tasks = []

    # Get model configurations
    answering_models = config.answering_models
    parsing_models = config.parsing_models
    replicate_count = config.replicate_count

    for answering_model in answering_models:
        for parsing_model in parsing_models:
            for replicate in range(1, replicate_count + 1):
                # For single replicate, don't include replicate numbers
                answering_replicate = None if replicate_count == 1 else replicate
                parsing_replicate = None if replicate_count == 1 else replicate

                task = _create_verification_task(
                    question_id=question_id,
                    question_text=question_text,
                    template_code=template_code,
                    answering_model=answering_model,
                    parsing_model=parsing_model,
                    run_name=getattr(config, "run_name", None),
                    job_id=getattr(config, "job_id", None),
                    answering_replicate=answering_replicate,
                    parsing_replicate=parsing_replicate,
                    rubric=rubric,
                    keywords=keywords,
                    few_shot_examples=few_shot_examples,
                    few_shot_enabled=config.is_few_shot_enabled(),
                    abstention_enabled=getattr(config, "abstention_enabled", False),
                    deep_judgment_enabled=getattr(config, "deep_judgment_enabled", False),
                    rubric_evaluation_strategy=getattr(config, "rubric_evaluation_strategy", "batch"),
                    deep_judgment_max_excerpts_per_attribute=getattr(
                        config, "deep_judgment_max_excerpts_per_attribute", 3
                    ),
                    deep_judgment_fuzzy_match_threshold=getattr(config, "deep_judgment_fuzzy_match_threshold", 0.80),
                    deep_judgment_excerpt_retry_attempts=getattr(config, "deep_judgment_excerpt_retry_attempts", 2),
                    deep_judgment_search_enabled=getattr(config, "deep_judgment_search_enabled", False),
                    deep_judgment_search_tool=getattr(config, "deep_judgment_search_tool", "tavily"),
                )
                verification_tasks.append(task)

    # Execute tasks (sync or async based on config)
    if async_enabled and len(verification_tasks) > 1:
        # Parallel execution using ThreadPoolExecutor
        task_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and collect futures
            future_to_task = {executor.submit(_execute_verification_task, task): task for task in verification_tasks}

            # Collect results as they complete
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    task_results.append(result)
                except Exception as e:
                    print(f"Warning: Task execution failed: {e}")
                    continue
    else:
        # Sequential execution: simple loop (original behavior)
        task_results = []
        for task in verification_tasks:
            task_result = _execute_verification_task(task)
            task_results.append(task_result)

    # Convert results to dictionary format
    results = {}
    for task_result in task_results:
        if isinstance(task_result, Exception):
            # Handle exceptions from async execution
            print(f"Warning: Task execution failed: {task_result}")
            continue

        result_key, verification_result = task_result
        results[result_key] = verification_result

    # Clean up embedding model cache after job completion
    try:
        from .tools.embedding_check import _should_use_embedding_check

        if _should_use_embedding_check():
            clear_embedding_model_cache()
    except Exception:
        # Cleanup failure is not critical
        pass

    return results
