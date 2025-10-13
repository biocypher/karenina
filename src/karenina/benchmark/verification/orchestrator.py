"""Orchestration logic for multi-model verification."""

import asyncio
from typing import Any, Literal, cast

from ...schemas.rubric_class import Rubric
from ...utils.async_utils import AsyncConfig, execute_with_config
from ..models import ModelConfig, VerificationConfig, VerificationResult
from .embedding_utils import clear_embedding_model_cache, preload_embedding_model
from .runner import run_single_model_verification


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
    )

    return result_key, result


def run_question_verification(
    question_id: str,
    question_text: str,
    template_code: str,
    config: VerificationConfig,
    rubric: Rubric | None = None,
    async_config: AsyncConfig | None = None,
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
        async_config: Optional async configuration (uses environment default if not provided)

    Returns:
        Dictionary of VerificationResult keyed by combination ID
    """
    if async_config is None:
        async_config = AsyncConfig.from_env()

    # Preload embedding model if embedding check is enabled
    try:
        from .embedding_utils import _should_use_embedding_check

        if _should_use_embedding_check():
            preload_embedding_model()
    except Exception:
        # If preloading fails, embedding check will handle it gracefully per question
        pass

    # Build list of all verification tasks
    verification_tasks = []

    # Handle legacy single model config
    if hasattr(config, "answering_model_provider") and config.answering_model_provider:
        # Legacy single model mode - create single model configs and handle replicates
        answering_model = ModelConfig(
            id="answering-legacy",
            model_provider=config.answering_model_provider or "",
            model_name=config.answering_model_name or "",
            temperature=config.answering_temperature or 0.1,
            interface=cast(Literal["langchain", "openrouter", "manual"], config.answering_interface or "langchain"),
            system_prompt=config.answering_system_prompt
            or "You are an expert assistant. Answer the question accurately and concisely.",
        )

        parsing_model = ModelConfig(
            id="parsing-legacy",
            model_provider=config.parsing_model_provider or "",
            model_name=config.parsing_model_name or "",
            temperature=config.parsing_temperature or 0.1,
            interface=cast(Literal["langchain", "openrouter", "manual"], config.parsing_interface or "langchain"),
            system_prompt=config.parsing_system_prompt
            or "You are a validation assistant. Parse and validate responses against the given Pydantic template.",
        )

        # Create tasks for legacy mode with replicates
        for replicate in range(1, getattr(config, "replicate_count", 1) + 1):
            # For single replicate, don't include replicate numbers
            answering_replicate = None if getattr(config, "replicate_count", 1) == 1 else replicate
            parsing_replicate = None if getattr(config, "replicate_count", 1) == 1 else replicate

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
            )
            verification_tasks.append(task)

    else:
        # New multi-model mode
        answering_models = getattr(config, "answering_models", [])
        parsing_models = getattr(config, "parsing_models", [])
        replicate_count = getattr(config, "replicate_count", 1)

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
                    )
                    verification_tasks.append(task)

    # Execute tasks (sync or async based on config)
    if async_config.enabled and len(verification_tasks) > 1:
        # Async execution: chunk and parallelize
        try:
            task_results = asyncio.run(
                execute_with_config(
                    items=verification_tasks,
                    sync_function=_execute_verification_task,
                    config=async_config,
                )
            )
        except Exception as e:
            # Fallback to sync execution if async fails
            print(f"Warning: Async execution failed, falling back to sync: {e}")
            task_results = [_execute_verification_task(task) for task in verification_tasks]
    else:
        # Sync execution: simple loop (original behavior)
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
        from .embedding_utils import _should_use_embedding_check

        if _should_use_embedding_check():
            clear_embedding_model_cache()
    except Exception:
        # Cleanup failure is not critical
        pass

    return results
