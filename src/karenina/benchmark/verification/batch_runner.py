"""Batch verification runner with combinatorial task expansion.

This module provides the core verification orchestration logic that can be used
standalone (without karenina-server) to run combinatorial verification tests.
"""

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from karenina.schemas.entities import Rubric
from karenina.schemas.entities.rubric import DynamicRubric
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import (
    FinishedTemplate,
    VerificationConfig,
    VerificationResult,
)
from karenina.schemas.verification.config import DEFAULT_ASYNC_ENABLED
from karenina.utils.answer_cache import AnswerTraceCache

from .utils.cache_helpers import (
    extract_answer_data_from_result,
    generate_answer_cache_key,
)
from .utils.resource_helpers import cleanup_resources
from .utils.storage_helpers import auto_save_results
from .utils.task_helpers import (
    extract_feature_flags,
    merge_dynamic_rubrics_for_task,
    merge_rubrics_for_task,
    resolve_few_shot_for_task,
)

logger = logging.getLogger(__name__)


def _apply_request_timeout(model: Any, pipeline_timeout: float | None) -> Any:
    """Stamp pipeline-level request_timeout onto a ModelConfig if not already set.

    Returns the original model if no change is needed, or a copy with the timeout applied.
    """
    if pipeline_timeout is not None and model.request_timeout is None:
        return model.model_copy(update={"request_timeout": pipeline_timeout})
    return model


# ============================================================================
# Config Helpers
# ============================================================================


def _apply_retry_config(model: Any, retry_policy: Any | None) -> Any:
    """Stamp pipeline-level retry_policy onto a ModelConfig if not already set.

    Returns the original model if no change is needed, or a copy with the policy applied.
    """
    if retry_policy is not None and model.retry_policy is None:
        return model.model_copy(update={"retry_policy": retry_policy})
    return model


def _normalize_answerer_limits(
    value: int | dict[str, int] | None,
    answering_models: list[Any],
) -> dict[str, int] | None:
    """Normalize ``VerificationConfig.answerer_concurrency_limits`` to a dict or None.

    - ``None``: returned as ``None``.
    - ``int``: broadcast to every model in ``answering_models`` that has an ``id``.
    - ``dict``: returned as a shallow copy. Keys not present in
      ``answering_models`` are still retained (in case the caller wants them)
      but logged at WARNING so operator typos are visible.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return {m.id: value for m in answering_models if getattr(m, "id", None)}

    known_ids = {m.id for m in answering_models if getattr(m, "id", None)}
    unknown = sorted(set(value) - known_ids)
    if unknown:
        logger.warning(
            "answerer_concurrency_limits contains ids not in answering_models: %s",
            unknown,
        )
    return dict(value)


def _resolve_task_ordering(config: VerificationConfig) -> str:
    """Resolve ``task_ordering='auto'`` based on the number of distinct answerer identities.

    Returns the resolved strategy name. Non-``auto`` values pass through
    unchanged. A single INFO log records the resolution for operator
    visibility.
    """
    if config.task_ordering != "auto":
        return config.task_ordering

    from karenina.schemas.verification.model_identity import ModelIdentity

    answerer_keys = {
        ModelIdentity.from_model_config(m, role="answering").canonical_key for m in config.answering_models
    }
    if len(answerer_keys) > 1:
        logger.info(
            "task_ordering=auto -> distribute_answerers (%d distinct answerer identities)",
            len(answerer_keys),
        )
        return "distribute_answerers"

    logger.info("task_ordering=auto -> prefix_cache (single answerer identity)")
    return "prefix_cache"


def _apply_task_ordering(task_queue: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    """Apply the given task-ordering strategy to ``task_queue`` and return the result.

    Mutates the input list for in-place strategies (``prefix_cache``,
    ``random``) and returns the same reference. Returns a new list for
    ``distribute_answerers``. ``generation_order`` returns the input as-is.
    """
    from .utils.task_helpers import interleave_by_answerer, model_sort_key

    if strategy == "prefix_cache":
        task_queue.sort(
            key=lambda t: (
                model_sort_key(t["answering_model"]),
                t["question_id"],
                model_sort_key(t["parsing_model"]),
                t.get("replicate") or 0,
            )
        )
        return task_queue
    if strategy == "distribute_answerers":
        return interleave_by_answerer(task_queue)
    if strategy == "random":
        import random

        random.shuffle(task_queue)
        return task_queue
    # "generation_order": no-op, preserve loop order
    return task_queue


# ============================================================================
# Task Queue Generation
# ============================================================================


def generate_task_queue(
    templates: list[FinishedTemplate],
    config: VerificationConfig,
    global_rubric: Rubric | None = None,
    global_dynamic_rubric: DynamicRubric | None = None,
    run_name: str | None = None,
    workspace_root: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Generate complete task queue via combinatorial expansion.

    Expansion formula:
      Templates × Answering Models × Parsing Models × Replicates = Total Tasks

    Args:
        templates: List of finished templates to verify
        config: Verification configuration with models and settings
        global_rubric: Optional global rubric for evaluation
        global_dynamic_rubric: Optional global dynamic rubric for evaluation
        run_name: Optional name for this verification run

    Returns:
        List of task dictionaries with all arguments for verification
    """
    tasks = []
    skip_triples = config.skip_triples
    n_skipped = 0
    if skip_triples is not None:
        from karenina.schemas.verification.model_identity import ModelIdentity

    for template in templates:
        # Prepare rubric for this question
        rubric, trait_provenance = merge_rubrics_for_task(global_rubric, template, config)

        # Prepare dynamic rubric for this question
        dynamic_rubric = merge_dynamic_rubrics_for_task(global_dynamic_rubric, template, config)

        # Resolve few-shot examples
        few_shot = resolve_few_shot_for_task(template, config)

        # Expand over model combinations
        for ans_model_raw in config.answering_models:
            for parse_model_raw in config.parsing_models:
                # Stamp pipeline-level request_timeout onto models that don't have their own
                ans_model = _apply_request_timeout(ans_model_raw, config.request_timeout)
                parse_model = _apply_request_timeout(parse_model_raw, config.request_timeout)
                # Stamp pipeline-level retry_policy onto models that don't have their own
                ans_model = _apply_retry_config(ans_model, config.retry_policy)
                parse_model = _apply_retry_config(parse_model, config.retry_policy)
                # Expand over replicates
                for rep in range(1, config.replicate_count + 1):
                    # For single replicate, don't include replicate numbers
                    replicate = None if config.replicate_count == 1 else rep

                    if skip_triples is not None:
                        ans_key = ModelIdentity.from_model_config(ans_model, role="answering").canonical_key
                        parse_key = ModelIdentity.from_model_config(parse_model, role="parsing").canonical_key
                        if (template.question_id, ans_key, parse_key, replicate) in skip_triples:
                            n_skipped += 1
                            continue

                    task = {
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
                        "dynamic_rubric": dynamic_rubric,
                        "trait_provenance": trait_provenance,
                        "keywords": template.keywords,
                        "question_workspace_path": template.workspace_path,
                        "few_shot_examples": few_shot,
                        # Feature flags (from config)
                        **extract_feature_flags(config),
                    }
                    # Replay layer (see karenina/replay)
                    task.update(
                        {
                            "replay_store": config.replay_store,
                            "replay_parse_on_hydration_mismatch": config.replay_parse_on_hydration_mismatch,
                        }
                    )
                    # Benchmark-level workspace_root overrides config
                    if workspace_root is not None:
                        task["workspace_root"] = workspace_root
                    tasks.append(task)

    if skip_triples is not None and n_skipped:
        logger.info("generate_task_queue: skipped %d tasks covered by prior_results", n_skipped)

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
    # Generate unique result key using canonical_key for full identity (interface+model+tools)
    from karenina.schemas.verification.model_identity import ModelIdentity

    from .runner import run_single_model_verification

    answering_key = ModelIdentity.from_model_config(task["answering_model"], role="answering").canonical_key
    parsing_key = ModelIdentity.from_model_config(task["parsing_model"], role="parsing").canonical_key

    key_parts = [
        task["question_id"],
        answering_key,
        parsing_key,
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
            dynamic_rubric=task.get("dynamic_rubric"),
            keywords=task.get("keywords"),
            raw_answer=task.get("raw_answer"),
            few_shot_examples=task.get("few_shot_examples"),
            few_shot_enabled=task.get("few_shot_enabled", False),
            abstention_enabled=task.get("abstention_enabled", False),
            sufficiency_enabled=task.get("sufficiency_enabled", False),
            deep_judgment_mode=task.get("deep_judgment_mode", "disabled"),
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
            # Agentic parsing
            agentic_parsing=task.get("agentic_parsing", False),
            agentic_judge_context=task.get("agentic_judge_context", "workspace_only"),
            agentic_parsing_max_turns=task.get("agentic_parsing_max_turns", 15),
            agentic_parsing_timeout=task.get("agentic_parsing_timeout", 120.0),
            agentic_parsing_materialize_trace=task.get("agentic_parsing_materialize_trace", False),
            agentic_parsing_persist_trace=task.get("agentic_parsing_persist_trace", False),
            workspace_root=task.get("workspace_root"),
            workspace_copy=task.get("workspace_copy", True),
            workspace_cleanup=task.get("workspace_cleanup", True),
            question_workspace_path=task.get("question_workspace_path"),
            # Agentic rubric evaluation
            agentic_rubric_strategy=task.get("agentic_rubric_strategy", "individual"),
            agentic_rubric_parallel=task.get("agentic_rubric_parallel", False),
            # Trait provenance
            trait_provenance=task.get("trait_provenance"),
            cached_answer_data=cached_answer_data,
            # Replay layer
            replay_store=task.get("replay_store"),
            replay_parse_on_hydration_mismatch=task.get("replay_parse_on_hydration_mismatch", "fall_through"),
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
    global_dynamic_rubric: DynamicRubric | None = None,
    async_enabled: bool | None = None,
    max_workers: int | None = None,
    storage_url: str | None = None,
    benchmark_name: str | None = None,
    progress_callback: Callable[[int, int, VerificationResult | None], None] | None = None,
    workspace_root: Path | None = None,
    sink: Any = None,
) -> VerificationResultSet:
    """
    Run batch verification with combinatorial expansion.

    This is the main entry point for standalone verification runs.

    Args:
        templates: List of finished templates to verify
        config: Verification configuration with models and settings
        run_name: Optional name for this run (auto-generated if not provided)
        global_rubric: Optional global rubric for evaluation
        global_dynamic_rubric: Optional global dynamic rubric for evaluation
        async_enabled: Whether to run in parallel (defaults to KARENINA_ASYNC_ENABLED env var)
        max_workers: Maximum parallel workers (defaults to KARENINA_ASYNC_MAX_WORKERS env var)
        storage_url: Optional database URL for auto-save
        benchmark_name: Optional benchmark name for auto-save
        progress_callback: Optional callback(current, total, result | None) for progress updates
                          Called before starting each task with preview result
        workspace_root: Root directory for task workspaces (from Benchmark).
        sink: Optional :class:`ResultSink` for progressive save / crash recovery.
            When present, its ``completed_triples()`` are merged into
            ``config.skip_triples`` before task expansion, ``on_result`` is
            called for each completed task, and ``on_finalize`` is called
            exactly once before returning. A :class:`VerificationBatchError`
            raised by the executor is caught and converted into a partial
            ``VerificationResultSet`` return; without a sink, the exception
            propagates (back-compat).

    Returns:
        VerificationResultSet containing all verification results
    """
    # Guard: parsing_only is designed for TaskEval, which bypasses the batch
    # runner entirely. Using it here would silently produce 0 tasks because
    # the task queue iterates over answering_models (which is empty).
    if config.parsing_only:
        raise ValueError(
            "parsing_only=True is not supported in the batch verification path. "
            "Use TaskEval for evaluating pre-recorded responses, or provide "
            "answering_models for the standard Benchmark pipeline."
        )

    # Generate run name if not provided
    if run_name is None:
        import uuid

        run_name = f"run_{uuid.uuid4().hex[:8]}"

    # Determine async mode
    if async_enabled is None:
        async_enabled = os.getenv("KARENINA_ASYNC_ENABLED", str(DEFAULT_ASYNC_ENABLED).lower()).lower() == "true"

    # Determine max workers: explicit arg > config > env var > default
    if max_workers is None:
        max_workers = config.async_max_workers  # Uses env var fallback internally

    # Merge sink-completed triples into skip_triples so resume does not
    # re-execute work a ProgressiveFileSink / DBSink already persisted.
    if sink is not None:
        sink_triples = sink.completed_triples()
        if sink_triples:
            existing = config.skip_triples or frozenset()
            config = config.model_copy(update={"skip_triples": frozenset(existing | sink_triples)})
            logger.info("Sink reports %d already-completed triples; merged into skip_triples", len(sink_triples))

    # Generate task queue
    logger.info(f"Generating task queue for {len(templates)} templates...")
    task_queue = generate_task_queue(
        templates=templates,
        config=config,
        global_rubric=global_rubric,
        global_dynamic_rubric=global_dynamic_rubric,
        run_name=run_name,
        workspace_root=workspace_root,
    )

    # Inform the sink of the planned manifest and build a result-dispatch
    # callback that wraps the caller's progress_callback so the sink sees
    # every completed result without the caller having to wire it manually.
    if sink is not None:
        from karenina.utils.progressive_save import TaskIdentifier

        full_manifest = [TaskIdentifier.from_task_dict(task).to_key() for task in task_queue]
        sink.on_start(full_manifest, config)

        caller_progress = progress_callback

        def _sink_progress_adapter(current: int, total: int, result: VerificationResult | None) -> None:
            if result is not None:
                try:
                    sink.on_result(result)
                except Exception:  # noqa: BLE001
                    logger.warning("Sink on_result raised; continuing", exc_info=True)
            if caller_progress is not None:
                caller_progress(current, total, result)

        progress_callback = _sink_progress_adapter

    # Apply task ordering strategy
    effective = _resolve_task_ordering(config)
    task_queue = _apply_task_ordering(task_queue, effective)

    # Log execution plan
    logger.info(f"Starting verification: {len(task_queue)} tasks ({'parallel' if async_enabled else 'sequential'})")

    # Execute tasks using the VerificationExecutor
    from .executor import ExecutorConfig, VerificationExecutor

    answerer_limits = _normalize_answerer_limits(config.answerer_concurrency_limits, config.answering_models)
    if answerer_limits:
        logger.info(
            "answerer_concurrency_limits active: %s",
            answerer_limits,
        )
    executor = VerificationExecutor(
        parallel=async_enabled,
        config=ExecutorConfig(
            max_workers=max_workers,
            max_requeue_count=config.max_requeue_count,
            answerer_concurrency_limits=answerer_limits,
        ),
    )

    from karenina.exceptions import VerificationBatchError

    all_complete = True
    try:
        results = executor.run_batch(task_queue, progress_callback)
    except VerificationBatchError as exc:
        # Preserve partial results so the sink-enabled path can still write
        # a partial export and keep resume state. Without a sink, re-raise
        # to preserve back-compat for existing callers.
        if sink is None:
            raise
        results = exc.partial_results or {}
        all_complete = False
        logger.warning(
            "Batch completed with %d partial result(s) after failures: %s",
            len(results),
            exc,
        )

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

    # Finalize sink. On partial runs the sink keeps its sidecars so a later
    # `--resume` can continue. On full completion it writes the canonical
    # export and deletes sidecars.
    if sink is not None:
        all_complete = all_complete and len(results) == len(task_queue)
        try:
            sink.on_finalize(all_complete=all_complete)
        except Exception:  # noqa: BLE001
            logger.warning("Sink on_finalize raised; continuing", exc_info=True)

    # Convert dict to VerificationResultSet
    result_list = list(results.values())
    result_set = VerificationResultSet(results=result_list)

    logger.info(f"Verification complete: {len(result_set)} results")

    # Clean up lingering resources (HTTP clients, event loops, etc.)
    cleanup_resources()

    return result_set
