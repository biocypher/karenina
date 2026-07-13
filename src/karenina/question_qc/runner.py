"""Batch runner for question quality control."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .agent_adapter import build_qc_agent
from .config import QcConfig
from .loop import QcLoop
from .models import QcQuestion, QcResult, QcResultSet
from .prompts import load_evidence_context

logger = logging.getLogger(__name__)


def _default_async_enabled() -> bool:
    return os.getenv("KARENINA_ASYNC_ENABLED", "true").lower() in ("true", "1", "yes")


async def _evaluate_one(
    question: QcQuestion,
    config: QcConfig,
    *,
    evidence_context: str,
    run_name: str | None,
) -> QcResult:
    rt = config.runtime
    common = dict(
        tool_time_buffer_seconds=rt.tool_time_buffer_seconds,
        exclude_tool_name_substrings=rt.exclude_tool_name_substrings,
    )
    proposer = build_qc_agent(config.proposer, **common)
    validator = build_qc_agent(config.validator, **common)
    reviewer = build_qc_agent(config.reviewer, **common)
    loop = QcLoop(
        proposer,
        validator,
        reviewer,
        max_attempts=rt.max_attempts,
        invalid_output_retries=rt.invalid_output_retries,
        evidence_context=evidence_context,
        investigation_seconds=rt.investigation_seconds,
        wrap_up_seconds=rt.wrap_up_seconds,
        conclusion_seconds=rt.conclusion_seconds,
        exclude_tool_time=rt.exclude_tool_time,
        exclude_tool_name_substrings=rt.exclude_tool_name_substrings,
    )
    result = await loop.evaluate(question)
    if run_name is not None:
        result = result.model_copy(update={"run_name": run_name})
    return result


def _evaluate_one_sync(
    question: QcQuestion,
    config: QcConfig,
    *,
    evidence_context: str,
    run_name: str | None,
) -> QcResult:
    return asyncio.run(
        _evaluate_one(question, config, evidence_context=evidence_context, run_name=run_name)
    )


def run_qc_batch(
    questions: list[QcQuestion],
    config: QcConfig,
    *,
    run_name: str | None = None,
    async_enabled: bool | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> QcResultSet:
    """Run QC over a list of questions."""
    if run_name is None:
        run_name = f"qc_{uuid.uuid4().hex[:8]}"

    if async_enabled is None:
        async_enabled = _default_async_enabled()

    evidence_context = load_evidence_context(config.evidence_context_path)
    results: list[QcResult] = []
    total = len(questions)
    if total == 0:
        return QcResultSet(results=[], run_name=run_name)

    if progress_callback:
        progress_callback(0.0, f"Starting QC for {total} question(s)")

    if not async_enabled or total == 1:
        for index, question in enumerate(questions):
            result = _evaluate_one_sync(
                question, config, evidence_context=evidence_context, run_name=run_name
            )
            results.append(result)
            if progress_callback:
                progress_callback(
                    (index + 1) / total * 100.0,
                    f"QC {index + 1}/{total}: {question.question_id} → {result.terminal_status}",
                )
    else:
        max_workers = config.runtime.async_max_workers
        if max_workers is None:
            max_workers = int(os.getenv("KARENINA_ASYNC_MAX_WORKERS", "2"))
        max_workers = max(1, min(max_workers, total))
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _evaluate_one_sync,
                    question,
                    config,
                    evidence_context=evidence_context,
                    run_name=run_name,
                ): question
                for question in questions
            }
            for future in as_completed(futures):
                question = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("QC failed for %s", question.question_id)
                    result = QcResult(
                        question_id=question.question_id,
                        terminal_status="error",
                        error_stage="runner",
                        error_message=str(exc),
                        run_name=run_name,
                    )
                results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(
                        completed / total * 100.0,
                        f"QC {completed}/{total}: {question.question_id} → {result.terminal_status}",
                    )

    by_id = {r.question_id: r for r in results}
    ordered = [by_id[q.question_id] for q in questions if q.question_id in by_id]
    return QcResultSet(results=ordered, run_name=run_name)
