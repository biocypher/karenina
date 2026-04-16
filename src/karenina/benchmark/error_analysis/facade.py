"""Facade for the error-analysis feature: glue materializer + prompt + launcher."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from karenina.benchmark.error_analysis.exceptions import LauncherNoOutputError
from karenina.benchmark.error_analysis.launcher import (
    ErrorAnalystLauncher,
    get_launcher,
)
from karenina.benchmark.error_analysis.materializer import ErrorAnalysisMaterializer
from karenina.benchmark.error_analysis.prompt_io import (
    PromptContext,
    resolve_and_write_prompt,
)

logger = logging.getLogger(__name__)

_CLAUDE_CODE_MODULE = "karenina.benchmark.error_analysis.launchers.claude_code"


def analyze_errors(
    results: Any,
    checkpoint: Any,
    out_dir: Path,
    *,
    prompt_path: Path | None = None,
    launcher: str | ErrorAnalystLauncher = "prepare-only",
    launcher_kwargs: dict[str, object] | None = None,
    max_trace_chars: int | None = None,
    force: bool = False,
) -> Path:
    """Materialize a run into out_dir and invoke a launcher to produce REPORT.md.

    Args:
        results: a VerificationResultSet or a path to its JSON export.
        checkpoint: a Benchmark or a path to its JSON-LD file.
        out_dir: destination directory for the analysis (created if absent).
        prompt_path: optional user-supplied PROMPT.md source.
        launcher: registered launcher name, or an instance conforming to
            ErrorAnalystLauncher.
        launcher_kwargs: forwarded to launcher.run().
        max_trace_chars: override for KARENINA_TRACE_TRUNCATION_THRESHOLD
            just for this analysis.
        force: allow writing into a non-empty out_dir.

    Returns:
        Path to out_dir / "REPORT.md" on success.

    Raises:
        MaterializationError: on any materialization failure.
        LauncherNotFoundError: if launcher is an unknown name.
        LauncherUnavailableError: if the launcher is registered but its
            dependency (for example, the claude binary) is missing.
        LauncherExecutionError: if the launcher subprocess failed.
        LauncherNoOutputError: if the launcher claimed success but
            REPORT.md is absent.
    """
    results_obj = _load_results(results)
    checkpoint_obj = _load_checkpoint(checkpoint)

    materializer = ErrorAnalysisMaterializer(max_trace_chars=max_trace_chars)
    materializer.build(results_obj, checkpoint_obj, out_dir, force=force)

    context = _prompt_context(checkpoint_obj, results_obj)
    resolve_and_write_prompt(prompt_path=prompt_path, out_dir=out_dir, context=context)

    resolved_launcher = _resolve_launcher(launcher)
    resolved_launcher.run(out_dir, **(launcher_kwargs or {}))

    report_path = out_dir / "REPORT.md"
    if not report_path.exists():
        raise LauncherNoOutputError(out_dir)
    logger.info("Analysis complete: %s", report_path)
    return report_path


def _load_results(results: Any) -> Any:
    from karenina.schemas.results.verification_result_set import VerificationResultSet

    if isinstance(results, Path):
        return VerificationResultSet.model_validate_json(results.read_text(encoding="utf-8"))
    return results


def _load_checkpoint(checkpoint: Any) -> Any:
    from karenina.benchmark.benchmark import Benchmark

    if isinstance(checkpoint, Path):
        return Benchmark.load(checkpoint)
    return checkpoint


def _prompt_context(checkpoint_obj: Any, result_set: Any) -> PromptContext:
    answering = sorted({r.metadata.answering.display_string for r in result_set.results})
    failed = [r for r in result_set.results if r.metadata.failure is not None]
    categories = sorted({r.metadata.failure.category.value for r in failed})
    return PromptContext(
        benchmark_name=getattr(checkpoint_obj, "name", "(unknown)"),
        answering_model=", ".join(answering) if answering else "(unknown)",
        total=len(result_set.results),
        passed=len(result_set.results) - len(failed),
        failed=len(failed),
        failure_categories=categories,
    )


def _resolve_launcher(launcher: str | ErrorAnalystLauncher) -> ErrorAnalystLauncher:
    if isinstance(launcher, str):
        if launcher == "claude-code":
            import importlib

            importlib.import_module(_CLAUDE_CODE_MODULE)  # triggers registration
        return get_launcher(launcher)()
    return launcher
