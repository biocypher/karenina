"""Materialize a VerificationResultSet + Benchmark into an analysis directory.

This module owns: the partition over results vs scenario_results, filename
sanitization and collision handling, benchmark artifact writing, the
force/REPORT.previous.md rules, and the INDEX.md assembly (via indexer).
Per-case rendering is delegated to case_renderer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from karenina.benchmark.error_analysis.case_renderer import (
    render_qa_case,
    render_scenario_case,
)
from karenina.benchmark.error_analysis.exceptions import MaterializationError
from karenina.benchmark.error_analysis.indexer import IndexEntry, build_index_markdown
from karenina.schemas.results.verification_result_set import VerificationResultSet
from karenina.schemas.scenario.state import ScenarioExecutionResult
from karenina.schemas.verification.result import VerificationResult
from karenina.schemas.verification.result_components import VerificationResultMetadata

logger = logging.getLogger(__name__)

_ID_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_-]")

BENCHMARK_DIRNAME = "benchmark"
TEMPLATES_DIRNAME = "templates"
PASSES_DIRNAME = "passes"
FAILURES_DIRNAME = "failures"
CASE_ASSETS_DIRNAME = "case_assets"
QUESTIONS_FILENAME = "questions.jsonl"
RUBRIC_FILENAME = "rubric.json"
BENCHMARK_METADATA_FILENAME = "metadata.md"


def sanitize_id(raw: str) -> str:
    """Replace any character that is not alphanumeric, underscore, or dash with an underscore.

    Args:
        raw: Original identifier (question_id, scenario_id, etc.).

    Returns:
        Sanitized identifier safe for use as a filename component.
    """
    return _ID_SANITIZE_RE.sub("_", raw)


def _hash_suffix(result_id: str) -> str:
    """Compute an 8-char SHA-1 suffix for collision disambiguation."""
    return hashlib.sha1(result_id.encode("utf-8")).hexdigest()[:8]


def case_filename(
    *,
    metadata: VerificationResultMetadata | None = None,
    scenario: ScenarioExecutionResult | None = None,
    monotonic_n: int = 1,
    existing: set[str] | None = None,
) -> str:
    """Generate the on-disk filename for a QA or scenario case.

    QA cases use ``q_{question_id}[__rep_{N}].md``. Scenario cases use
    ``scenario_{scenario_id}__run_{N}.md`` where N is the scenario's
    ``replicate`` if present or a monotonic counter otherwise. On
    collision with ``existing``, append an 8-char SHA-1 suffix derived
    from the result_id (QA) or ``scenario_id:monotonic_n`` (scenario).

    Args:
        metadata: VerificationResultMetadata for a QA case. Mutually
            exclusive with ``scenario``.
        scenario: ScenarioExecutionResult for a scenario case.
        monotonic_n: Fallback scenario run number when
            ``scenario.replicate`` is None.
        existing: Set of filenames already assigned in the current
            analysis directory; triggers the SHA-1 suffix on match.

    Returns:
        The filename (basename only, no directory component).
    """
    if scenario is None and metadata is None:
        raise ValueError("case_filename requires either metadata or scenario")
    existing = existing or set()
    if scenario is not None:
        raw_id = sanitize_id(scenario.scenario_id)
        n = scenario.replicate if scenario.replicate is not None else monotonic_n
        base = f"scenario_{raw_id}__run_{n}.md"
    else:
        # metadata is guaranteed non-None here by the guard at the top;
        # the `assert` is for the type checker only (runtime safety is
        # provided by the ValueError above, which survives python -O).
        assert metadata is not None
        raw_id = sanitize_id(metadata.question_id)
        base = f"q_{raw_id}"
        if metadata.replicate is not None:
            base = f"{base}__rep_{metadata.replicate}"
        base = f"{base}.md"

    if base not in existing:
        return base
    # Collision: append an 8-char SHA-1 suffix of the result_id.
    if metadata is not None:
        suffix = _hash_suffix(metadata.result_id)
    else:
        # Scenario collision: use scenario_id + run number.
        assert scenario is not None
        suffix = _hash_suffix(f"{scenario.scenario_id}:{monotonic_n}")
    stem, _, _ = base.rpartition(".md")
    hashed = f"{stem}__h{suffix}.md"
    if hashed in existing:
        raise MaterializationError(
            "Filename collision after SHA-1 disambiguation.",
            details={"base": base, "hashed": hashed},
        )
    return hashed


def partition_results(
    result_set: VerificationResultSet,
) -> tuple[list[VerificationResult], list[ScenarioExecutionResult]]:
    """Split VerificationResultSet into classical QA results and scenario runs.

    Classical QA results are those whose ``metadata.scenario_id`` is None.
    Scenario runs come directly from ``result_set.scenario_results``.

    Args:
        result_set: The aggregated result set to partition.

    Returns:
        A 2-tuple ``(classical, scenarios)``.

    Raises:
        MaterializationError: If any result carries ``scenario_id`` but
            ``scenario_results`` is None (legacy result set; the
            aggregated per-scenario view is required for rendering).
    """
    scenario_aware = [r for r in result_set.results if r.metadata.scenario_id is not None]
    if scenario_aware and not result_set.scenario_results:
        raise MaterializationError(
            "Result set carries scenario turns but no aggregated scenario_results; "
            "this usually means the run was produced by an older pipeline version. "
            "Re-run the benchmark to produce a result set with scenario_results populated.",
            details={"scenario_turn_count": len(scenario_aware)},
        )
    classical = [r for r in result_set.results if r.metadata.scenario_id is None]
    scenarios: list[Any] = list(result_set.scenario_results or [])
    return classical, scenarios


class ErrorAnalysisMaterializer:
    """Orchestrator that turns a VerificationResultSet + Benchmark into a navigable directory.

    The materializer writes the full analysis tree (benchmark artifacts,
    per-case markdown files bucketed by outcome/category, and INDEX.md).
    It does not launch the analyst agent; that is the facade's job in
    Task 10.

    Attributes:
        max_trace_chars: Optional override of the trace truncation
            threshold applied to every rendered case. When None, the
            case renderer consults the env var
            ``KARENINA_TRACE_TRUNCATION_THRESHOLD`` or its module default.
    """

    def __init__(self, *, max_trace_chars: int | None = None) -> None:
        self.max_trace_chars = max_trace_chars

    def build(
        self,
        result_set: VerificationResultSet,
        checkpoint: Any,
        out_dir: Path,
        *,
        force: bool = False,
    ) -> Path:
        """Materialize the full analysis directory for a verification run.

        Args:
            result_set: The aggregated results to render.
            checkpoint: Benchmark-like object exposing ``name``,
                ``questions``, ``get_question``, ``get_template_source``,
                and ``rubric``.
            out_dir: Destination directory. Created if missing. See
                ``_prepare_out_dir`` for the force/overwrite rules.
            force: If True, overwrite the existing directory, preserving
                any prior ``REPORT.md`` as ``REPORT.previous.md``.

        Returns:
            The resolved ``out_dir`` path.
        """
        self._prepare_out_dir(out_dir, force=force)
        self._checkpoint_name = checkpoint.name
        classical, scenarios = partition_results(result_set)
        self._write_benchmark_artifacts(checkpoint, result_set, out_dir)

        entries: list[IndexEntry] = []
        used_by_bucket: dict[Path, set[str]] = defaultdict(set)

        for result in classical:
            entries.append(self._materialize_qa(result, checkpoint, out_dir, used_by_bucket))

        monotonic_by_scenario: dict[str, int] = defaultdict(int)
        for scenario in scenarios:
            monotonic_by_scenario[scenario.scenario_id] += 1
            entries.append(
                self._materialize_scenario(
                    scenario,
                    checkpoint,
                    out_dir,
                    used_by_bucket,
                    monotonic_n=monotonic_by_scenario[scenario.scenario_id],
                )
            )

        self._write_index(result_set, out_dir, entries)
        # PROMPT.md is written by the facade (Task 10), not the
        # materializer, so it can accept a user-supplied prompt_path.
        return out_dir

    def _prepare_out_dir(self, out_dir: Path, *, force: bool) -> None:
        """Clear ``out_dir`` per the overwrite rules.

        If ``out_dir`` is non-empty and ``force`` is False, raises
        MaterializationError. Otherwise preserves any prior REPORT.md as
        REPORT.previous.md and removes everything else.
        """
        if out_dir.exists() and any(out_dir.iterdir()):
            if not force:
                raise MaterializationError(
                    f"Output directory {out_dir} is not empty; pass force=True to overwrite.",
                    details={"out_dir": str(out_dir)},
                )
            prior_report = out_dir / "REPORT.md"
            if prior_report.exists():
                target = out_dir / "REPORT.previous.md"
                if target.exists():
                    target.unlink()
                prior_report.rename(target)
            for child in out_dir.iterdir():
                if child.name == "REPORT.previous.md":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)

    def _write_benchmark_artifacts(
        self,
        checkpoint: Any,
        result_set: VerificationResultSet,
        out_dir: Path,
    ) -> None:
        """Write questions.jsonl, rubric.json, metadata.md, and templates."""
        bench_dir = out_dir / BENCHMARK_DIRNAME
        bench_dir.mkdir(parents=True, exist_ok=True)
        templates_dir = bench_dir / TEMPLATES_DIRNAME
        templates_dir.mkdir(exist_ok=True)

        # questions.jsonl
        with (bench_dir / QUESTIONS_FILENAME).open("w", encoding="utf-8") as handle:
            for question in checkpoint.questions:
                record = {
                    "id": question.id,
                    "text": question.question,
                    "keywords": list(question.keywords or []),
                    "raw_answer": question.raw_answer,
                }
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                # Copy the template source for this question.
                source = checkpoint.get_template_source(question.id)
                if source is not None:
                    (templates_dir / f"q_{sanitize_id(question.id)}.py").write_text(source, encoding="utf-8")

        # rubric.json
        rubric = checkpoint.rubric
        rubric_dump = rubric.model_dump() if rubric is not None else {}
        (bench_dir / RUBRIC_FILENAME).write_text(
            json.dumps(rubric_dump, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # metadata.md
        answering = sorted({r.metadata.answering.display_string for r in result_set.results})
        parsing = sorted({r.metadata.parsing.display_string for r in result_set.results})
        lines = [
            f"# Benchmark: {checkpoint.name}",
            "",
            "## Run",
            "",
            f"- Answering models: {', '.join(answering) if answering else '-'}",
            f"- Parsing models: {', '.join(parsing) if parsing else '-'}",
            "",
        ]
        (bench_dir / BENCHMARK_METADATA_FILENAME).write_text("\n".join(lines), encoding="utf-8")

    def _materialize_qa(
        self,
        result: VerificationResult,
        checkpoint: Any,
        out_dir: Path,
        used_by_bucket: dict[Path, set[str]],
    ) -> IndexEntry:
        """Render a single QA case and return the matching IndexEntry."""
        failure = result.metadata.failure
        bucket = out_dir / PASSES_DIRNAME if failure is None else out_dir / FAILURES_DIRNAME / failure.category.value
        bucket.mkdir(parents=True, exist_ok=True)
        filename = case_filename(metadata=result.metadata, existing=used_by_bucket[bucket])
        used_by_bucket[bucket].add(filename)
        case_path = bucket / filename

        question = checkpoint.get_question(result.metadata.question_id)
        template_source = checkpoint.get_template_source(result.metadata.question_id)
        template_link: str | None = None
        if question is not None:
            template_link = f"../../{BENCHMARK_DIRNAME}/{TEMPLATES_DIRNAME}/q_{sanitize_id(question.id)}.py"

        assets_dir = out_dir / CASE_ASSETS_DIRNAME / case_path.stem / "traces" / "artifacts"
        body = render_qa_case(
            result,
            template_source=template_source,
            template_link=template_link,
            artifacts_dir=assets_dir,
            max_trace_chars=self.max_trace_chars,
        )
        case_path.write_text(body, encoding="utf-8")

        case_id = case_path.stem
        rel = case_path.relative_to(out_dir).as_posix()
        if failure is None:
            return IndexEntry(
                case_id=case_id,
                bucket_path=rel,
                outcome="pass",
                category="",
                group="",
                stage="",
                reason="",
            )
        return IndexEntry(
            case_id=case_id,
            bucket_path=rel,
            outcome="failure",
            category=failure.category.value,
            group=failure.group.value,
            stage=failure.stage,
            reason=failure.reason or "",
        )

    def _materialize_scenario(
        self,
        scenario: Any,
        checkpoint: Any,
        out_dir: Path,
        used_by_bucket: dict[Path, set[str]],
        *,
        monotonic_n: int,
    ) -> IndexEntry:
        """Render a scenario run, bucketing by the first failing turn's category."""
        failing = next(
            (r for r in scenario.turn_results if r.metadata.failure is not None),
            None,
        )
        if failing is None:
            bucket = out_dir / PASSES_DIRNAME
            failure_category = ""
            failure_group = ""
            failure_stage = ""
            failure_reason = ""
            outcome = "pass"
        else:
            assert failing.metadata.failure is not None
            bucket = out_dir / FAILURES_DIRNAME / failing.metadata.failure.category.value
            failure_category = failing.metadata.failure.category.value
            failure_group = failing.metadata.failure.group.value
            failure_stage = failing.metadata.failure.stage
            failure_reason = failing.metadata.failure.reason or ""
            outcome = "failure"

        bucket.mkdir(parents=True, exist_ok=True)
        filename = case_filename(
            scenario=scenario,
            monotonic_n=monotonic_n,
            existing=used_by_bucket[bucket],
        )
        used_by_bucket[bucket].add(filename)
        case_path = bucket / filename

        template_sources: dict[str, str | None] = {}
        template_links: dict[str, str] = {}
        for turn in scenario.turn_results:
            qid = turn.metadata.question_id
            template_sources[qid] = checkpoint.get_template_source(qid)
            template_links[qid] = f"../../{BENCHMARK_DIRNAME}/{TEMPLATES_DIRNAME}/q_{sanitize_id(qid)}.py"

        assets_dir = out_dir / CASE_ASSETS_DIRNAME / case_path.stem / "traces" / "artifacts"
        body = render_scenario_case(
            scenario,
            template_sources=template_sources,
            template_links=template_links,
            artifacts_dir=assets_dir,
            max_trace_chars=self.max_trace_chars,
            monotonic_n=monotonic_n,
        )
        case_path.write_text(body, encoding="utf-8")

        return IndexEntry(
            case_id=case_path.stem,
            bucket_path=case_path.relative_to(out_dir).as_posix(),
            outcome=outcome,
            category=failure_category,
            group=failure_group,
            stage=failure_stage,
            reason=failure_reason,
        )

    def _write_index(
        self,
        result_set: VerificationResultSet,
        out_dir: Path,
        entries: list[IndexEntry],
    ) -> None:
        """Compose and write INDEX.md from the entries collected during build."""
        answering = sorted({r.metadata.answering.display_string for r in result_set.results})
        answering_model = ", ".join(answering) if answering else "-"
        timestamp = sorted({r.metadata.timestamp for r in result_set.results})[-1] if result_set.results else ""
        benchmark_name = getattr(self, "_checkpoint_name", "(unknown)")
        index_md = build_index_markdown(
            benchmark_name=benchmark_name,
            answering_model=answering_model,
            run_timestamp=timestamp,
            entries=entries,
        )
        (out_dir / "INDEX.md").write_text(index_md, encoding="utf-8")
