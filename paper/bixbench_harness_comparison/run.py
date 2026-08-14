"""Run the complete BixBench agent-harness comparison.

The default path executes the selected matrix from scratch and then freshly
judges all answer traces with GLM-5.1. Archived result exports and archived
judgments are available only through explicit reuse flags.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from karenina.benchmark import (
    Benchmark,
    PostHocJudgment,
    ProgressiveFileSink,
    RowContext,
    evaluate_rubric_on_results,
    managed_run_directory,
)
from karenina.schemas.verification import VerificationResult
from paper.bixbench_harness_comparison.analysis import (
    ResultSource,
    build_result_tables,
    iter_source_results,
    load_stored_burdens,
    normalize_archive_harness,
    normalize_archive_model,
    write_analysis,
)
from paper.bixbench_harness_comparison.config import (
    ExperimentCondition,
    RuntimeLabel,
    build_config,
    failure_burden_model,
    probe_condition_endpoints,
    selected_conditions,
)
from paper.bixbench_harness_comparison.rubrics import (
    BURDEN_FIELDS,
    failure_burden_rubric,
    reconstruct_burdens,
    trace_text,
)
from paper.common.bootstrap import bootstrap, input_path
from paper.config import (
    BIXBENCH_ARCHIVED_BURDENS,
    BIXBENCH_ARCHIVED_RUNS,
    BIXBENCH_BENCHMARK_JSONLD,
    BIXBENCH_OUTPUT_ROOT,
)

logger = logging.getLogger(__name__)
_REPLICATE_PATTERN = re.compile(r"^rep(\d+)$")


def _archived_condition(run_name: str) -> tuple[ExperimentCondition, int] | None:
    """Parse immutable run names without exposing them in output labels."""

    parts = run_name.split("__")
    if len(parts) < 4:
        return None
    match = _REPLICATE_PATTERN.fullmatch(parts[-1])
    if match is None:
        return None
    profile = parts[1]
    if profile.startswith("qwen122-"):
        model = normalize_archive_model("qwen")
    elif profile.startswith("glm51-"):
        model = normalize_archive_model("glm")
    elif profile.startswith("opus46-"):
        model = normalize_archive_model("opus")
    else:
        return None
    harness = normalize_archive_harness("csdk" if "-csdk" in profile else "da")
    return ExperimentCondition(model=model, harness=harness), int(match.group(1))


def discover_archived_sources(
    runs_dir: Path,
    conditions: list[ExperimentCondition],
) -> list[ResultSource]:
    """Find the curated 3-replicate exports for selected matrix cells."""

    selected = set(conditions)
    found: dict[tuple[ExperimentCondition, int], ResultSource] = {}
    for path in sorted(runs_dir.glob("*/results.json")):
        parsed = _archived_condition(path.parent.name)
        if parsed is None or parsed[0] not in selected:
            continue
        if parsed in found:
            raise ValueError(
                "Archived BixBench runs contain duplicate selected matrix cells. "
                "Point KARENINA_PAPER_DATA at the curated 18-run deposit."
            )
        found[parsed] = ResultSource(
            condition=parsed[0],
            path=path,
            replicate_override=parsed[1],
        )
    missing = [
        (condition.slug, replicate)
        for condition in conditions
        for replicate in range(1, 4)
        if (condition, replicate) not in found
    ]
    if missing:
        raise FileNotFoundError(f"Archived BixBench matrix cells are missing: {missing}")
    return [found[key] for key in sorted(found, key=lambda item: (item[0].slug, item[1]))]


def _execute_condition(
    benchmark: Benchmark,
    benchmark_path: Path,
    output_root: Path,
    condition: ExperimentCondition,
    *,
    runtime: RuntimeLabel,
    replicates: int,
    workers: int,
    timeout: int | None,
    limit: int | None,
    question_ids: tuple[str, ...] | None = None,
) -> ResultSource:
    available_question_ids = set(benchmark.get_question_ids())
    if question_ids is None:
        selected_question_ids = sorted(available_question_ids)
        if limit is not None:
            selected_question_ids = selected_question_ids[:limit]
    else:
        missing = sorted(set(question_ids) - available_question_ids)
        if missing:
            raise ValueError(f"Unknown BixBench question ids: {missing}")
        selected_question_ids = list(question_ids)
    condition_timeout = timeout or (
        2400 if condition.model == "Claude Opus 4.6" else 900
    )
    config = build_config(
        condition,
        runtime=runtime,
        replicates=replicates,
        workers=workers,
        timeout=condition_timeout,
        workspace_output_dir=output_root / "workspaces" / condition.slug,
    )
    with managed_run_directory(
        output_root / "runs",
        condition.slug,
        benchmark_path=benchmark_path,
        config=config,
        run_name=condition.slug,
        selected_question_count=len(selected_question_ids),
    ) as run_directory:
        sink = ProgressiveFileSink(
            output_path=run_directory.results_path,
            config=config,
            benchmark_path=str(benchmark_path),
            global_rubric=benchmark.get_global_rubric(),
        )
        benchmark.run_verification(
            config,
            question_ids=selected_question_ids,
            run_name=condition.slug,
            sink=sink,
        )
        return ResultSource(condition=condition, path=run_directory.results_path)


def _fresh_burdens(
    sources: list[ResultSource],
    *,
    workers: int,
    timeout: int,
) -> list[dict[str, object]]:
    rows = [
        (source, result)
        for source in sources
        for result in iter_source_results(source)
    ]
    identities = {
        result.metadata.result_id: (
            source.condition,
            source.replicate_override or result.metadata.replicate or 1,
            result.metadata.question_id,
        )
        for source, result in rows
    }
    results_by_id = {result.metadata.result_id: result for _source, result in rows}

    def evaluate(results: list[VerificationResult]) -> Iterator[PostHocJudgment]:
        return evaluate_rubric_on_results(
            results,
            failure_burden_rubric(timeout),
            failure_burden_model(timeout),
            text_selector=trace_text,
            row_context=lambda result: RowContext(question=result.metadata.question_text),
            collapse_parser_siblings=False,
            max_workers=workers,
        )

    judgments_by_id = {
        judgment.representative_result_id: judgment
        for judgment in evaluate(list(results_by_id.values()))
    }
    failed_ids = [
        result_id
        for result_id, judgment in judgments_by_id.items()
        if result_id is not None and (judgment.error or not judgment.scores)
    ]
    if failed_ids:
        logger.warning(
            "Retrying %d failed GLM-5.1 burden judgments once.",
            len(failed_ids),
        )
        judgments_by_id.update(
            {
                judgment.representative_result_id: judgment
                for judgment in evaluate([results_by_id[result_id] for result_id in failed_ids])
            }
        )
    output: list[dict[str, object]] = []
    for judgment in judgments_by_id.values():
        if judgment.error or not judgment.scores:
            raise RuntimeError(judgment.error or "Failure-burden judge returned no verdict")
        result_id = judgment.representative_result_id
        if result_id is None or result_id not in identities:
            raise RuntimeError("Failure-burden judgment cannot be linked to its source result")
        condition, replicate, task = identities[result_id]
        verdict = reconstruct_burdens(judgment.scores)
        output.append(
            {
                "model": condition.model,
                "harness": condition.harness,
                "replicate": replicate,
                "task": task,
                **{field: getattr(verdict, field) for field in BURDEN_FIELDS},
                "evidence": verdict.evidence,
            }
        )
    return output


def run(
    output_root: Path,
    *,
    reuse_stored_results: bool = False,
    reuse_stored_judgments: bool = False,
    model: str = "all",
    harness: str = "both",
    runtime: RuntimeLabel = "docker",
    replicates: int = 3,
    workers: int = 4,
    burden_workers: int = 4,
    timeout: int | None = None,
    burden_timeout: int = 600,
    limit: int | None = None,
    question_ids: tuple[str, ...] | None = None,
) -> None:
    """Execute or explicitly reuse results, then regenerate all tables."""

    if reuse_stored_judgments:
        reuse_stored_results = True
    if reuse_stored_results and question_ids is not None:
        raise ValueError("question_ids is available only for fresh execution")
    conditions = selected_conditions(model, harness)
    output_root.mkdir(parents=True, exist_ok=True)
    if reuse_stored_results:
        sources = discover_archived_sources(input_path(BIXBENCH_ARCHIVED_RUNS), conditions)
    else:
        logger.warning(
            "Executing %d BixBench conditions from scratch. This calls answerers and GLM-5.1 parsers.",
            len(conditions),
        )
        probe_condition_endpoints(conditions)
        benchmark_path = input_path(BIXBENCH_BENCHMARK_JSONLD)
        benchmark = Benchmark.load(
            benchmark_path,
            workspace_root=benchmark_path.parent,
        )
        sources = [
            _execute_condition(
                benchmark,
                benchmark_path,
                output_root,
                condition,
                runtime=runtime,
                replicates=replicates,
                workers=workers,
                timeout=timeout,
                limit=limit,
                question_ids=question_ids,
            )
            for condition in conditions
        ]
    field_rows, task_rows = build_result_tables(sources)
    if reuse_stored_judgments:
        selected = set(conditions)
        burden_rows = [
            row
            for row in load_stored_burdens(input_path(BIXBENCH_ARCHIVED_BURDENS))
            if ExperimentCondition(
                model=row["model"],  # type: ignore[arg-type]
                harness=row["harness"],  # type: ignore[arg-type]
            )
            in selected
        ]
    else:
        logger.warning(
            "Freshly judging %d stored answer traces with the GLM-5.1 subscription route.",
            len(task_rows),
        )
        burden_rows = _fresh_burdens(
            sources,
            workers=burden_workers,
            timeout=burden_timeout,
        )
    write_analysis(output_root / "analysis", field_rows, task_rows, burden_rows)
    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "completed_at": datetime.now(UTC).isoformat(),
                "mode": "fresh" if not reuse_stored_results else "reuse_stored_results",
                "judgments": "stored" if reuse_stored_judgments else "fresh",
                "conditions": [
                    {"model": condition.model, "harness": condition.harness}
                    for condition in conditions
                ],
                "result_inputs": [str(source.path) for source in sources],
                "model_calls": not reuse_stored_judgments,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Parse command-line options and run the experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-stored-results", action="store_true")
    parser.add_argument("--reuse-stored-judgments", action="store_true")
    parser.add_argument("--model", choices=("all", "qwen", "glm", "opus"), default="all")
    parser.add_argument(
        "--harness",
        choices=("both", "claude-code", "deepagents"),
        default="both",
    )
    parser.add_argument(
        "--runtime",
        choices=("docker", "singularity", "apptainer", "host"),
        default="docker",
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--burden-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--burden-timeout", type=int, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path, default=BIXBENCH_OUTPUT_ROOT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    bootstrap(args.verbose)
    run(
        args.output_root,
        reuse_stored_results=args.reuse_stored_results,
        reuse_stored_judgments=args.reuse_stored_judgments,
        model=args.model,
        harness=args.harness,
        runtime=args.runtime,
        replicates=args.replicates,
        workers=args.workers,
        burden_workers=args.burden_workers,
        timeout=args.timeout,
        burden_timeout=args.burden_timeout,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
