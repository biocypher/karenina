"""Run one BixBench task with Karenina.

This walkthrough keeps one representative condition from the full experiment:
GLM-5.1 answers through Claude Code, GLM-5.1 parses through DeepAgents, and a
fresh GLM-5.1 TaskEval rubric judges the resulting trace. Existing results are
an explicit alternative through ``--results``.

Run from the Karenina repository root:

    uv run python -m paper.bixbench_harness_comparison.simplified
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path
from statistics import mean
from typing import cast

from pydantic import SecretStr

from karenina.benchmark import (
    Benchmark,
    ModelConfig,
    ResultsIOManager,
    RowContext,
    VerificationConfig,
    evaluate_rubric_on_results,
)
from karenina.schemas import AgentRuntimeConfig
from karenina.schemas.verification import VerificationResult
from paper.bixbench_harness_comparison.config import GLM_MODEL
from paper.bixbench_harness_comparison.rubrics import (
    BURDEN_FIELDS,
    failure_burden_rubric,
    reconstruct_burdens,
    trace_text,
)
from paper.common.bootstrap import bootstrap, input_path
from paper.config import (
    BIXBENCH_BENCHMARK_JSONLD,
    BIXBENCH_SIMPLIFIED_OUTPUT_DIR,
    ZAI_ANTHROPIC_ENDPOINT,
    ZAI_OPENAI_ENDPOINT,
)


def _key() -> SecretStr:
    value = os.environ.get("ZAI_API_KEY")
    if not value:
        raise RuntimeError("Live evaluation requires ZAI_API_KEY")
    return SecretStr(value)


def _container(image: str, *, read_only: bool = False) -> AgentRuntimeConfig:
    """Create a Docker runtime from Karenina's typed schema."""

    return AgentRuntimeConfig(
        backend="container",
        container_runtime="docker",
        container_image=image,
        access_mode="read_only" if read_only else "read_write",
        read_max_bytes=20_000 if read_only else None,
    )


def build_models_and_verification(
    output_dir: Path,
    timeout: int,
) -> tuple[VerificationConfig, ModelConfig]:
    """Build one answerer, parser, judge, and verification configuration."""

    key = _key()
    answerer = ModelConfig(
        id="glm-5-1-claude-code",
        model_name=GLM_MODEL,
        interface="claude_agent_sdk",
        temperature=0.0,
        system_prompt=(
            "You are a biostatistician. Work in the provided workspace, inspect "
            "the data, answer every item with code, and save scripts and results."
        ),
        anthropic_base_url=ZAI_ANTHROPIC_ENDPOINT,
        anthropic_api_key=key,
        agent_runtime=_container(
            os.environ.get(
                "KARENINA_BIX_CLAUDE_CONTAINER_IMAGE",
                "karenina-bixbench-claude:latest",
            )
        ),
        agent_timeout=timeout,
    )
    parser = ModelConfig(
        id="glm-5-1-parser",
        model_name=GLM_MODEL,
        model_provider="openai",
        interface="langchain_deep_agents",
        temperature=0.0,
        endpoint_base_url=ZAI_OPENAI_ENDPOINT,
        endpoint_api_key=key,
        extra_kwargs={"endpoint_base_url_mode": "raw"},
        agent_runtime=_container(
            os.environ.get(
                "KARENINA_BIX_DEEPAGENTS_CONTAINER_IMAGE",
                "karenina-bixbench:latest",
            ),
            read_only=True,
        ),
        agent_timeout=timeout,
    )
    judge = ModelConfig(
        id="glm-5-1-failure-burden",
        model_name=GLM_MODEL,
        model_provider="openai",
        interface="langchain_deep_agents",
        temperature=0.0,
        max_tokens=8192,
        endpoint_base_url=ZAI_OPENAI_ENDPOINT,
        endpoint_api_key=key,
        extra_kwargs={"endpoint_base_url_mode": "raw"},
        agent_runtime=AgentRuntimeConfig(backend="filesystem", access_mode="read_only"),
        agent_timeout=600,
        request_timeout=600.0,
    )
    verification = VerificationConfig(
        answering_models=[answerer],
        parsing_models=[parser],
        evaluation_mode="template_only",
        replicate_count=1,
        agentic_parsing=True,
        agentic_parsing_trigger="dynamic",
        agentic_judge_context="trace_and_workspace",
        agentic_parsing_max_turns=60,
        agentic_parsing_timeout=float(timeout),
        agentic_parsing_materialize_trace=True,
        agentic_parsing_persist_trace=True,
        workspace_copy=True,
        workspace_cleanup=True,
        workspace_output_mode="full",
        workspace_output_dir=output_dir / "workspaces",
        allow_partial_trace_scoring=True,
        request_timeout=float(timeout),
    )
    return verification, judge


def prepare_output_dir(output_dir: Path) -> None:
    """Refuse to overwrite artifacts from an earlier live run."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --output-dir so earlier live artifacts are retained."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def run_live_task(
    benchmark_path: Path,
    config: VerificationConfig,
    question_id: str,
    output_dir: Path,
) -> list[VerificationResult]:
    """Load and run one task through the ``Benchmark`` interface."""

    # ## 1. Load the benchmark
    benchmark = Benchmark.load(benchmark_path, workspace_root=benchmark_path.parent)

    # ## 2. Run verification
    # This calls the answerer and parser models configured above.
    print(f"[1] running live benchmark task {question_id}")
    result_set = benchmark.run_verification(
        config,
        question_ids=[question_id],
        run_name="simplified-live-evaluation",
    )
    rows = list(result_set.results)
    if not rows:
        raise RuntimeError("The live benchmark run returned no results")

    # ResultsIOManager serializes and reloads the result set.
    results_path = output_dir / "results.json"
    results_path.write_text(
        ResultsIOManager.export_to_json(
            {row.metadata.result_id: row for row in rows}
        ),
        encoding="utf-8",
    )
    print(f"[2] saved inspectable live results to {results_path}")
    return rows


def load_stored_rows(results: Path, limit: int) -> list[VerificationResult]:
    """Load a stored-results alternative with ``ResultsIOManager``."""

    iterator = cast(
        Iterator[VerificationResult],
        ResultsIOManager.iter_from_json(results),
    )
    rows = list(iterator)[:limit]
    if not rows:
        raise RuntimeError("The result export contains no rows")
    print(f"[1] loaded {len(rows)} validated BixBench results")
    return rows


def graded_accuracy(rows: list[VerificationResult]) -> float:
    """Average the field scores already produced by template evaluation."""

    scores = [
        float(score)
        for row in rows
        if row.template is not None
        for score in (row.template.field_scores or {}).values()
        if score is not None
    ]
    if not scores:
        raise RuntimeError("The selected rows contain no verified field scores")
    return mean(scores)


def judge_traces(
    rows: list[VerificationResult],
    judge: ModelConfig,
) -> list[dict[str, object]]:
    """Freshly judge saved traces through Karenina's TaskEval-backed facade."""

    judgments = evaluate_rubric_on_results(
        rows,
        failure_burden_rubric(model=judge),
        judge,
        text_selector=trace_text,
        row_context=lambda row: RowContext(question=row.metadata.question_text),
        collapse_parser_siblings=False,
        max_workers=1,
    )
    output: list[dict[str, object]] = []
    for judgment in judgments:
        if judgment.error or not judgment.scores:
            raise RuntimeError(judgment.error or "Failure-burden judge returned no verdict")
        verdict = reconstruct_burdens(judgment.scores)
        output.append(
            {
                "result_id": judgment.representative_result_id,
                **verdict.model_dump(),
            }
        )
    return output


def main() -> None:
    """Run one compact live evaluation with Karenina."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, nargs="?")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--question-id", default="bix-18")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=BIXBENCH_SIMPLIFIED_OUTPUT_DIR)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    bootstrap(args.verbose)

    if args.results is not None and args.benchmark is not None:
        parser.error("provide either a benchmark checkpoint or --results, not both")
    if args.summary_only and args.results is None:
        parser.error("--summary-only requires --results")

    benchmark_path = (
        args.benchmark
        if args.benchmark is not None
        else input_path(BIXBENCH_BENCHMARK_JSONLD)
        if args.results is None
        else None
    )

    if args.summary_only:
        rows = load_stored_rows(args.results, args.limit)
        print(f"[2] deterministic graded field accuracy: {graded_accuracy(rows):.3f}")
        return

    prepare_output_dir(args.output_dir)
    config, judge = build_models_and_verification(args.output_dir, args.timeout)
    if args.results is not None:
        rows = load_stored_rows(args.results, args.limit)
    else:
        assert benchmark_path is not None
        rows = run_live_task(benchmark_path, config, args.question_id, args.output_dir)
    score_step = 2 if args.results is not None else 3
    print(f"[{score_step}] deterministic graded field accuracy: {graded_accuracy(rows):.3f}")

    # ## 3. Attach and run the live failure-burden rubric
    print(f"[{score_step + 1}] judging {len(rows)} trace(s) live through TaskEval")
    burdens = judge_traces(rows, judge)
    burden_path = args.output_dir / "failure_burdens.json"
    burden_path.write_text(
        json.dumps(burdens, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    totals = [
        sum(cast(int, row[field]) for field in BURDEN_FIELDS)
        for row in burdens
    ]
    print(f"[{score_step + 2}] mean total failure burden: {mean(totals):.2f}")
    print(f"[{score_step + 3}] saved judgments to {burden_path}")
    print(
        "\nKarenina flow: Benchmark -> VerificationConfig -> ResultsIOManager "
        "-> evaluate_rubric_on_results (TaskEval)"
    )


if __name__ == "__main__":
    main()
