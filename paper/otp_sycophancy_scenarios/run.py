"""Run the full Open Targets sycophancy scenarios and sidecar checks.

The default command replays the experiment's exact first QA replicate, then
calls the configured models for every continuation, parser, guardrail, and
sidecar judgment. ``--reuse-stored-results`` skips scenario model calls but
reruns GPT-OSS sidecars. ``--reuse-stored-judgments`` is fully offline and
implies stored scenario results.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from karenina.benchmark import Benchmark, ProgressiveFileSink, ResultsIOManager, managed_run_directory
from karenina.schemas.results import VerificationResultSet
from karenina.utils.mcp import ManagedMcpServer, managed_mcp_server
from paper.common.bootstrap import bootstrap, input_path
from paper.config import (
    OTP_ADVERSARIAL_CSV,
    OTP_BENCHMARK_JSONLD,
    OTP_MCP_RESULTS,
    OTP_PARAMETRIC_RESULTS,
    SYCOPHANCY_ABSTENTION_JUDGMENTS,
    SYCOPHANCY_CAVE_GROUNDING_JUDGMENTS,
    SYCOPHANCY_CAVE_REGEX_JUDGMENTS,
    SYCOPHANCY_DEFINITIVE_RESULTS,
    SYCOPHANCY_OUTPUT_DIR,
)
from paper.otp_sycophancy_scenarios.analysis import write_analysis
from paper.otp_sycophancy_scenarios.config import (
    ANSWERERS,
    DIFFICULTIES,
    FRAMINGS,
    REGIMES,
    answerer_model,
    build_config,
    guardrail_model,
    mcp_spec,
    reference_parser_model,
    rubric_judge_model,
)
from paper.otp_sycophancy_scenarios.handover import token_light_guardrail_handover
from paper.otp_sycophancy_scenarios.replay import build_ask_replay
from paper.otp_sycophancy_scenarios.rubrics import (
    evaluate_abstention,
    evaluate_caves,
    load_archived_jsonl,
    write_sidecars,
)
from paper.otp_sycophancy_scenarios.scenarios import (
    build_scenario_benchmark,
    hydrate_scenario_models,
)

logger = logging.getLogger(__name__)

def _checkpoint_path(output_dir: Path, difficulty: str, framing: str) -> Path:
    return output_dir / "checkpoints" / f"otp_sycophancy_{difficulty}_{framing}.jsonld"


def _build_checkpoints(output_dir: Path) -> dict[tuple[str, str], Path]:
    """Build and validate the four scenario-graph variants."""
    paths: dict[tuple[str, str], Path] = {}
    parser = reference_parser_model()
    baseline_guardrail = guardrail_model(ANSWERERS[0])
    for difficulty in DIFFICULTIES:
        for framing in FRAMINGS:
            benchmark = build_scenario_benchmark(
                input_path(OTP_BENCHMARK_JSONLD),
                input_path(OTP_ADVERSARIAL_CSV),
                difficulty=difficulty,
                framing=framing,
                parser_model=parser,
                guardrail_model=baseline_guardrail,
            )
            path = _checkpoint_path(output_dir, difficulty, framing)
            path.parent.mkdir(parents=True, exist_ok=True)
            benchmark.save(path)
            Benchmark.load(path)
            paths[(difficulty, framing)] = path
    return paths


def _select_scenarios(benchmark: Benchmark, limit: int | None) -> Benchmark:
    """Apply an optional recovery limit without changing the default matrix."""
    if limit is None:
        return benchmark
    selected = {scenario.name for scenario in benchmark.get_scenarios()[:limit]}
    for scenario in list(benchmark.get_scenarios()):
        if scenario.name not in selected:
            benchmark.remove_scenario(scenario.name)
    if not selected:
        raise ValueError("Scenario limit selected no rows")
    return benchmark


def _token_light_handover(benchmark: Benchmark) -> Benchmark:
    """Replace only the MCP adversarial-to-guardrail handover."""
    for scenario in benchmark.get_scenarios():
        for edge in scenario.edges:
            if edge.source == "adversarial" and edge.target == "guardrail_check":
                edge.handover = None
                edge.handover_callable = token_light_guardrail_handover
    return benchmark


def _require_completed_scenarios(result_set: VerificationResultSet) -> None:
    """Raise when a scenario terminates before producing its outcomes."""
    incomplete = [
        result
        for result in result_set.scenario_results
        if result.status != "completed"
    ]
    if not incomplete:
        return
    preview = []
    for result in incomplete[:5]:
        failure = result.terminal_failure
        reason = failure.reason if failure else "no terminal failure recorded"
        preview.append(f"{result.scenario_id} ({result.status}): {reason}")
    raise RuntimeError("Scenario execution did not complete: " + "; ".join(preview))


def _run_cell(
    output_dir: Path,
    checkpoint_path: Path,
    *,
    answerer: str,
    regime: str,
    difficulty: str,
    framing: str,
    mcp_url: str | None,
    limit: int | None,
    workers: int,
) -> tuple[VerificationResultSet, Path, list[dict[str, object]]]:
    """Execute one crossed cell with strict ask replay and progressive saving."""
    live_answerer = answerer_model(answerer, mcp_url=mcp_url if regime == "mcp" else None)
    benchmark = hydrate_scenario_models(
        _select_scenarios(Benchmark.load(checkpoint_path), limit),
        parser_model=reference_parser_model(),
        guardrail_model=guardrail_model(answerer),
    )
    if regime == "mcp":
        benchmark = _token_light_handover(benchmark)
    qa_path = input_path(
        OTP_MCP_RESULTS
        if regime == "mcp"
        else OTP_PARAMETRIC_RESULTS
    )
    replay = build_ask_replay(benchmark, qa_path, answerer=answerer, regime=regime)
    config = build_config(live_answerer, replay.store, workers=workers)
    run_name = f"sycophancy_{answerer}_{regime}_{difficulty}_{framing}"
    with managed_run_directory(
        output_dir / "runs",
        run_name,
        benchmark_path=checkpoint_path,
        config=config,
        run_name=run_name,
        selected_question_count=len(replay.benchmark.get_scenarios()),
    ) as run_directory:
        sink = ProgressiveFileSink.open_or_resume(
            run_directory.results_path,
            config,
            str(checkpoint_path),
            global_rubric=replay.benchmark.get_global_rubric(),
        )
        result_set = replay.benchmark.run_verification(
            config,
            run_name=run_name,
            async_enabled=True,
            sink=sink,
        )
        result_path = run_directory.results_path
    _require_completed_scenarios(result_set)
    result_set.metadata.update(
        {
            "answerer": answerer,
            "regime": regime,
            "difficulty": difficulty,
            "framing": framing,
            "checkpoint": str(checkpoint_path),
            "replay_source": str(qa_path),
            "source_file": str(result_path),
        }
    )
    for row in replay.exclusions:
        row.update({"answerer": answerer, "regime": regime, "difficulty": difficulty, "framing": framing})
    return result_set, result_path, replay.exclusions


def _archive_paths(root: Path) -> Iterator[Path]:
    """Yield only definitive scenario result exports."""
    yield from sorted(path for path in root.rglob("*.json") if path.parent.name == "results")


def _limited_result_set(result_set: VerificationResultSet, limit: int | None) -> VerificationResultSet:
    if limit is None or not result_set.scenario_results:
        return result_set
    scenarios = list(result_set.scenario_results)[:limit]
    scenario_ids = {scenario.scenario_id for scenario in scenarios}
    return VerificationResultSet(
        results=[result for result in result_set.results if result.metadata.scenario_id in scenario_ids],
        scenario_results=scenarios,
        metadata=dict(result_set.metadata),
    )


def _load_archived_cells(
    *,
    answerers: Sequence[str],
    regimes: Sequence[str],
    difficulties: Sequence[str],
    framings: Sequence[str],
    limit: int | None,
) -> list[tuple[VerificationResultSet, Path]]:
    """Load selected definitive exports only through ResultsIOManager."""
    cells: list[tuple[VerificationResultSet, Path]] = []
    for path in _archive_paths(input_path(SYCOPHANCY_DEFINITIVE_RESULTS)):
        result_set = ResultsIOManager.load_result_set_from_json(path)
        metadata = result_set.metadata
        regime = "parametric" if metadata.get("regime") == "nomcp" else metadata.get("regime")
        if (
            metadata.get("answerer") not in answerers
            or regime not in regimes
            or metadata.get("difficulty") not in difficulties
            or metadata.get("framing") not in framings
        ):
            continue
        result_set.metadata.update({"regime": regime, "source_file": str(path)})
        cells.append((_limited_result_set(result_set, limit), path))
    selected_cells = len(answerers) * len(regimes) * len(difficulties) * len(framings)
    if len(cells) != selected_cells:
        raise ValueError(f"Selected {selected_cells} archive cells but loaded {len(cells)}")
    return cells


def _filter_archived_sidecars(
    rows: list[dict[str, object]],
    *,
    answerers: Sequence[str],
    regimes: Sequence[str],
    difficulties: Sequence[str],
    framings: Sequence[str],
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row.get("answerer") in answerers
        and ("regime" not in row or row.get("regime") in regimes)
        and ("difficulty" not in row or row.get("difficulty") in difficulties)
        and ("framing" not in row or row.get("framing") in framings)
    ]


def run(
    output_dir: Path,
    *,
    reuse_stored_results: bool,
    reuse_stored_judgments: bool,
    answerers: Sequence[str] = ANSWERERS,
    regimes: Sequence[str] = REGIMES,
    difficulties: Sequence[str] = DIFFICULTIES,
    framings: Sequence[str] = FRAMINGS,
    limit: int | None = None,
    workers: int = 8,
) -> None:
    """Run the selected matrix and its sidecars, fresh by default."""
    if reuse_stored_judgments:
        reuse_stored_results = True
    output_dir.mkdir(parents=True, exist_ok=True)
    result_paths: list[Path] = []
    replay_exclusions: list[dict[str, object]] = []
    if reuse_stored_results:
        loaded = _load_archived_cells(
            answerers=answerers,
            regimes=regimes,
            difficulties=difficulties,
            framings=framings,
            limit=limit,
        )
        cells = [result_set for result_set, _path in loaded]
        analysis_cells: list[tuple[VerificationResultSet, Path | None]] = list(loaded)
        result_paths = [path for _result_set, path in loaded]
    else:
        checkpoints = _build_checkpoints(output_dir)
        cells = []
        analysis_cells = []
        mcp_context: Any = (
            managed_mcp_server(mcp_spec(output_dir / "mcp_logs" / "otp_mcp.log"))
            if "mcp" in regimes
            else nullcontext(None)
        )
        with mcp_context as server:
            mcp_url = server.url if isinstance(server, ManagedMcpServer) else None
            for answerer in answerers:
                for regime in regimes:
                    for difficulty in difficulties:
                        for framing in framings:
                            result_set, path, exclusions = _run_cell(
                                output_dir,
                                checkpoints[(difficulty, framing)],
                                answerer=answerer,
                                regime=regime,
                                difficulty=difficulty,
                                framing=framing,
                                mcp_url=mcp_url,
                                limit=limit,
                                workers=workers,
                            )
                            cells.append(result_set)
                            analysis_cells.append((result_set, path))
                            result_paths.append(path)
                            replay_exclusions.extend(exclusions)
    write_analysis(analysis_cells, output_dir / "analysis", replay_exclusions=replay_exclusions)

    if reuse_stored_judgments:
        abstention = _filter_archived_sidecars(
            load_archived_jsonl(input_path(SYCOPHANCY_ABSTENTION_JUDGMENTS)),
            answerers=answerers, regimes=regimes, difficulties=difficulties, framings=framings,
        )
        cave_regex = _filter_archived_sidecars(
            load_archived_jsonl(input_path(SYCOPHANCY_CAVE_REGEX_JUDGMENTS)),
            answerers=answerers, regimes=regimes, difficulties=difficulties, framings=framings,
        )
        cave_grounding = _filter_archived_sidecars(
            load_archived_jsonl(input_path(SYCOPHANCY_CAVE_GROUNDING_JUDGMENTS)),
            answerers=answerers, regimes=regimes, difficulties=difficulties, framings=framings,
        )
    else:
        judge = rubric_judge_model()
        abstention = evaluate_abstention(cells, judge, workers=workers)
        cave_regex, cave_grounding = evaluate_caves(cells, judge, workers=workers)
    write_sidecars(output_dir / "sidecars", abstention, cave_regex, cave_grounding)

    manifest = {
        "schema_version": 1,
        "completed_at": datetime.now(UTC).isoformat(),
        "mode": "reuse_stored_judgments" if reuse_stored_judgments else "reuse_stored_results" if reuse_stored_results else "fresh",
        "scenario_model_calls": not reuse_stored_results,
        "sidecar_model_calls": not reuse_stored_judgments,
        "answerers": list(answerers),
        "regimes": list(regimes),
        "difficulties": list(difficulties),
        "framings": list(framings),
        "per_cell_limit": limit,
        "result_inputs": [str(path) for path in result_paths],
        "analysis_directory": str(output_dir / "analysis"),
        "sidecar_directory": str(output_dir / "sidecars"),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Parse command-line options and run the full scenario workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-stored-results", action="store_true")
    parser.add_argument("--reuse-stored-judgments", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=SYCOPHANCY_OUTPUT_DIR)
    parser.add_argument("--answerer", action="append", choices=ANSWERERS)
    parser.add_argument("--regime", action="append", choices=REGIMES)
    parser.add_argument("--difficulty", action="append", choices=DIFFICULTIES)
    parser.add_argument("--framing", action="append", choices=FRAMINGS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    bootstrap(args.verbose)
    run(
        args.output_dir,
        reuse_stored_results=args.reuse_stored_results,
        reuse_stored_judgments=args.reuse_stored_judgments,
        answerers=args.answerer or ANSWERERS,
        regimes=args.regime or REGIMES,
        difficulties=args.difficulty or DIFFICULTIES,
        framings=args.framing or FRAMINGS,
        limit=args.limit,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
