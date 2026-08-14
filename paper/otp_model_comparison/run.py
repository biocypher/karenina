"""Run the full Open Targets parametric and MCP model comparison.

By default this executes both benchmark arms and therefore calls every
configured answerer and parser model. Pass ``--reuse-stored-results`` to load
the archived exports and regenerate only the deterministic analysis tables.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from karenina.benchmark import (
    Benchmark,
    ProgressiveFileSink,
    ResultsIOManager,
    managed_run_directory,
)
from karenina.schemas.verification import VerificationResult
from karenina.utils.mcp import McpServerSpec, managed_mcp_servers
from paper.common.bootstrap import bootstrap, input_path
from paper.config import (
    MODEL_COMPARISON_OUTPUT_ROOT,
    OTP_BENCHMARK_JSONLD,
    OTP_MCP_RESULTS,
    OTP_PARAMETRIC_RESULTS,
)
from paper.otp_model_comparison.analysis import build_long_form, write_analysis
from paper.otp_model_comparison.config import ANSWERER_NAMES, build_config

logger = logging.getLogger(__name__)

def _iter_results(path: Path) -> Iterator[VerificationResult]:
    return cast(Iterator[VerificationResult], ResultsIOManager.iter_from_json(path))


def _mcp_specs(model_names: Sequence[str], log_dir: Path) -> list[McpServerSpec]:
    host = os.environ.get("KARENINA_PAPER_MCP_HOST", "127.0.0.1")
    base_port = int(os.environ.get("KARENINA_PAPER_MCP_BASE_PORT", "8765"))
    local_source = os.environ.get("KARENINA_PAPER_OTP_MCP_SOURCE")
    specs: list[McpServerSpec] = []
    for offset, model_name in enumerate(model_names):
        port = base_port + offset
        command: tuple[str, ...]
        if local_source:
            command = (
                "uv",
                "run",
                "--directory",
                local_source,
                "otp-mcp",
                "--transport",
                "http",
                "--host",
                host,
                "--port",
                str(port),
                "--timeout",
                "180",
            )
        else:
            command = (
                "uvx",
                "--from",
                "git+https://github.com/opentargets/open-targets-platform-mcp",
                "otp-mcp",
                "--transport",
                "http",
                "--host",
                host,
                "--port",
                str(port),
                "--timeout",
                "180",
            )
        specs.append(
            McpServerSpec(
                name=f"Open Targets MCP for {model_name}",
                command=command,
                host=host,
                port=port,
                log_path=log_dir / f"otp_mcp_{model_name}.log",
                startup_timeout=300.0,
            )
        )
    return specs


def _execute_arm(
    benchmark: Benchmark,
    benchmark_path: Path,
    output_root: Path,
    regime: str,
    *,
    model_names: Sequence[str],
    mcp_urls: dict[str, str] | None,
    limit: int | None,
    replicates: int,
    workers: int,
) -> Path:
    config = build_config(
        mcp_urls,
        model_names=model_names,
        replicates=replicates,
        workers=workers,
    )
    question_ids = sorted(benchmark.get_question_ids())
    if limit is not None:
        question_ids = question_ids[:limit]
    run_name = f"otp_{regime}"
    with managed_run_directory(
        output_root / "runs",
        run_name,
        benchmark_path=benchmark_path,
        config=config,
        run_name=run_name,
        selected_question_count=len(question_ids),
    ) as run_directory:
        sink = ProgressiveFileSink(
            output_path=run_directory.results_path,
            config=config,
            benchmark_path=str(benchmark_path),
            global_rubric=benchmark.get_global_rubric(),
        )
        benchmark.run_verification(
            config,
            question_ids=question_ids,
            run_name=run_name,
            sink=sink,
        )
        return run_directory.results_path


def run(
    output_root: Path,
    *,
    reuse_stored_results: bool,
    limit: int | None = None,
    model_names: Sequence[str] = ANSWERER_NAMES,
    replicates: int = 3,
    workers: int = 16,
    arm: str = "both",
) -> None:
    """Run or load both arms and regenerate their deterministic tables."""
    if arm not in {"both", "parametric", "mcp"}:
        raise ValueError(f"Unknown comparison arm: {arm}")
    output_root.mkdir(parents=True, exist_ok=True)
    result_paths: dict[str, Path] = {}
    if reuse_stored_results:
        if arm in {"both", "parametric"}:
            result_paths["parametric"] = input_path(OTP_PARAMETRIC_RESULTS)
        if arm in {"both", "mcp"}:
            result_paths["mcp"] = input_path(OTP_MCP_RESULTS)
    else:
        benchmark_path = input_path(OTP_BENCHMARK_JSONLD)
        benchmark = Benchmark.load(benchmark_path)
        logger.warning(
            "Executing both comparison arms with live answerer and parser models: %s",
            ", ".join(model_names),
        )
        if arm in {"both", "parametric"}:
            result_paths["parametric"] = _execute_arm(
                benchmark,
                benchmark_path,
                output_root,
                "parametric",
                model_names=model_names,
                mcp_urls=None,
                limit=limit,
                replicates=replicates,
                workers=workers,
            )
        if arm in {"both", "mcp"}:
            specs = _mcp_specs(model_names, output_root / "mcp_logs")
            with managed_mcp_servers(specs) as servers:
                urls = dict(zip(model_names, (server.url for server in servers), strict=True))
                result_paths["mcp"] = _execute_arm(
                    benchmark,
                    benchmark_path,
                    output_root,
                    "mcp",
                    model_names=model_names,
                    mcp_urls=urls,
                    limit=limit,
                    replicates=replicates,
                    workers=workers,
                )
    parametric = build_long_form(
        _iter_results(result_paths["parametric"]) if "parametric" in result_paths else [],
        "parametric",
    )
    mcp = build_long_form(
        _iter_results(result_paths["mcp"]) if "mcp" in result_paths else [],
        "mcp",
    )
    write_analysis(parametric, mcp, output_root / "analysis")
    (output_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "completed_at": datetime.now(UTC).isoformat(),
                "mode": "reuse_stored_results" if reuse_stored_results else "live",
                "arm": arm,
                "result_inputs": {name: str(path) for name, path in result_paths.items()},
                "analysis_directory": str(output_root / "analysis"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Parse command-line options and run the model comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-stored-results", action="store_true")
    parser.add_argument("--output-root", type=Path, default=MODEL_COMPARISON_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--arm", choices=("both", "parametric", "mcp"), default="both")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    bootstrap(args.verbose)
    run(
        args.output_root,
        reuse_stored_results=args.reuse_stored_results,
        limit=args.limit,
        replicates=args.replicates,
        workers=args.workers,
        arm=args.arm,
    )


if __name__ == "__main__":
    main()
