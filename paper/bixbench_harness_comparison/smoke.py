"""Run one live BixBench task through a selected subscription route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karenina.benchmark import ResultsIOManager
from karenina.schemas.results.failure import FailureGroup
from paper.bixbench_harness_comparison.run import run
from paper.common.bootstrap import bootstrap
from paper.config import BIXBENCH_OUTPUT_ROOT


def _require_operational_smoke_success(output_root: Path) -> None:
    """Reject infrastructure failures while retaining content-level outcomes."""
    manifest = json.loads((output_root / "run_manifest.json").read_text())
    failures = []
    for result_path in manifest["result_inputs"]:
        result_set = ResultsIOManager.load_result_set_from_json(result_path)
        for result in result_set.results:
            failure = result.metadata.failure
            if failure is not None and failure.group != FailureGroup.CONTENT:
                failures.append(
                    f"{result.metadata.question_id}: "
                    f"{failure.category.value} at {failure.stage}: {failure.reason}"
                )
    if failures:
        raise RuntimeError("BixBench smoke had operational failures: " + "; ".join(failures))


def main() -> None:
    """Run a small fresh check of comparison and rubric execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("glm", "opus"),
        default="glm",
        help="Answering model. Opus uses Claude Code subscription authentication.",
    )
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
    parser.add_argument(
        "--question-id",
        default="bix-18",
        help="BixBench task to run. The default passed in all 18 archived conditions.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=BIXBENCH_OUTPUT_ROOT / "smoke"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    bootstrap(args.verbose)
    run(
        args.output_root,
        reuse_stored_results=False,
        reuse_stored_judgments=False,
        model=args.model,
        harness=args.harness,
        runtime=args.runtime,
        replicates=1,
        workers=1,
        burden_workers=1,
        question_ids=(args.question_id,),
    )
    _require_operational_smoke_success(args.output_root)


if __name__ == "__main__":
    main()
