"""Run the full response-characterization analyses from archived QA results.

The missing-final-message and evidence-grounding analyses rerun their LLM
rubrics by default. Pass ``--reuse-stored-judgments`` for an offline analysis
of the archived stochastic judgments instead.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from paper.common.bootstrap import bootstrap, input_path
from paper.config import RESPONSE_EMPTY_TRAILING_JUDGMENTS, RESPONSE_OUTPUT_DIR

logger = logging.getLogger(__name__)

ANALYSES: dict[str, str] = {
    "missing_final_message": (
        "paper.otp_response_characterization.analyses.missing_final_message"
    ),
    "failure_tree": "paper.otp_response_characterization.analyses.failure_tree",
    "no_tool_call": "paper.otp_response_characterization.analyses.no_tool_call",
    "evidence_grounding": "paper.otp_response_characterization.analyses.evidence_grounding",
    "trace_tokens": "paper.otp_response_characterization.analyses.trace_tokens",
}
LIVE_ANALYSES = frozenset({"missing_final_message", "evidence_grounding"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse response-characterization command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analyses",
        nargs="*",
        choices=[*ANALYSES, "all"],
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESPONSE_OUTPUT_DIR,
        help="Directory for regenerated judgments and analysis tables.",
    )
    parser.add_argument(
        "--reuse-stored-judgments",
        action="store_true",
        help=(
            "Analyze archived LLM judgments instead of rerunning the two "
            "model-backed rubrics. This makes the command offline."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    """Parse command-line options and run the selected full analyses."""
    args = parse_args()
    bootstrap(verbose=args.verbose)

    requested = args.analyses or ["all"]
    selected = list(ANALYSES) if "all" in requested else requested
    live_selected = [name for name in selected if name in LIVE_ANALYSES]
    if "failure_tree" in selected and "missing_final_message" not in live_selected:
        live_selected.insert(0, "missing_final_message")
    if live_selected and not args.reuse_stored_judgments:
        logger.warning(
            "The following full analyses will call the configured GPT-OSS judge: %s",
            ", ".join(live_selected),
        )
    completed: set[str] = set()
    if "failure_tree" in selected and not args.reuse_stored_judgments:
        dependency_name = "missing_final_message"
        module = __import__(ANALYSES[dependency_name], fromlist=["run"])
        dependency_out = args.output_dir / dependency_name
        dependency_out.mkdir(parents=True, exist_ok=True)
        logger.info("Running %s as a failure-tree dependency", dependency_name)
        module.run(dependency_out, reuse_stored_judgments=False)
        completed.add(dependency_name)

    for name in selected:
        if name in completed:
            continue
        module = __import__(ANALYSES[name], fromlist=["run"])
        out_dir = args.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Running %s into %s", name, out_dir)
        if name == "failure_tree":
            if args.reuse_stored_judgments:
                judgments_path = input_path(RESPONSE_EMPTY_TRAILING_JUDGMENTS)
            else:
                judgments_path = (
                    args.output_dir
                    / "missing_final_message"
                    / "missing_final_message_judgments.jsonl"
                )
            module.run(out_dir, missing_final_judgments=judgments_path)
        elif name in LIVE_ANALYSES:
            module.run(
                out_dir,
                reuse_stored_judgments=args.reuse_stored_judgments,
            )
        else:
            module.run(out_dir)


if __name__ == "__main__":
    main()
