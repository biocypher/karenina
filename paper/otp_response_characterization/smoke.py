"""Run a small live smoke check for an LLM-based response analysis.

Smoke checks demonstrate that the live judging path works. Their outputs are
not reproduction targets because LLM judgments are stochastic.
"""

from __future__ import annotations

import argparse

from paper.common.bootstrap import bootstrap

LIVE_ANALYSES: dict[str, str] = {
    "missing_final_message": (
        "paper.otp_response_characterization.analyses.missing_final_message"
    ),
    "evidence_grounding": "paper.otp_response_characterization.analyses.evidence_grounding",
}


def main() -> None:
    """Parse command-line options and dispatch one live smoke check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=sorted(LIVE_ANALYSES))
    parser.add_argument("--limit", type=int, default=3, help="Maximum traces to judge.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    bootstrap(verbose=args.verbose)

    module = __import__(LIVE_ANALYSES[args.analysis], fromlist=["run_smoke"])
    module.run_smoke(limit=args.limit)


if __name__ == "__main__":
    main()
