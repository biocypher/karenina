"""Run a small live slice of both Open Targets comparison arms."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper.common.bootstrap import bootstrap
from paper.config import MODEL_COMPARISON_SMOKE_OUTPUT_ROOT
from paper.otp_model_comparison.run import run


def main() -> None:
    """Execute a reduced live comparison through the full experiment path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--output-root", type=Path, default=MODEL_COMPARISON_SMOKE_OUTPUT_ROOT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    bootstrap(args.verbose)
    run(
        args.output_root,
        reuse_stored_results=False,
        limit=args.limit,
        model_names=[args.model],
        replicates=1,
        workers=1,
    )


if __name__ == "__main__":
    main()
