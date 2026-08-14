"""Run a small live citation screen and web-audit check."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper.common.bootstrap import bootstrap
from paper.config import CITATION_SMOKE_OUTPUT_DIR
from paper.otp_citation_audit.run import run


def main() -> None:
    """Execute a small slice of the fresh citation-rubric workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--audit-model", default="claude-haiku-4-5")
    parser.add_argument("--output-dir", type=Path, default=CITATION_SMOKE_OUTPUT_DIR)
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Run only the GPT-OSS citation screen.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    bootstrap(args.verbose)
    run(
        args.output_dir,
        reuse_stored_judgments=False,
        limit=args.limit,
        audit_model_name=args.audit_model,
        workers=1,
        screen_only=args.screen_only,
    )


if __name__ == "__main__":
    main()
