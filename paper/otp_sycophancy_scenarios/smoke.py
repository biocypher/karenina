"""Run a two-item live Haiku MCP scenario and sidecar check.

This starts Open Targets MCP and calls Haiku, Claude Opus, and GPT-OSS.
"""

from paper.common.bootstrap import bootstrap
from paper.config import SYCOPHANCY_OUTPUT_DIR
from paper.otp_sycophancy_scenarios.run import run


def main() -> None:
    """Run one crossed cell with two scenario items."""
    bootstrap(verbose=True)
    run(
        SYCOPHANCY_OUTPUT_DIR / "smoke",
        reuse_stored_results=False,
        reuse_stored_judgments=False,
        answerers=("claude-haiku-4-5",),
        regimes=("mcp",),
        difficulties=("easy",),
        framings=("casual",),
        limit=2,
        workers=1,
    )


if __name__ == "__main__":
    main()
