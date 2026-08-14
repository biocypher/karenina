"""Run one live standalone Claude Code generation check.

This calls Claude Opus with the Open Targets MCP server configured in Claude
Code. It does not call Karenina.
"""

from paper.common.bootstrap import bootstrap
from paper.config import ADVERSARIAL_OUTPUT_DIR
from paper.otp_adversarial_generation.run import run


def main() -> None:
    """Run the first non-binary source item through standalone Claude Code."""
    bootstrap(verbose=True)
    run(
        ADVERSARIAL_OUTPUT_DIR / "smoke",
        reuse_stored_samples=False,
        limit=1,
        non_binary_only=True,
    )


if __name__ == "__main__":
    main()
