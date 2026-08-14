"""Configuration for the standalone Claude Code generation workflow."""

GENERATOR_MODEL = "claude-opus-4-6[1m]"
CLAUDE_TIMEOUT_SECONDS = 600
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 30
INTER_SAMPLE_DELAY_SECONDS = 5

__all__ = [
    "CLAUDE_TIMEOUT_SECONDS",
    "GENERATOR_MODEL",
    "INTER_SAMPLE_DELAY_SECONDS",
    "MAX_RETRIES",
    "RETRY_DELAY_SECONDS",
]
