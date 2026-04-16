"""Opt-in launcher that shells out to the `claude` CLI.

Registered on import; the facade imports this module lazily when the
user selects launcher="claude-code".
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from karenina.benchmark.error_analysis.exceptions import (
    LauncherExecutionError,
    LauncherUnavailableError,
)
from karenina.benchmark.error_analysis.launcher import register_launcher

logger = logging.getLogger(__name__)

_CLAUDE_BINARY = "claude"
_INSTALL_HINT = (
    "Install the Claude Code CLI from https://docs.anthropic.com/claude/code and ensure `claude` is on PATH."
)


class ClaudeCodeLauncher:
    """Shells out to the `claude` CLI against the analysis directory."""

    def run(self, analysis_dir: Path, **kwargs: object) -> Path:
        if shutil.which(_CLAUDE_BINARY) is None:
            raise LauncherUnavailableError(launcher="claude-code", hint=_INSTALL_HINT)
        timeout_arg = kwargs.get("timeout", 1800)
        if not isinstance(timeout_arg, int | float):
            raise TypeError(f"timeout must be int or float, got {type(timeout_arg).__name__}")
        timeout = int(timeout_arg)
        prompt_arg = f"@{analysis_dir / 'PROMPT.md'}"
        cmd = [
            _CLAUDE_BINARY,
            "-p",
            prompt_arg,
            "--permission-mode",
            "acceptEdits",
        ]
        logger.info("Invoking claude-code launcher in %s", analysis_dir)
        try:
            subprocess.run(cmd, cwd=analysis_dir, check=True, timeout=timeout, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            raise LauncherExecutionError("claude-code", exc) from exc
        except subprocess.TimeoutExpired as exc:
            # Synthesize a CalledProcessError so LauncherExecutionError can stash returncode + stderr.
            stderr_bytes = exc.stderr if isinstance(exc.stderr, bytes) else b""
            synthetic = subprocess.CalledProcessError(
                returncode=-1,
                cmd=cmd,
                stderr=stderr_bytes or f"timeout after {timeout}s".encode(),
            )
            raise LauncherExecutionError("claude-code", synthetic) from exc
        return analysis_dir / "REPORT.md"


register_launcher("claude-code", ClaudeCodeLauncher)
