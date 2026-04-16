"""Exception hierarchy for the error-analysis feature.

All exceptions inherit from ErrorAnalysisError, which inherits from
KareninaError. The hierarchy is re-exported from karenina.exceptions
for convenience.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from karenina.exceptions import KareninaError


class ErrorAnalysisError(KareninaError):
    """Base exception for the error-analysis feature."""


class MaterializationError(ErrorAnalysisError):
    """Raised when the materializer cannot produce the analysis directory.

    Args:
        message: Human-readable description.
        details: Structured context (unwritable path, missing template id, etc.).
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class LauncherNotFoundError(ErrorAnalysisError):
    """Raised when a launcher name is not present in the registry.

    Args:
        launcher: The requested name.
        registered: Names currently in the registry.
    """

    def __init__(self, launcher: str, registered: list[str]) -> None:
        self.launcher = launcher
        self.registered = registered
        message = (
            f"Launcher '{launcher}' is not registered. "
            f"Available launchers: {', '.join(registered) if registered else '<none>'}."
        )
        super().__init__(message)


class LauncherUnavailableError(ErrorAnalysisError):
    """Raised when a launcher is registered but its runtime dependency is missing.

    Args:
        launcher: The launcher name.
        hint: Install or configuration hint shown to the user.
    """

    def __init__(self, launcher: str, hint: str) -> None:
        self.launcher = launcher
        self.hint = hint
        super().__init__(f"Launcher '{launcher}' is unavailable: {hint}")


class LauncherExecutionError(ErrorAnalysisError):
    """Raised when a launcher subprocess failed.

    Preserves the returncode and the last 2048 bytes of stderr so callers
    can diagnose without replaying the subprocess.

    Args:
        launcher: The launcher name.
        cause: The originating CalledProcessError.
    """

    STDERR_TAIL_BYTES = 2048

    def __init__(self, launcher: str, cause: subprocess.CalledProcessError) -> None:
        self.launcher = launcher
        self.returncode = cause.returncode
        raw = cause.stderr or b""
        if isinstance(raw, bytes):
            tail_bytes = raw[-self.STDERR_TAIL_BYTES :]
            self.stderr_tail = tail_bytes.decode("utf-8", errors="replace")
        else:
            self.stderr_tail = raw[-self.STDERR_TAIL_BYTES :]
        super().__init__(
            f"Launcher '{launcher}' exited with status {self.returncode}. Last stderr: {self.stderr_tail!r}"
        )


class LauncherNoOutputError(ErrorAnalysisError):
    """Raised when a launcher returned without error but REPORT.md is absent.

    Args:
        analysis_dir: The analysis directory that was expected to contain REPORT.md.
    """

    def __init__(self, analysis_dir: Path) -> None:
        self.analysis_dir = analysis_dir
        super().__init__(f"Launcher completed but no REPORT.md was produced at {analysis_dir / 'REPORT.md'}.")
