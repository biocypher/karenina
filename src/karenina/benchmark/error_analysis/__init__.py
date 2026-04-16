"""Post-run error analysis: materialize a verification run into a navigable
directory and optionally invoke a pluggable agent to produce REPORT.md.

Public surface:
  - analyze_errors: facade orchestrator (added in Task 10)
  - ErrorAnalysisMaterializer: prepare the directory without launching (Task 8)
  - ErrorAnalystLauncher, register_launcher, get_launcher, list_launchers (Task 9)
  - exceptions: ErrorAnalysisError and subclasses (Task 1)
"""

from __future__ import annotations

from karenina.benchmark.error_analysis.exceptions import (
    ErrorAnalysisError,
    LauncherExecutionError,
    LauncherNoOutputError,
    LauncherNotFoundError,
    LauncherUnavailableError,
    MaterializationError,
)

__all__ = [
    "ErrorAnalysisError",
    "LauncherExecutionError",
    "LauncherNoOutputError",
    "LauncherNotFoundError",
    "LauncherUnavailableError",
    "MaterializationError",
]
