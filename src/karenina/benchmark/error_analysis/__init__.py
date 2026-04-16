"""Post-run error analysis: materialize a verification run into a navigable
directory and optionally invoke a pluggable agent to produce REPORT.md.

Public surface:
  - analyze_errors: facade orchestrator (added in Task 10)
  - ErrorAnalysisMaterializer: prepare the directory without launching (Task 7)
  - ErrorAnalystLauncher, register_launcher, get_launcher, list_launchers (Task 8)
  - exceptions: ErrorAnalysisError and subclasses (Task 1)

``ErrorAnalysisMaterializer`` is exposed via ``__getattr__`` so that
importing ``karenina.benchmark.error_analysis.exceptions`` (the canonical
site for the exception hierarchy, re-exported by ``karenina.exceptions``)
does not force the rendering pipeline, scenario schemas, and ``ModelConfig``
to load. Loading the materializer before ``karenina.schemas.config`` has
finished initializing triggers a circular import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from karenina.benchmark.error_analysis.exceptions import (
    ErrorAnalysisError,
    LauncherExecutionError,
    LauncherNoOutputError,
    LauncherNotFoundError,
    LauncherUnavailableError,
    MaterializationError,
)

if TYPE_CHECKING:
    from karenina.benchmark.error_analysis.materializer import ErrorAnalysisMaterializer

__all__ = [
    "ErrorAnalysisError",
    "ErrorAnalysisMaterializer",
    "LauncherExecutionError",
    "LauncherNoOutputError",
    "LauncherNotFoundError",
    "LauncherUnavailableError",
    "MaterializationError",
]


def __getattr__(name: str) -> Any:
    if name == "ErrorAnalysisMaterializer":
        from karenina.benchmark.error_analysis.materializer import (
            ErrorAnalysisMaterializer as _ErrorAnalysisMaterializer,
        )

        return _ErrorAnalysisMaterializer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
