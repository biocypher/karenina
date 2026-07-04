"""Pluggable launcher Protocol + registry for error-analysis agents.

A launcher receives a materialized analysis directory and must cause a
REPORT.md to appear at <analysis_dir>/REPORT.md. Launchers never mutate
files outside analysis_dir.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from karenina.benchmark.error_analysis.exceptions import LauncherNotFoundError

logger = logging.getLogger(__name__)


@runtime_checkable
class ErrorAnalystLauncher(Protocol):
    """Protocol for swappable analyst launchers."""

    def run(self, analysis_dir: Path, **kwargs: object) -> Path: ...


_REGISTRY: dict[str, type[ErrorAnalystLauncher]] = {}


def register_launcher(name: str, cls: type[ErrorAnalystLauncher]) -> None:
    """Register a launcher class under a name.

    Subsequent registrations under the same name overwrite the previous
    binding; this is intentional, to support user overrides.
    """
    _REGISTRY[name] = cls
    logger.debug("Registered error-analyst launcher %s", name)


def get_launcher(name: str) -> type[ErrorAnalystLauncher]:
    """Look up a launcher class by name.

    Raises:
        LauncherNotFoundError: if the name is not registered. The error
            message lists all currently-registered launchers.
    """
    if name not in _REGISTRY:
        raise LauncherNotFoundError(name, registered=sorted(_REGISTRY))
    return _REGISTRY[name]


def list_launchers() -> list[str]:
    """Return the sorted list of currently-registered launcher names."""
    return sorted(_REGISTRY)
