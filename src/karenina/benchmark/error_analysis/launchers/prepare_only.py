"""The default launcher: materialize only; do not invoke any agent.

Useful for workflows where the user runs their own agent (Claude Code,
Codex, etc.) against the prepared directory by hand.
"""

from __future__ import annotations

import logging
from pathlib import Path

from karenina.benchmark.error_analysis.exceptions import LauncherNoOutputError

logger = logging.getLogger(__name__)


class PrepareOnlyLauncher:
    """No-op launcher that verifies REPORT.md exists after the fact."""

    def run(self, analysis_dir: Path, **_: object) -> Path:
        report = analysis_dir / "REPORT.md"
        if not report.exists():
            raise LauncherNoOutputError(analysis_dir)
        logger.info("prepare-only launcher: REPORT.md already present at %s", report)
        return report
