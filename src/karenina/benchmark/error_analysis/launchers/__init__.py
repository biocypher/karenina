"""Built-in error-analyst launchers.

Importing this package registers prepare-only. claude-code is not
imported by default; the facade imports it lazily when the user selects
launcher="claude-code".
"""

from __future__ import annotations

from karenina.benchmark.error_analysis.launcher import register_launcher
from karenina.benchmark.error_analysis.launchers.prepare_only import PrepareOnlyLauncher

register_launcher("prepare-only", PrepareOnlyLauncher)

__all__ = ["PrepareOnlyLauncher"]
