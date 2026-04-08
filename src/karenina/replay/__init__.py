"""Replay layer: keyed trace injection for QA and scenarios.

See docs/superpowers/specs/2026-04-08-scenario-replay-store-design.md
for the design document.
"""

from __future__ import annotations

from karenina.replay.capture import (
    capture_from_result_set,
    capture_from_scenario_result,
)
from karenina.replay.exceptions import (
    ReplayError,
    ReplayHydrationError,
    ReplayMissError,
    ReplayPersistenceError,
)
from karenina.replay.persistence import dump, load
from karenina.replay.store import (
    ReplayEntry,
    ReplayKey,
    ReplayMissPolicy,
    ReplayStore,
)

__all__ = [
    "ReplayEntry",
    "ReplayError",
    "ReplayHydrationError",
    "ReplayKey",
    "ReplayMissError",
    "ReplayMissPolicy",
    "ReplayPersistenceError",
    "ReplayStore",
    "capture_from_result_set",
    "capture_from_scenario_result",
    "dump",
    "load",
]
