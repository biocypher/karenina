"""Replay layer: keyed trace injection for QA and scenarios.

See docs/superpowers/specs/2026-04-08-scenario-replay-store-design.md
for the design document.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from karenina.replay.capture import (
    capture_from_result_set,
    capture_from_scenario_result,
)
from karenina.replay.exceptions import (
    ProjectionError,
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

if TYPE_CHECKING:
    from karenina.replay.projection import (
        OrphanEntry,
        ProjectionReport,
        ScenarioReplayBuilder,
        UnmatchedTarget,
    )

__all__ = [
    "OrphanEntry",
    "ProjectionError",
    "ProjectionReport",
    "ReplayEntry",
    "ReplayError",
    "ReplayHydrationError",
    "ReplayKey",
    "ReplayMissError",
    "ReplayMissPolicy",
    "ReplayPersistenceError",
    "ReplayStore",
    "ScenarioReplayBuilder",
    "UnmatchedTarget",
    "capture_from_result_set",
    "capture_from_scenario_result",
    "dump",
    "load",
]

# Projection symbols are loaded lazily to avoid a circular import:
# ``projection`` depends on ``schemas.verification.config``, which during
# package import is still a partially-initialised module because
# ``karenina.exceptions`` pulls in ``karenina.replay.exceptions`` (and thus
# this package's ``__init__``) while ``schemas.config.models`` is being
# constructed.
_LAZY_PROJECTION_EXPORTS = {
    "OrphanEntry",
    "ProjectionReport",
    "ScenarioReplayBuilder",
    "UnmatchedTarget",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_PROJECTION_EXPORTS:
        from karenina.replay import projection

        return getattr(projection, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
