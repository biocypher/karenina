"""Project QA-mode ReplayStore entries onto scenario-mode keys.

ScenarioReplayBuilder is a build-time transform that takes one or
more QA-mode ReplayStore instances and produces a single scenario-mode
ReplayStore the verification pipeline consumes unchanged.

See docs/superpowers/specs/2026-04-14-scenario-replay-projection-design.md
for the design rationale.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict

from karenina.replay.store import ReplayKey, ReplayStore
from karenina.schemas.verification.config import VerificationConfig

logger = logging.getLogger(__name__)


class UnmatchedTarget(BaseModel):
    """A (scenario, node) target that did not resolve to a QA entry.

    ``question_id`` and ``answering_model_id`` are None when the miss
    happened before resolution (missing_scenario or missing_node).
    """

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    node_id: str
    question_id: str | None
    answering_model_id: str | None
    reason: Literal["missing_scenario", "missing_node", "no_qa_entry"]


class OrphanEntry(BaseModel):
    """A QA entry that no staged projection consumed.

    ``no_target_scenario``: the entry's question_id is not attached
    to any node in any of the projection's declared scenarios.

    ``model_id_never_requested``: the entry's question_id is attached
    to at least one targeted scenario/node, but the effective runtime
    model at every such target differs from the entry's
    answering_model_id.
    """

    model_config = ConfigDict(extra="forbid")

    question_id: str
    answering_model_id: str | None
    reason: Literal["no_target_scenario", "model_id_never_requested"]


class ProjectionReport(BaseModel):
    """Result of ScenarioReplayBuilder.validate().

    ``projected_keys`` is the single source of truth for matched
    projections; ``matched`` is a derived length property so the two
    cannot diverge.
    """

    model_config = ConfigDict(extra="forbid")

    projected_keys: list[ReplayKey]
    unmatched_targets: list[UnmatchedTarget]
    orphan_qa_entries: list[OrphanEntry]
    duplicate_targets: list[tuple[str, str]]

    @property
    def matched(self) -> int:
        """Number of projection targets that resolved to a QA entry."""
        return len(self.projected_keys)


class _StagedProjection(BaseModel):
    """Internal staging record for one add_qa() call.

    Not part of the public API. The builder holds a list of these
    between add_qa() calls and consumes them in validate().
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    qa_store: ReplayStore
    target_nodes: list[str]
    scenarios: list[str] | None
    config: VerificationConfig
