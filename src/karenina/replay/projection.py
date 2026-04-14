"""Project QA-mode ReplayStore entries onto scenario-mode keys.

ScenarioReplayBuilder is a build-time transform that takes one or
more QA-mode ReplayStore instances and produces a single scenario-mode
ReplayStore the verification pipeline consumes unchanged.

See docs/superpowers/specs/2026-04-14-scenario-replay-projection-design.md
for the design rationale.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from karenina.replay.store import ReplayKey, ReplayMissPolicy, ReplayStore
from karenina.schemas.verification.config import VerificationConfig

if TYPE_CHECKING:
    from karenina.benchmark.benchmark import Benchmark

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


def _dedupe_with_warning(values: list[str], *, kind: str) -> list[str]:
    """Dedupe preserving input order; emit a warning if duplicates were dropped."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    if len(out) != len(values):
        logger.warning(
            "ScenarioReplayBuilder: duplicate values deduped in %s",
            kind,
        )
    return out


class ScenarioReplayBuilder:
    """Project QA-mode ReplayStores onto scenario targets in a benchmark.

    Three-phase flow::

        builder = ScenarioReplayBuilder(benchmark, config=cfg)
        builder.add_qa(qa_store, target_nodes=["ask"], scenarios=["s1", "s2"])
        report = builder.validate()
        store = builder.build(strict=True)

    Attributes:
        benchmark: The benchmark whose scenarios are projected onto.
        config: Default VerificationConfig used when a node has no
            ``ModelOverride.answering_model``.
        miss_policy: Miss policy written onto the produced ReplayStore.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        *,
        config: VerificationConfig,
        miss_policy: ReplayMissPolicy = "strict",
    ) -> None:
        if benchmark is None:
            raise TypeError("benchmark must not be None")
        if config is None:
            raise TypeError("config must not be None")
        if not config.answering_models:
            raise ValueError("config.answering_models must be non-empty")

        self.benchmark = benchmark
        self.config = config
        self.miss_policy: ReplayMissPolicy = miss_policy
        self._staged: list[_StagedProjection] = []

    def add_qa(
        self,
        qa_store: ReplayStore,
        *,
        target_nodes: list[str],
        scenarios: list[str] | None = None,
        config: VerificationConfig | None = None,
    ) -> ScenarioReplayBuilder:
        """Stage a QA store for projection onto the given scenario targets.

        Args:
            qa_store: A QA-mode, replicate-canonicalized ReplayStore.
                Every entry's ReplayKey must have scenario_id is None
                and replicate is None.
            target_nodes: Non-empty list of node_ids to project onto.
            scenarios: Either a non-empty list of scenario ids, or None
                to mean "all scenarios in the benchmark". Empty list is
                a hard error (caller explicitly said no scenarios).
            config: Optional override for the builder's default
                VerificationConfig. Snapshotted via model_copy(deep=True)
                at staging time so later caller mutations do not leak.

        Returns:
            self, to support chaining.

        Raises:
            TypeError: If qa_store is None.
            ValueError: If target_nodes is empty, scenarios is an empty
                list, qa_store contains scenario-mode entries or
                per-replicate entries, or a config override has an
                empty answering_models list.
        """
        if qa_store is None:
            raise TypeError("qa_store must not be None")
        if not target_nodes:
            raise ValueError("target_nodes must be a non-empty list")
        if scenarios is not None and len(scenarios) == 0:
            raise ValueError("scenarios must be None (meaning all) or a non-empty list")

        for key, _entry in qa_store.entries:
            if key.scenario_id is not None:
                raise ValueError("qa_store contains scenario-mode entries; expected a QA-captured store")
            if key.replicate is not None:
                raise ValueError(
                    "qa_store contains per-replicate entries; re-capture with "
                    "replicate_selector='first' or 'last' to canonicalize"
                )

        effective_config: VerificationConfig
        if config is None:
            effective_config = self.config
        else:
            if not config.answering_models:
                raise ValueError("config override answering_models must be non-empty")
            effective_config = config.model_copy(deep=True)

        deduped_nodes = _dedupe_with_warning(target_nodes, kind="target_nodes")
        deduped_scenarios = _dedupe_with_warning(scenarios, kind="scenarios") if scenarios is not None else None

        self._staged.append(
            _StagedProjection(
                qa_store=qa_store,
                target_nodes=deduped_nodes,
                scenarios=deduped_scenarios,
                config=effective_config,
            )
        )
        return self
