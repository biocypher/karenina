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

from karenina.replay.exceptions import ProjectionError  # noqa: F401  # re-exported for Task 9's build()
from karenina.replay.store import ReplayEntry, ReplayKey, ReplayMissPolicy, ReplayStore
from karenina.schemas.verification.config import VerificationConfig

if TYPE_CHECKING:
    from karenina.benchmark.benchmark import Benchmark
    from karenina.schemas.scenario.types import ScenarioNode

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


def _probe_qa(
    qa_store: ReplayStore,
    *,
    question_id: str,
    answering_model_id: str,
) -> ReplayEntry | None:
    """Probe a QA store with fall-through semantics regardless of its miss_policy.

    Saves and restores the original policy so the caller's store is
    unchanged on exit. Returns the ReplayEntry or None.
    """
    saved = qa_store.miss_policy
    try:
        qa_store.miss_policy = "fall_through"
        return qa_store.lookup(
            question_id=question_id,
            answering_model_id=answering_model_id,
        )
    finally:
        qa_store.miss_policy = saved


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

    def validate(self) -> ProjectionReport:
        """Walk every staged projection and build a ProjectionReport.

        Does not mutate builder state. For each staged projection,
        expands ``scenarios=None`` to all benchmark scenarios, then
        iterates the cartesian product of scenarios x target_nodes.

        Orphan/duplicate detection uses ``self._projection_consumed_ids``
        which is populated in this method. Orphan classification lands
        in Task 7; duplicate classification lands in Task 6.

        Returns:
            ProjectionReport with projected_keys, unmatched_targets,
            orphan_qa_entries, duplicate_targets.
        """
        projected_keys: list[ReplayKey] = []
        unmatched_targets: list[UnmatchedTarget] = []

        # One set per staged projection; each set holds ``id(entry)`` for
        # every ReplayEntry the projection consumed via _probe_qa. Using
        # object identity correctly handles wildcard QA entries
        # (``answering_model_id=None``) that the ladder matches from a
        # concrete probe; those entries are consumed even though their
        # abstract key differs from the projected key.
        self._projection_consumed_ids: list[set[int]] = []

        for projection in self._staged:
            consumed_ids: set[int] = set()
            self._projection_consumed_ids.append(consumed_ids)

            scenarios = (
                projection.scenarios
                if projection.scenarios is not None
                else [sc.name for sc in self.benchmark.get_scenarios()]
            )

            for scenario_id in scenarios:
                try:
                    scenario_def = self.benchmark.get_scenario(scenario_id)
                except KeyError:
                    for node_id in projection.target_nodes:
                        unmatched_targets.append(
                            UnmatchedTarget(
                                scenario_id=scenario_id,
                                node_id=node_id,
                                question_id=None,
                                answering_model_id=None,
                                reason="missing_scenario",
                            )
                        )
                    continue

                for node_id in projection.target_nodes:
                    node = scenario_def.nodes.get(node_id)
                    if node is None:
                        unmatched_targets.append(
                            UnmatchedTarget(
                                scenario_id=scenario_id,
                                node_id=node_id,
                                question_id=None,
                                answering_model_id=None,
                                reason="missing_node",
                            )
                        )
                        continue

                    question_id, model_display = self._resolve_node_identity(
                        node,
                        projection.config,
                    )

                    hit = _probe_qa(
                        projection.qa_store,
                        question_id=question_id,
                        answering_model_id=model_display,
                    )
                    if hit is None:
                        unmatched_targets.append(
                            UnmatchedTarget(
                                scenario_id=scenario_id,
                                node_id=node_id,
                                question_id=question_id,
                                answering_model_id=model_display,
                                reason="no_qa_entry",
                            )
                        )
                        continue

                    consumed_ids.add(id(hit))
                    projected_keys.append(
                        ReplayKey(
                            question_id=question_id,
                            scenario_id=scenario_id,
                            scenario_node=node_id,
                            answering_model_id=model_display,
                            visit_index=None,
                            replicate=None,
                        )
                    )

        # Duplicate detection across projected_keys (Task 6).
        seen_pairs: dict[tuple[str, str], int] = {}
        for key in projected_keys:
            pair = (key.scenario_id or "", key.scenario_node or "")
            seen_pairs[pair] = seen_pairs.get(pair, 0) + 1
        duplicate_targets: list[tuple[str, str]] = [pair for pair, count in seen_pairs.items() if count > 1]

        # Orphan detection (Task 7): any staged QA entry whose object
        # identity is NOT in the projection's consumed_ids is orphan.
        # Classification: if the entry's question_id was never requested
        # by that projection's targets, it is "no_target_scenario";
        # otherwise the question was asked but under a different model
        # id, so it is "model_id_never_requested".
        orphan_qa_entries: list[OrphanEntry] = []
        for projection, consumed_ids in zip(self._staged, self._projection_consumed_ids, strict=True):
            requested_questions = self._compute_requested_questions(projection)
            for key, entry in projection.qa_store.entries:
                if id(entry) in consumed_ids:
                    continue
                if key.question_id not in requested_questions:
                    orphan_qa_entries.append(
                        OrphanEntry(
                            question_id=key.question_id,
                            answering_model_id=key.answering_model_id,
                            reason="no_target_scenario",
                        )
                    )
                    continue
                orphan_qa_entries.append(
                    OrphanEntry(
                        question_id=key.question_id,
                        answering_model_id=key.answering_model_id,
                        reason="model_id_never_requested",
                    )
                )

        return ProjectionReport(
            projected_keys=projected_keys,
            unmatched_targets=unmatched_targets,
            orphan_qa_entries=orphan_qa_entries,
            duplicate_targets=duplicate_targets,
        )

    def _compute_requested_questions(
        self,
        projection: _StagedProjection,
    ) -> set[str]:
        """Question_ids the projection's declared targets would ask for.

        Used by orphan classification to distinguish ``no_target_scenario``
        (question never requested) from ``model_id_never_requested`` (question
        requested but under a different runtime model). A missing scenario or
        missing node contributes nothing; only resolved targets show up here.
        """
        questions: set[str] = set()
        scenarios = (
            projection.scenarios
            if projection.scenarios is not None
            else [sc.name for sc in self.benchmark.get_scenarios()]
        )
        for scenario_id in scenarios:
            try:
                scenario_def = self.benchmark.get_scenario(scenario_id)
            except KeyError:
                continue
            for node_id in projection.target_nodes:
                node = scenario_def.nodes.get(node_id)
                if node is None:
                    continue
                question_id, _model_display = self._resolve_node_identity(node, projection.config)
                questions.add(question_id)
        return questions

    def _resolve_node_identity(
        self,
        node: ScenarioNode,
        config: VerificationConfig,
    ) -> tuple[str, str]:
        """Compute (question_id, answering_model_display_string) for a node."""
        from karenina.schemas.verification.model_identity import ModelIdentity
        from karenina.utils.checkpoint import generate_question_id

        question_id = generate_question_id(node.question.question)
        if node.model_override is not None and node.model_override.answering_model is not None:
            model = node.model_override.answering_model
        else:
            model = config.answering_models[0]
        model_display = ModelIdentity.from_model_config(model, role="answering").display_string
        return question_id, model_display
