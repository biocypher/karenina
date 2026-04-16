"""ScenarioManager: executes scenario graphs turn by turn.

Peer to VerificationManager, following karenina's {Domain}Manager naming.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import logging
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from karenina.benchmark.verification.stages.core.base import ArtifactKeys, VerificationContext
from karenina.benchmark.verification.stages.core.orchestrator import StageOrchestrator
from karenina.ports.messages import Message
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import Rubric
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.state import ScenarioExecutionResult, ScenarioState, TurnRecord
from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.utils.answer_cache import AnswerTraceCache
from karenina.utils.checkpoint import generate_question_id, generate_template_id

from .edge_resolution import resolve_next_node
from .handover import TaggedMessage, apply_handover

logger = logging.getLogger(__name__)


def build_scenario_cache_key(
    scenario_id: str,
    node_id: str,
    answering_model_id: str,
    conversation_history_strs: list[str],
    replicate: int | None = None,
) -> str:
    """Build a cache key for node-level answer caching in scenarios.

    When ``replicate`` is not None, the key carries a ``_rep{N}`` suffix so
    distinct replicates of the same scenario node do not share cached answers.
    ``replicate=None`` preserves the pre-R2 key shape byte-for-byte.
    """
    history_str = "|".join(conversation_history_strs)
    history_hash = hashlib.sha256(history_str.encode()).hexdigest()[:16]
    base = f"{scenario_id}_{node_id}_{answering_model_id}_{history_hash}"
    return base if replicate is None else f"{base}_rep{replicate}"


def _sanitize_for_filesystem(name: str) -> str:
    """Replace non-alphanumeric characters with underscores for filesystem safety."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _model_dir_name(model: ModelConfig) -> str:
    """Derive a filesystem-safe directory name from a ModelConfig."""
    raw = model.id or model.model_name or "unknown"
    return _sanitize_for_filesystem(raw)


class ScenarioManager:
    """Executes scenario graphs turn by turn through the verification pipeline."""

    # Active ScenarioState for the current run(); set inside run() and
    # cleared in its finally block. Read by _peek_visit_index so the
    # per-turn VerificationContext can carry scenario_node_visit_index
    # without threading the state through _run_turn's signature.
    _current_state: ScenarioState | None = None

    def run(
        self,
        scenario: ScenarioDefinition,
        config: VerificationConfig,
        base_answering_model: ModelConfig,
        base_parsing_model: ModelConfig,
        run_name: str | None = None,
        global_rubric: Rubric | None = None,
        progress_callback: Callable[..., None] | None = None,
        answer_cache: AnswerTraceCache | None = None,
        workspace_root: Path | None = None,
        replicate: int | None = None,
    ) -> ScenarioExecutionResult:
        """Execute a scenario with a specific model pair.

        Args:
            scenario: Frozen, validated scenario definition.
            config: Verification configuration (non-model settings).
            base_answering_model: Base answering model for this run.
            base_parsing_model: Base parsing model for this run.
            run_name: Optional run name for tracking.
            global_rubric: Optional global rubric applied per-turn.
            progress_callback: Optional callback for turn-level progress.
            answer_cache: Optional cache for sharing answers across runs.
            workspace_root: Optional workspace root directory, required when
                agentic_parsing is True. Plumbed from Benchmark.workspace_root
                through ScenarioExecutor.run_batch. See GenerateAnswer stage's
                _resolve_workspace which hard-requires this when agentic_parsing.
            replicate: Run-level replicate index, shared across all turns
                in this scenario execution. None for single-replicate runs
                (matches the QA convention). Stamped onto the returned
                ``ScenarioExecutionResult`` and threaded into every
                ``VerificationContext.replicate``.

        Returns:
            ScenarioExecutionResult with all turn data.
        """
        if config.evaluation_mode == "rubric_only":
            warnings.warn(
                "evaluation_mode='rubric_only' is not supported in scenarios. "
                "Scenarios always auto-detect evaluation mode from rubric presence. "
                "The rubric_only setting will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        # Initialize state
        state = ScenarioState(
            turn=0,
            current_node=scenario.entry_node,
            verify_result=None,
            parsed={},
            node_visits={},
            history=[],
            accumulated={},
            node_results={},
        )

        # Stamp pipeline-level request_timeout onto base models (same as batch_runner)
        if config.request_timeout is not None:
            if base_answering_model.request_timeout is None:
                base_answering_model = base_answering_model.model_copy(
                    update={"request_timeout": config.request_timeout},
                )
            if base_parsing_model.request_timeout is None:
                base_parsing_model = base_parsing_model.model_copy(
                    update={"request_timeout": config.request_timeout},
                )

        tagged_messages: list[TaggedMessage] = []
        pending_handover_edge: ScenarioEdge | None = None
        turn_results: list[VerificationResult] = []
        path: list[str] = []
        previous_agent_id: str | None = None
        previous_system_prompt: str | None = None
        status: Literal["completed", "limit_reached", "error"] = "completed"

        # Resolve base identity from entry node
        entry_node_obj = scenario.nodes[scenario.entry_node]
        base_identity = (
            entry_node_obj.agent_identity or base_answering_model.id or base_answering_model.model_name or "unknown"
        )

        # Create scenario-level workspace directory when workspace_root is set.
        # Structure: workspace_root / scenario_name / model_id / [rep_{N} /]
        # The optional rep_{N} segment isolates replicate workspaces so
        # parallel runs do not interleave writes and sequential runs do not
        # overwrite each other. replicate=None preserves the pre-R2 shape
        # (no rep_ segment) byte-for-byte.
        scenario_dir: Path | None = None
        if workspace_root is not None:
            scenario_dir = (
                workspace_root / _sanitize_for_filesystem(scenario.name) / _model_dir_name(base_answering_model)
            )
            if replicate is not None:
                scenario_dir = scenario_dir / f"rep_{replicate}"
            scenario_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created scenario workspace: %s", scenario_dir)

        # Expose the current ScenarioState to _run_turn via an instance
        # attribute so _peek_visit_index can read node_visits without
        # threading state through the _run_turn signature. Cleared in the
        # finally block below so the attribute does not leak between runs.
        self._current_state = state

        try:
            while True:
                node = scenario.nodes[state.current_node]
                path.append(state.current_node)
                agent_id = node.agent_identity or base_identity

                # Resolve models for this turn
                answering_model, parsing_model = _resolve_models(
                    node,
                    base_answering_model,
                    base_parsing_model,
                )

                # Create per-turn workspace directory before handover so traces
                # land inside the turn folder instead of a shared flat directory.
                turn_dir: Path | None = None
                if scenario_dir is not None:
                    turn_dir = scenario_dir / f"turn_{state.turn}"
                    turn_dir.mkdir(parents=True, exist_ok=True)

                # Build conversation_history for this turn
                question_text = node.question.question
                if pending_handover_edge is not None:
                    handover_result = apply_handover(
                        pending_handover_edge,
                        tagged_messages,
                        state,
                        question_text,
                        turn_dir=turn_dir,
                    )
                    if handover_result is not None:
                        question_text, conversation_history = handover_result
                    else:
                        conversation_history = [tm.message for tm in tagged_messages]
                else:
                    conversation_history = [tm.message for tm in tagged_messages]

                # Inject system prompt into tagged_messages on agent or prompt change.
                # This makes the system prompt visible in transcripts for downstream
                # nodes (e.g. the agentic guardrail judge).
                agent_changed = (previous_agent_id is None) or (previous_agent_id != agent_id)
                prompt_changed = answering_model.system_prompt != previous_system_prompt
                if (agent_changed or prompt_changed) and answering_model.system_prompt:
                    system_msg = Message.system(answering_model.system_prompt)
                    tagged_messages.append(TaggedMessage(system_msg, agent_id=agent_id))

                # Build question message and tag it
                question_msg = Message.user(question_text)
                tagged_messages.append(TaggedMessage(question_msg, agent_id="__user__"))

                # Determine question_text_override (only when handover modified it)
                question_text_override = question_text if question_text != node.question.question else None

                # Build answer cache key for this turn. Skip the cache
                # entirely when a replay hit is known for the upcoming
                # node: GenerateAnswerStage will short-circuit and we
                # would otherwise leak an orphaned IN_PROGRESS slot.
                cached_answer_data = None
                cache_key = None

                will_replay = False
                if config.replay_store is not None:
                    will_replay = config.replay_store.has_any_for(
                        question_id=generate_question_id(node.question.question),
                        scenario_id=scenario.name,
                        scenario_node=state.current_node,
                    )

                if answer_cache is not None and not will_replay:
                    answering_model_id = answering_model.id or answering_model.model_name or "unknown"
                    history_strs = [str(m) for m in conversation_history]
                    cache_key = build_scenario_cache_key(
                        scenario.name,
                        state.current_node,
                        answering_model_id,
                        history_strs,
                        replicate=replicate,
                    )
                    cache_status, cached_answer_data = answer_cache.get_or_reserve(cache_key)
                    if cache_status == "IN_PROGRESS":
                        answer_cache.wait_for_completion(cache_key, timeout=config.request_timeout or 120.0)
                        cache_status, cached_answer_data = answer_cache.get_or_reserve(cache_key)
                    if cache_status == "HIT":
                        logger.debug("Scenario cache hit for %s/%s", scenario.name, state.current_node)

                # Run the verification pipeline for this turn (adapter handles retries internally)
                vr, trace_messages, parsed_answer, raw_response = self._run_turn(
                    node=node,
                    conversation_history=conversation_history,
                    answering_model=answering_model,
                    parsing_model=parsing_model,
                    config=config,
                    run_name=run_name,
                    global_rubric=global_rubric,
                    turn_index=state.turn,
                    scenario_id=scenario.name,
                    scenario_node=state.current_node,
                    scenario_path=list(path),
                    replicate=replicate,
                    question_text_override=question_text_override,
                    cached_answer_data=cached_answer_data,
                    workspace_root=workspace_root,
                    turn_workspace_path=turn_dir,
                    node_results=dict(state.node_results),
                )

                if vr.metadata.failure is not None:
                    logger.error(
                        "Scenario %s: node '%s' failed with error: %s (category: %s)",
                        scenario.name,
                        state.current_node,
                        vr.metadata.failure.reason,
                        vr.metadata.failure.category.value,
                    )

                # Complete cache entry after final attempt
                if answer_cache is not None and cache_key is not None and cached_answer_data is None:
                    from karenina.benchmark.verification.utils.cache_helpers import extract_answer_data_from_result

                    try:
                        answer_data = extract_answer_data_from_result(vr)
                        answer_cache.complete(cache_key, answer_data)
                    except Exception:
                        logger.warning(
                            "Failed to cache answer for %s/%s",
                            scenario.name,
                            state.current_node,
                            exc_info=True,
                        )
                        answer_cache.complete(cache_key, None, error=Exception("extraction failed"))

                # Grow tagged history with trace
                if trace_messages:
                    for msg in trace_messages:
                        tagged_messages.append(TaggedMessage(msg, agent_id=agent_id))

                # Build TurnRecord
                verify_result = vr.template.verify_result if vr.template is not None else None
                parsed_fields: dict[str, Any] = {}
                if parsed_answer is not None:
                    with contextlib.suppress(Exception):
                        parsed_fields = parsed_answer.model_dump()

                record = TurnRecord(
                    node_id=state.current_node,
                    question_text=node.question.question,
                    question_messages=[Message.user(node.question.question)],
                    trace_messages=trace_messages or [],
                    raw_response=raw_response or "",
                    parsed_answer=parsed_answer,
                    parsed_fields=parsed_fields,
                    verify_result=verify_result,
                    verification_result_id=vr.metadata.result_id,
                )

                turn_results.append(vr)

                # Extract rubric trait scores
                rubric_results: dict[str, Any] = {}
                if vr.rubric is not None and vr.rubric.rubric_evaluation_performed:
                    rubric_results = vr.rubric.get_all_trait_scores()

                # Auto-populate node_results (last-write-wins on revisits)
                state.node_results[state.current_node] = {
                    "verify_result": verify_result,
                    "parsed": parsed_fields,
                    "rubric": rubric_results,
                }

                # Run state_update if defined (opt-in custom state)
                if node.state_update is not None:
                    snapshot = copy.deepcopy(state.accumulated)
                    try:
                        state.accumulated = node.state_update(
                            state.accumulated,
                            parsed_fields,
                        )
                    except Exception:
                        logger.warning(
                            "state_update for node '%s' raised; restoring from snapshot",
                            state.current_node,
                            exc_info=True,
                        )
                        state.accumulated = snapshot

                # Update state
                state.turn += 1
                state.node_visits[state.current_node] = state.node_visits.get(state.current_node, 0) + 1
                state.history.append(record)
                state.verify_result = verify_result
                state.parsed = parsed_fields

                # Check for pipeline error
                if vr.metadata.failure is not None:
                    self._report_progress(
                        progress_callback,
                        scenario.name,
                        state.turn - 1,
                        record.node_id,
                        verify_result,
                        None,
                    )
                    status = "error"
                    break

                # Check global turn limit
                if state.turn >= config.scenario_turn_limit:
                    self._report_progress(
                        progress_callback,
                        scenario.name,
                        state.turn - 1,
                        record.node_id,
                        verify_result,
                        None,
                    )
                    status = "limit_reached"
                    break

                # Resolve next node
                outbound_edges = [e for e in scenario.edges if e.source == state.current_node]
                next_node, followed_edge = resolve_next_node(outbound_edges, state)

                # Report progress (after edge resolution so next_node is known)
                self._report_progress(
                    progress_callback,
                    scenario.name,
                    state.turn - 1,
                    record.node_id,
                    verify_result,
                    next_node,
                )

                if next_node is None:
                    # Implicit terminal (no outbound edges)
                    status = "completed"
                    break

                if next_node == END:
                    status = "completed"
                    break

                pending_handover_edge = followed_edge
                state.current_node = next_node
                previous_agent_id = agent_id
                previous_system_prompt = answering_model.system_prompt
        finally:
            # Clear the per-run state pointer so the attribute does not
            # leak between runs or mask bugs in _peek_visit_index callers.
            self._current_state = None

        # Evaluate outcome criteria
        result = ScenarioExecutionResult(
            scenario_id=scenario.name,
            status=status,
            path=path,
            turn_count=state.turn,
            history=state.history,
            turn_results=turn_results,
            final_state=state,
            outcome_results={},
            replicate=replicate,
        )

        result.outcome_results = _evaluate_outcome_criteria(
            scenario,
            result,
        )

        return result

    def _run_turn(
        self,
        node: ScenarioNode,
        conversation_history: list[Message],
        answering_model: ModelConfig,
        parsing_model: ModelConfig,
        config: VerificationConfig,
        run_name: str | None,
        global_rubric: Rubric | None,
        turn_index: int = 0,
        scenario_id: str | None = None,
        scenario_node: str | None = None,
        scenario_path: list[str] | None = None,
        replicate: int | None = None,
        question_text_override: str | None = None,
        cached_answer_data: dict[str, Any] | None = None,
        workspace_root: Path | None = None,
        turn_workspace_path: Path | None = None,
        node_results: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[VerificationResult, list[Message] | None, Any, str | None]:
        """Execute one turn of the verification pipeline.

        Args:
            turn_workspace_path: Pre-created per-turn workspace directory.
                When set, GenerateAnswerStage skips its own workspace creation
                and uses this path directly.

        Returns:
            Tuple of (VerificationResult, trace_messages, parsed_answer, raw_response).
        """
        template_code = node.question.answer_template or ""
        template_id = generate_template_id(template_code)

        # Determine rubric: per-question rubric takes precedence, then global.
        # The global_rubric arrives already stamped from the benchmark facade,
        # but per-question rubrics on scenario nodes are deserialized here and
        # have not yet had pipeline-level retry_policy / request_timeout
        # propagated onto any AgenticRubricTrait.model_override. Stamp them
        # via the same helper used by the QA path so that scenario per-question
        # agentic traits inherit the same defaults as global rubric traits.
        rubric = None
        if node.question.question_rubric:
            from karenina.benchmark.verification.utils.task_helpers import (
                stamp_agentic_trait_overrides,
            )
            from karenina.schemas.entities.rubric import Rubric as RubricCls

            if isinstance(node.question.question_rubric, dict):
                rubric = RubricCls.model_validate(node.question.question_rubric)
            elif isinstance(node.question.question_rubric, RubricCls):
                rubric = node.question.question_rubric
            rubric = stamp_agentic_trait_overrides(rubric, config)
        if rubric is None:
            rubric = global_rubric

        # Determine evaluation mode
        evaluation_mode = "template_only"
        if rubric and (
            rubric.llm_traits
            or rubric.regex_traits
            or rubric.callable_traits
            or rubric.metric_traits
            or rubric.agentic_traits
        ):
            evaluation_mode = "template_and_rubric"

        # Build ErrorRegistry from config custom patterns
        from karenina.benchmark.verification.runner import _build_error_registry

        error_registry = _build_error_registry(config.custom_error_patterns)

        # Build VerificationContext
        context = VerificationContext(
            question_id=generate_question_id(node.question.question),
            template_id=template_id,
            question_text=question_text_override or node.question.question,
            template_code=template_code,
            answering_model=answering_model,
            parsing_model=parsing_model,
            rubric=rubric,
            keywords=node.question.keywords,
            raw_answer=node.question.raw_answer,
            run_name=run_name,
            few_shot_enabled=bool(node.question.few_shot_examples),
            few_shot_examples=node.question.few_shot_examples,
            use_full_trace_for_template=config.use_full_trace_for_template,
            use_full_trace_for_rubric=config.use_full_trace_for_rubric,
            scenario_turn=turn_index,
            scenario_id=scenario_id,
            scenario_node=scenario_node,
            scenario_path=scenario_path,
            replicate=replicate,
            # Agentic parsing configuration (forwarded from VerificationConfig so
            # scenario nodes actually reach Stage 7b when agentic_parsing is set).
            agentic_parsing=config.agentic_parsing,
            agentic_judge_context=config.agentic_judge_context,
            agentic_parsing_max_turns=config.agentic_parsing_max_turns,
            agentic_parsing_timeout=config.agentic_parsing_timeout,
            agentic_parsing_materialize_trace=config.agentic_parsing_materialize_trace,
            agentic_parsing_persist_trace=config.agentic_parsing_persist_trace,
            # Workspace configuration. When turn_workspace_path is set (scenario
            # path), pre-populate workspace_path so GenerateAnswerStage skips
            # its own directory creation. Otherwise fall back to workspace_root
            # for the QA path.
            workspace_root=workspace_root,
            workspace_path=turn_workspace_path,
            workspace_is_copy=turn_workspace_path is not None,
            workspace_copy=config.workspace_copy,
            workspace_cleanup=config.workspace_cleanup,
            question_workspace_path=getattr(node.question, "workspace_path", None),
            # Replay fields: copied from config onto the per-turn context.
            # scenario_node_visit_index is read from the current ScenarioState
            # BEFORE the post-turn increment in run(), so retry loops that
            # revisit the same node observe distinct visit counts.
            replay_store=config.replay_store,
            replay_parse_on_hydration_mismatch=config.replay_parse_on_hydration_mismatch,
            scenario_node_visit_index=(self._peek_visit_index(scenario_node) if scenario_node else None),
            error_registry=error_registry,
        )

        # Set conversation history artifact (the key integration point).
        # Pass a copy so the pipeline does not mutate our accumulator.
        context.set_artifact("conversation_history", list(conversation_history))

        # Pass node_results for ConditionalGroundTruth resolution in VerifyTemplateStage.
        if node_results is not None:
            context.set_artifact(ArtifactKeys.SCENARIO_NODE_RESULTS, dict(node_results))

        # Set model identity artifacts
        answering_identity = ModelIdentity.from_model_config(answering_model, role="answering")
        parsing_identity = ModelIdentity.from_model_config(parsing_model, role="parsing")
        context.set_artifact(ArtifactKeys.ANSWERING_MODEL_IDENTITY, answering_identity)
        context.set_artifact(ArtifactKeys.PARSING_MODEL_IDENTITY, parsing_identity)

        if cached_answer_data is not None:
            context.cached_answer_data = cached_answer_data

        # Build and execute orchestrator
        orchestrator = StageOrchestrator.from_config(
            rubric=rubric,
            abstention_enabled=config.abstention_enabled,
            sufficiency_enabled=config.sufficiency_enabled,
            deep_judgment_enabled=False,  # Scenarios do not use template deep judgment
            evaluation_mode=evaluation_mode,
            agentic_parsing=config.agentic_parsing,
        )

        vr = orchestrator.execute(context)

        # Extract artifacts for TurnRecord
        trace_messages = context.get_artifact(ArtifactKeys.TRACE_MESSAGES)
        parsed_answer = context.get_artifact(ArtifactKeys.PARSED_ANSWER)
        raw_response = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)

        return vr, trace_messages, parsed_answer, raw_response

    def _peek_visit_index(self, node_id: str) -> int:
        """Return the current visit count for ``node_id`` without mutating state.

        Called from ``_run_turn`` to populate
        ``VerificationContext.scenario_node_visit_index``. The manager stashes
        the active ScenarioState on ``self._current_state`` for the duration
        of ``run()`` (see the try/finally in ``run()``). This method reads
        ``state.node_visits[node_id]`` before ``run()`` performs its post-turn
        increment, so the first visit to a node yields 0, the second yields 1,
        and so on.

        Returns 0 if no active state is bound (e.g. the helper is called
        outside of a ``run()`` invocation).
        """
        state = self._current_state
        if state is None:
            return 0
        return state.node_visits.get(node_id, 0)

    @staticmethod
    def _report_progress(
        callback: Callable[..., None] | None,
        scenario_id: str,
        turn: int,
        node_id: str,
        verify_result: bool | None,
        next_node: str | None,
    ) -> None:
        """Report turn-level progress via callback."""
        if callback is None:
            return
        try:
            callback(
                scenario_id=scenario_id,
                scenario_turn=turn,
                scenario_node=node_id,
                verify_result=verify_result,
                next_node=next_node,
            )
        except Exception:
            logger.warning("Progress callback failed", exc_info=True)


def _resolve_models(
    node: ScenarioNode,
    base_answering: ModelConfig,
    base_parsing: ModelConfig,
) -> tuple[ModelConfig, ModelConfig]:
    """Resolve per-turn models from node override or base config.

    Override models inherit ``request_timeout`` and ``retry_policy`` from the
    base model when not set, so the pipeline-level timeout and retry policy
    propagate to per-node model overrides. The benchmark facade also stamps
    these fields onto overrides up front (see
    :meth:`Benchmark._run_scenario_verification`); this fallback handles
    callers that drive the manager directly without going through the
    facade.
    """
    override = node.model_override
    answering = override.answering_model if override and override.answering_model else base_answering
    parsing = override.parsing_model if override and override.parsing_model else base_parsing

    # Propagate request_timeout from base models to overrides that don't set their own
    if answering.request_timeout is None and base_answering.request_timeout is not None:
        answering = answering.model_copy(update={"request_timeout": base_answering.request_timeout})
    if parsing.request_timeout is None and base_parsing.request_timeout is not None:
        parsing = parsing.model_copy(update={"request_timeout": base_parsing.request_timeout})

    # Propagate retry_policy from base models to overrides that don't set their own
    if answering.retry_policy is None and base_answering.retry_policy is not None:
        answering = answering.model_copy(update={"retry_policy": base_answering.retry_policy})
    if parsing.retry_policy is None and base_parsing.retry_policy is not None:
        parsing = parsing.model_copy(update={"retry_policy": base_parsing.retry_policy})

    return answering, parsing


def _evaluate_outcome_criteria(
    scenario: ScenarioDefinition,
    execution_result: ScenarioExecutionResult,
) -> dict[str, bool | int | float]:
    """Evaluate all outcome criteria against the execution result.

    Dispatches to ``evaluate_outcome`` for declarative checks
    (criterion.check is not None) and falls back to criterion.evaluate()
    for callable escape hatches.
    """
    results: dict[str, bool | int | float] = {}
    for criterion in scenario.outcome_criteria:
        try:
            if criterion.check is not None:
                from karenina.scenario.outcome_evaluation import evaluate_outcome

                results[criterion.name] = evaluate_outcome(criterion.check, execution_result)
            elif criterion.evaluate is not None:
                results[criterion.name] = criterion.evaluate(execution_result)
            else:
                logger.warning(
                    "Criterion '%s' has neither check nor evaluate",
                    criterion.name,
                )
        except Exception:
            logger.warning(
                "Outcome criterion '%s' raised an exception",
                criterion.name,
                exc_info=True,
            )
            results[criterion.name] = False
    return results
