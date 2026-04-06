"""ScenarioManager: executes scenario graphs turn by turn.

Peer to VerificationManager, following karenina's {Domain}Manager naming.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import logging
import time
import warnings
from collections.abc import Callable
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
) -> str:
    """Build a cache key for node-level answer caching in scenarios."""
    history_str = "|".join(conversation_history_strs)
    history_hash = hashlib.sha256(history_str.encode()).hexdigest()[:16]
    return f"{scenario_id}_{node_id}_{answering_model_id}_{history_hash}"


class ScenarioManager:
    """Executes scenario graphs turn by turn through the verification pipeline."""

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
        status: Literal["completed", "limit_reached", "error"] = "completed"

        # Resolve base identity from entry node
        entry_node_obj = scenario.nodes[scenario.entry_node]
        base_identity = (
            entry_node_obj.agent_identity or base_answering_model.id or base_answering_model.model_name or "unknown"
        )

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

            # Build conversation_history for this turn
            question_text = node.question.question
            if pending_handover_edge is not None:
                handover_result = apply_handover(
                    pending_handover_edge,
                    tagged_messages,
                    state,
                    question_text,
                )
                if handover_result is not None:
                    question_text, conversation_history = handover_result
                else:
                    conversation_history = [tm.message for tm in tagged_messages]
            else:
                conversation_history = [tm.message for tm in tagged_messages]

            # Build question message and tag it
            question_msg = Message.user(question_text)
            tagged_messages.append(TaggedMessage(question_msg, agent_id="__user__"))

            # Determine question_text_override (only when handover modified it)
            question_text_override = question_text if question_text != node.question.question else None

            # Build answer cache key for this turn
            cached_answer_data = None
            cache_key = None
            if answer_cache is not None:
                answering_model_id = answering_model.id or answering_model.model_name or "unknown"
                history_strs = [str(m) for m in conversation_history]
                cache_key = build_scenario_cache_key(
                    scenario.name,
                    state.current_node,
                    answering_model_id,
                    history_strs,
                )
                cache_status, cached_answer_data = answer_cache.get_or_reserve(cache_key)
                if cache_status == "IN_PROGRESS":
                    answer_cache.wait_for_completion(cache_key, timeout=config.request_timeout or 120.0)
                    cache_status, cached_answer_data = answer_cache.get_or_reserve(cache_key)
                if cache_status == "HIT":
                    logger.debug("Scenario cache hit for %s/%s", scenario.name, state.current_node)

            # Run the verification pipeline for this turn (with retry on transient errors)
            max_turn_attempts = config.retry_policy.derive_sdk_max_retries()
            for turn_attempt in range(1, max_turn_attempts + 1):
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
                    question_text_override=question_text_override,
                    cached_answer_data=cached_answer_data,
                )

                if vr.metadata.completed_without_errors:
                    break

                if vr.metadata.error_category is None or vr.metadata.error_category == "permanent":
                    break  # permanent error, no retry

                if turn_attempt < max_turn_attempts:
                    logger.warning(
                        "Scenario %s, node %s: transient pipeline error on attempt %d/%d, retrying: %s",
                        scenario.name,
                        state.current_node,
                        turn_attempt,
                        max_turn_attempts,
                        vr.metadata.error,
                    )
                    time.sleep(1)

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
            if not vr.metadata.completed_without_errors:
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
        question_text_override: str | None = None,
        cached_answer_data: dict[str, Any] | None = None,
    ) -> tuple[VerificationResult, list[Message] | None, Any, str | None]:
        """Execute one turn of the verification pipeline.

        Returns:
            Tuple of (VerificationResult, trace_messages, parsed_answer, raw_response).
        """
        template_code = node.question.answer_template or ""
        template_id = generate_template_id(template_code)

        # Determine rubric: per-question rubric takes precedence, then global
        rubric = None
        if node.question.question_rubric:
            from karenina.schemas.entities.rubric import Rubric as RubricCls

            if isinstance(node.question.question_rubric, dict):
                rubric = RubricCls.model_validate(node.question.question_rubric)
            elif isinstance(node.question.question_rubric, RubricCls):
                rubric = node.question.question_rubric
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
            error_registry=error_registry,
        )

        # Set conversation history artifact (the key integration point).
        # Pass a copy so the pipeline does not mutate our accumulator.
        context.set_artifact("conversation_history", list(conversation_history))

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
        )

        vr = orchestrator.execute(context)

        # Extract artifacts for TurnRecord
        trace_messages = context.get_artifact(ArtifactKeys.TRACE_MESSAGES)
        parsed_answer = context.get_artifact(ArtifactKeys.PARSED_ANSWER)
        raw_response = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)

        return vr, trace_messages, parsed_answer, raw_response

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

    Override models inherit request_timeout from the base model when not set,
    so the pipeline-level timeout propagates to per-node model overrides.
    """
    override = node.model_override
    answering = override.answering_model if override and override.answering_model else base_answering
    parsing = override.parsing_model if override and override.parsing_model else base_parsing

    # Propagate request_timeout from base models to overrides that don't set their own
    if answering.request_timeout is None and base_answering.request_timeout is not None:
        answering = answering.model_copy(update={"request_timeout": base_answering.request_timeout})
    if parsing.request_timeout is None and base_parsing.request_timeout is not None:
        parsing = parsing.model_copy(update={"request_timeout": base_parsing.request_timeout})

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
