"""Mutable runtime state types for scenario execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from karenina.ports.messages import Message
    from karenina.replay import ReplayStore
    from karenina.schemas.entities.answer import BaseAnswer
    from karenina.schemas.verification import VerificationResult


@dataclass
class TurnRecord:
    """Captures the full result of a single turn for history reconstruction."""

    node_id: str
    question_text: str
    question_messages: list[Message]
    trace_messages: list[Message]
    raw_response: str
    parsed_answer: BaseAnswer | None
    parsed_fields: dict[str, Any]
    verify_result: bool | None
    verification_result_id: str | None


@dataclass
class ScenarioState:
    """Accumulated state that flows through the scenario graph.

    Drives routing decisions via dot-path resolution in declarative conditions.
    ``node_results`` is auto-populated after each turn with a structured dict
    containing ``verify_result``, ``parsed`` (template fields), and ``rubric``
    (trait scores), keyed by node_id (last-write-wins on revisits).
    ``accumulated`` is opt-in custom state managed by ``state_update`` callables.
    """

    turn: int
    current_node: str
    verify_result: bool | None
    parsed: dict[str, Any]
    node_visits: dict[str, int]
    history: list[TurnRecord]
    accumulated: dict[str, Any]
    node_results: dict[str, dict[str, Any]]


@dataclass
class ScenarioExecutionResult:
    """Computed wrapper returned by ScenarioManager.run().

    Not stored in DB directly; per-turn VerificationResults are stored
    individually with scenario linking metadata.
    """

    scenario_id: str
    status: Literal["completed", "limit_reached", "error", "timeout"]
    path: list[str]
    turn_count: int
    history: list[TurnRecord]
    turn_results: list[VerificationResult]
    final_state: ScenarioState
    outcome_results: dict[str, bool | int | float]

    def to_replay_store(self, *, answering_model_id: str, **kwargs: Any) -> ReplayStore:
        """Build a ReplayStore from this single scenario execution.

        ``answering_model_id`` is required because this result does not
        carry per-turn model identity. See
        :py:func:`karenina.replay.capture.capture_from_scenario_result`
        for additional keyword arguments.
        """
        from karenina.replay.capture import capture_from_scenario_result

        return capture_from_scenario_result(self, answering_model_id=answering_model_id, **kwargs)
