"""Scenario builder for constructing multi-turn benchmark graphs.

The Scenario class provides a mutable builder API for assembling nodes,
edges, and outcome criteria. Calling validate() freezes the graph into
a ScenarioDefinition after running structural checks.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

from karenina.scenario.source_extraction import extract_callable_source
from karenina.scenario.validation import validate_scenario_graph
from karenina.schemas.entities.question import Question
from karenina.schemas.primitives import (
    BooleanMatch,
    ExactMatch,
    NumericExact,
    NumericRange,
    SetContainment,
    VerificationPrimitive,
)
from karenina.schemas.scenario.checks import OutcomeNode
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.types import (
    EdgeCondition,
    ModelOverride,
    ScenarioEdge,
    ScenarioNode,
    ScenarioOutcomeCriterion,
    StateCheck,
    ToolFilter,
)

logger = logging.getLogger(__name__)


def _normalize_state_check(d: dict[str, Any]) -> StateCheck:
    """Convert a dict shorthand into a StateCheck.

    Supports these forms:
    - ``{"verify_result": True}`` (bool): StateCheck with BooleanMatch
    - ``{"parsed.drug": "venetoclax"}`` (str): StateCheck with ExactMatch
    - ``{"turn": 5}`` (int/float): StateCheck with NumericExact
    - ``{"node_visits.retry": {"gte": 3}}`` (operator dict): StateCheck with range primitive
    - ``{"items": [1, 2]}`` (list): StateCheck with SetContainment

    Args:
        d: A single-key dict mapping a field name to an expected value or
           an operator dict.

    Returns:
        A StateCheck with the appropriate VerificationPrimitive.

    Raises:
        ValueError: If the dict has != 1 key or operator is unknown.
    """
    if len(d) != 1:
        raise ValueError(f"State check dict must have exactly one key, got {len(d)}: {sorted(d.keys())}")

    field, value = next(iter(d.items()))

    if isinstance(value, dict):
        return _normalize_operator_dict(field, value)

    primitive: VerificationPrimitive
    if isinstance(value, bool):
        primitive = BooleanMatch()
    elif isinstance(value, str):
        primitive = ExactMatch()
    elif isinstance(value, int | float):
        primitive = NumericExact()
    elif isinstance(value, list):
        primitive = SetContainment()
    else:
        primitive = ExactMatch()

    return StateCheck(field=field, expected=value, verify_with=primitive)


_OPERATOR_MAP: dict[str, Callable[[Any], VerificationPrimitive]] = {
    "gte": lambda v: NumericRange(min=v),
    "gt": lambda v: NumericRange(min=v, exclusive_min=True),
    "lte": lambda v: NumericRange(max=v),
    "lt": lambda v: NumericRange(max=v, exclusive_max=True),
    "eq": lambda _v: NumericExact(),
}


def _normalize_operator_dict(field: str, ops: dict[str, Any]) -> StateCheck:
    """Convert an operator dict like ``{"gte": 3}`` into a StateCheck.

    Args:
        field: The state field path.
        ops: Dict with a single operator key.

    Returns:
        A StateCheck with the appropriate primitive.

    Raises:
        ValueError: If the operator key is unknown.
    """
    if len(ops) != 1:
        raise ValueError(f"Operator dict must have exactly one key, got {len(ops)}: {sorted(ops.keys())}")

    op, value = next(iter(ops.items()))
    factory = _OPERATOR_MAP.get(op)
    if factory is None:
        raise ValueError(f"Unknown operator '{op}'. Supported: {sorted(_OPERATOR_MAP.keys())}")

    primitive = factory(value)
    # For "eq" operator, set expected; for range operators, expected stays None
    expected = value if op == "eq" else None
    return StateCheck(field=field, expected=expected, verify_with=primitive)


def _compile_callable_string(source: str) -> Any:
    """Compile a source string into a callable.

    Args:
        source: Python expression string (e.g., ``"lambda acc, p: ..."``)

    Returns:
        The compiled callable.

    Raises:
        ValueError: If the source cannot be compiled.
    """
    try:
        return eval(source)  # noqa: S307
    except Exception as exc:
        raise ValueError(f"Cannot compile callable from source string: {source}") from exc


class Scenario:
    """Mutable builder for multi-turn scenario graphs.

    Usage::

        s = Scenario("retry-loop")
        s.add_node("q1", question=q1)
        s.add_node("retry", question=q_retry)
        s.add_edge("q1", "retry", when={"verify_result": False})
        s.add_edge("q1", END, when={"verify_result": True})
        s.add_edge("retry", "q1")
        s.set_entry("q1")
        defn = s.validate()  # -> ScenarioDefinition (frozen)
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._nodes: dict[str, ScenarioNode] = {}
        self._edges: list[ScenarioEdge] = []
        self._entry_node: str | None = None
        self._outcome_criteria: list[ScenarioOutcomeCriterion] = []

    def add_node(
        self,
        node_id: str,
        *,
        question: Question,
        model_override: ModelOverride | None = None,
        tool_filter: ToolFilter | None = None,
        state_update: Any = None,
        agent_identity: str | None = None,
        **metadata: Any,
    ) -> None:
        """Add a node to the scenario graph.

        Args:
            node_id: Unique identifier for this node.
            question: The question for this node (deep copied).
            model_override: Optional per-node model override.
            tool_filter: Optional tool filter for this node.
            state_update: Optional state update callable, lambda, or source string.
            agent_identity: Optional identity label for this node's agent. The
                reserved value ``"__user__"`` is used internally for scenario
                prompts and cannot be assigned here.
            **metadata: Extra metadata key-value pairs stored on the node.

        Raises:
            ValueError: If node_id already exists or agent_identity is reserved.
        """
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists")

        if agent_identity == "__user__":
            raise ValueError("agent_identity='__user__' is reserved for scenario prompts. Choose a different identity.")

        # Deep copy the question to prevent external mutation
        question_copy = question.model_copy(deep=True)

        # Resolve state_update: callable or string
        state_update_fn = None
        state_update_source = None
        if state_update is not None:
            if isinstance(state_update, str):
                state_update_source = state_update
                state_update_fn = _compile_callable_string(state_update)
            else:
                state_update_source = extract_callable_source(state_update)
                state_update_fn = state_update

        self._nodes[node_id] = ScenarioNode(
            node_id=node_id,
            question=question_copy,
            model_override=model_override,
            tool_filter=tool_filter,
            metadata=metadata,
            agent_identity=agent_identity,
            state_update=state_update_fn,
            state_update_source=state_update_source,
        )

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        when: dict[str, Any] | list[dict[str, Any]] | StateCheck | Any | None = None,
        handover: str | Callable[..., Any] | None = None,
    ) -> None:
        """Add an edge between two nodes.

        Args:
            source: Source node_id.
            target: Target node_id or END.
            when: Edge condition. Accepted forms:
                - None: unconditional edge
                - dict: shorthand converted via _normalize_state_check
                - list[dict]: AND'd StateChecks
                - StateCheck: used directly
                - callable: condition callable with source extraction
                - str: compiled to callable
            handover: Optional handover strategy for transcript transfer between
                agents. Accepts a string strategy name (``"transcript_prepend"``,
                ``"transcript_append"``) or a callable ``(msgs, state) -> msgs``.

        Raises:
            ValueError: If condition form or handover strategy is invalid.
        """
        condition: EdgeCondition | None = None
        condition_callable = None
        condition_source = None

        if when is None:
            pass
        elif isinstance(when, StateCheck):
            condition = when
        elif isinstance(when, dict):
            condition = _normalize_state_check(when)
        elif isinstance(when, list):
            condition = [_normalize_state_check(d) for d in when]
        elif isinstance(when, str):
            condition_source = when
            condition_callable = _compile_callable_string(when)
        elif callable(when):
            condition_source = extract_callable_source(when)
            condition_callable = when
        else:
            raise ValueError(f"Unsupported 'when' type: {type(when).__name__}")

        # Handover dispatch
        handover_str: str | None = None
        handover_callable_fn = None

        if handover is not None:
            if isinstance(handover, str):
                _KNOWN_STRATEGIES = {"transcript_prepend", "transcript_append"}
                if handover not in _KNOWN_STRATEGIES:
                    raise ValueError(
                        f"Unknown handover strategy '{handover}'. Known strategies: {sorted(_KNOWN_STRATEGIES)}"
                    )
                handover_str = handover
            elif callable(handover):
                warnings.warn(
                    "Callable handover will not be serialized to checkpoints. "
                    "Use a string strategy for checkpoint persistence.",
                    UserWarning,
                    stacklevel=2,
                )
                handover_callable_fn = handover
            else:
                raise ValueError(f"Unsupported handover type: {type(handover).__name__}")

        self._edges.append(
            ScenarioEdge(
                source=source,
                target=target,
                condition=condition,
                condition_callable=condition_callable,
                condition_source=condition_source,
                handover=handover_str,
                handover_callable=handover_callable_fn,
            )
        )

    def add_outcome_criterion(self, criterion: ScenarioOutcomeCriterion) -> None:
        """Add a pre-built ScenarioOutcomeCriterion.

        If the criterion has an evaluate callable but no evaluate_source,
        the source is extracted automatically.

        Args:
            criterion: The outcome criterion to add.
        """
        if criterion.evaluate is not None and criterion.evaluate_source is None:
            criterion.evaluate_source = extract_callable_source(criterion.evaluate)
        self._outcome_criteria.append(criterion)

    def add_outcome(
        self,
        name: str,
        check: OutcomeNode,
        *,
        description: str = "",
    ) -> None:
        """Sugar for adding an outcome criterion with a declarative check.

        Args:
            name: Name of the outcome.
            check: Declarative check node (TurnCheck, ResultCheck, etc.).
            description: Optional description of the outcome.
        """
        self._outcome_criteria.append(
            ScenarioOutcomeCriterion(
                name=name,
                description=description,
                check=check,
            )
        )

    def set_entry(self, node_id: str) -> None:
        """Set the entry node for the scenario.

        Args:
            node_id: Must be a node that has already been added.

        Raises:
            ValueError: If node_id is not a known node.
        """
        if node_id not in self._nodes:
            raise ValueError(f"'{node_id}' is not a known node")
        self._entry_node = node_id

    def validate(self) -> ScenarioDefinition:
        """Validate the graph and freeze it into a ScenarioDefinition.

        Runs structural checks (reachability, edge validity, fallback
        coverage), then produces a frozen Pydantic model.

        Returns:
            A frozen ScenarioDefinition.

        Raises:
            ValueError: If validation fails (no entry, bad edges, orphans, etc.).
        """
        if self._entry_node is None:
            raise ValueError("No entry node set. Call set_entry() before validate().")

        if self._entry_node not in self._nodes:
            raise ValueError(
                f"Entry node '{self._entry_node}' is not a known node. Known nodes: {sorted(self._nodes.keys())}"
            )

        validate_scenario_graph(self._nodes, self._edges, self._entry_node)

        return ScenarioDefinition(
            name=self.name,
            description=self.description,
            nodes=dict(self._nodes),
            edges=list(self._edges),
            entry_node=self._entry_node,
            outcome_criteria=list(self._outcome_criteria),
        )
