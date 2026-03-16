"""Conversion functions between ScenarioDefinition and SchemaOrg checkpoint format.

Handles serialization (scenario_to_schema_org) and deserialization
(schema_org_to_scenario) for scenario benchmark persistence.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import TypeAdapter

from karenina.scenario.builder import _compile_callable_string
from karenina.schemas.checkpoint import (
    SchemaOrgAnswer,
    SchemaOrgQuestion,
    SchemaOrgScenario,
    SchemaOrgScenarioEdge,
    SchemaOrgScenarioNode,
    SchemaOrgScenarioOutcome,
    SchemaOrgSoftwareSourceCode,
)
from karenina.schemas.primitives.registry import _reconstruct_primitive
from karenina.schemas.scenario.checks import OutcomeNode
from karenina.schemas.scenario.definition import ScenarioDefinition
from karenina.schemas.scenario.types import (
    ScenarioEdge,
    ScenarioNode,
    ScenarioOutcomeCriterion,
    StateCheck,
)

logger = logging.getLogger(__name__)


def _question_to_schema_org(node: ScenarioNode) -> SchemaOrgQuestion:
    """Convert a ScenarioNode's Question to SchemaOrgQuestion."""
    q = node.question
    name = f"{q.question[:40]}... Answer Template" if len(q.question) > 40 else f"{q.question} Answer Template"
    return SchemaOrgQuestion(
        text=q.question,
        acceptedAnswer=SchemaOrgAnswer(text=q.raw_answer or ""),
        hasPart=SchemaOrgSoftwareSourceCode(
            name=name,
            text=q.answer_template or "",
        ),
    )


def _serialize_verify_with(primitive: Any) -> dict[str, Any]:
    """Serialize a VerificationPrimitive, injecting the class name as 'type'.

    VerificationPrimitive is NOT a Pydantic discriminated union: model_dump()
    produces an empty dict for primitives like BooleanMatch. We must manually
    inject the class name so the registry can reconstruct them on load.
    """
    data = primitive.model_dump() if hasattr(primitive, "model_dump") else {}
    data["type"] = type(primitive).__name__
    return data


def _serialize_state_check(check: StateCheck) -> dict[str, Any]:
    """Serialize a StateCheck with proper verify_with type injection."""
    data = check.model_dump()
    if check.verify_with is not None:
        data["verify_with"] = _serialize_verify_with(check.verify_with)
    return data


def _serialize_condition(
    condition: StateCheck | list[StateCheck] | None,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Serialize an EdgeCondition (StateCheck or list[StateCheck])."""
    if condition is None:
        return None
    if isinstance(condition, list):
        return [_serialize_state_check(c) for c in condition]
    return _serialize_state_check(condition)


def _inject_verify_with_types(obj: Any) -> dict[str, Any]:
    """Serialize an OutcomeCheckNode tree, injecting type keys throughout.

    model_dump() on check nodes uses Pydantic Literal type discriminators
    (e.g., "all_of", "turn_check"). For checkpoint storage we replace these
    with the Python class name (e.g., "AllOf", "TurnCheck") so the loader
    can reconstruct the correct types.

    VerificationPrimitive subclasses have no type field at all; model_dump()
    returns an empty dict. We walk the live Pydantic objects (not dicts) to
    inject the class name before serialization.
    """
    data = obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)

    # Replace Pydantic literal type with Python class name
    data["type"] = type(obj).__name__

    # Inject verify_with type from the live object
    if hasattr(obj, "verify_with") and obj.verify_with is not None:
        data["verify_with"] = _serialize_verify_with(obj.verify_with)

    # Recurse into conditions (AllOf, AnyOf, AtLeastN)
    if hasattr(obj, "conditions") and obj.conditions:
        data["conditions"] = [_inject_verify_with_types(c) for c in obj.conditions]

    return data


def _serialize_edge(edge: ScenarioEdge) -> SchemaOrgScenarioEdge:
    """Convert a ScenarioEdge to SchemaOrgScenarioEdge."""
    return SchemaOrgScenarioEdge(
        source=edge.source,
        target=edge.target,
        condition=_serialize_condition(edge.condition),
        conditionSource=edge.condition_source,
    )


def _serialize_outcome(
    criterion: ScenarioOutcomeCriterion,
) -> SchemaOrgScenarioOutcome:
    """Convert a ScenarioOutcomeCriterion to SchemaOrgScenarioOutcome."""
    check_data = None
    if criterion.check is not None:
        # Walk the live object tree to inject verify_with types before dumping
        check_data = _inject_verify_with_types(criterion.check)
    return SchemaOrgScenarioOutcome(
        name=criterion.name,
        description=criterion.description,
        check=check_data,
        evaluateSource=criterion.evaluate_source,
    )


def scenario_to_schema_org(defn: ScenarioDefinition) -> SchemaOrgScenario:
    """Convert a ScenarioDefinition to SchemaOrgScenario for checkpoint storage.

    Args:
        defn: The frozen scenario definition.

    Returns:
        A SchemaOrgScenario ready for JSON-LD serialization.
    """
    nodes = {}
    for node_id, node in defn.nodes.items():
        nodes[node_id] = SchemaOrgScenarioNode(
            nodeId=node_id,
            question=_question_to_schema_org(node),
            modelOverride=(node.model_override.model_dump() if node.model_override else None),
            toolFilter=(node.tool_filter.model_dump() if node.tool_filter else None),
            stateUpdateSource=node.state_update_source,
            metadata=node.metadata or {},
        )

    return SchemaOrgScenario(
        name=defn.name,
        description=defn.description,
        entryNode=defn.entry_node,
        nodes=nodes,
        edges=[_serialize_edge(e) for e in defn.edges],
        outcomeCriteria=[_serialize_outcome(c) for c in defn.outcome_criteria],
        metadata=defn.metadata or {},
    )


# ---------------------------------------------------------------------------
# Deserialization: SchemaOrgScenario -> ScenarioDefinition
# ---------------------------------------------------------------------------

# Map from Python class name (used in checkpoints) to Pydantic Literal
# discriminator value used in the OutcomeNode union.
_CLASS_TO_DISCRIMINATOR: dict[str, str] = {
    "AllOf": "all_of",
    "AnyOf": "any_of",
    "AtLeastN": "at_least_n",
    "TurnCheck": "turn_check",
    "ResultCheck": "result_check",
    "CrossTurnCheck": "cross_turn_check",
    "CountTurns": "count_turns",
    "FirstMatchIndex": "first_match_index",
}


def _restore_verify_with_in_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Walk a dict tree and restore verify_with primitives via the registry.

    Also converts Python class names back to Pydantic Literal discriminator
    values so the OutcomeNode union can validate correctly.

    Modifies the dict in place.
    """
    # Convert class name type to discriminator value
    if "type" in data and data["type"] in _CLASS_TO_DISCRIMINATOR:
        data["type"] = _CLASS_TO_DISCRIMINATOR[data["type"]]

    if "verify_with" in data and isinstance(data["verify_with"], dict):
        data["verify_with"] = _reconstruct_primitive(data["verify_with"])

    if "conditions" in data and isinstance(data["conditions"], list):
        for item in data["conditions"]:
            if isinstance(item, dict):
                _restore_verify_with_in_dict(item)

    # Restore scope discriminator values (already lowercase, but handle
    # potential class name variants for safety)
    if "scope" in data and isinstance(data["scope"], dict):
        scope_type = data["scope"].get("type", "")
        scope_map = {
            "LastTurn": "last_turn",
            "FirstTurn": "first_turn",
            "TurnAt": "turn_at",
            "AnyTurn": "any_turn",
            "AllTurns": "all_turns",
        }
        if scope_type in scope_map:
            data["scope"]["type"] = scope_map[scope_type]

    return data


def _deserialize_single_condition(data: dict[str, Any]) -> StateCheck:
    """Deserialize a single StateCheck, restoring verify_with via registry."""
    data = dict(data)  # Copy to avoid mutating the checkpoint
    if "verify_with" in data and isinstance(data["verify_with"], dict):
        data["verify_with"] = _reconstruct_primitive(data["verify_with"])
    return StateCheck.model_validate(data)


def _deserialize_condition(
    condition_data: dict[str, Any] | list[dict[str, Any]] | None,
) -> StateCheck | list[StateCheck] | None:
    """Deserialize an edge condition from checkpoint dict."""
    if condition_data is None:
        return None
    if isinstance(condition_data, list):
        return [_deserialize_single_condition(d) for d in condition_data]
    return _deserialize_single_condition(condition_data)


_outcome_adapter: TypeAdapter[OutcomeNode] = TypeAdapter(OutcomeNode)


def _deserialize_outcome_check(data: dict[str, Any]) -> OutcomeNode:
    """Deserialize an outcome check, restoring all nested verify_with primitives."""
    data = _restore_verify_with_in_dict(dict(data))
    result: OutcomeNode = _outcome_adapter.validate_python(data)
    return result


def schema_org_to_scenario(schema: SchemaOrgScenario) -> ScenarioDefinition:
    """Convert a SchemaOrgScenario back to a ScenarioDefinition.

    Reconstructs Question objects, edge conditions with concrete primitives,
    outcome checks, and recompiles callable source strings.

    Args:
        schema: The checkpoint scenario data.

    Returns:
        A frozen ScenarioDefinition ready for execution.
    """
    from karenina.schemas.entities.question import Question
    from karenina.schemas.scenario.types import (
        ModelOverride,
        ScenarioOutcomeCriterion,
        ToolFilter,
    )

    nodes: dict[str, ScenarioNode] = {}
    for node_id, snode in schema.nodes.items():
        q = Question(
            question=snode.question.text,
            raw_answer=snode.question.acceptedAnswer.text,
            answer_template=snode.question.hasPart.text,
        )

        state_update_fn = None
        if snode.stateUpdateSource:
            state_update_fn = _compile_callable_string(snode.stateUpdateSource)

        model_override = None
        if snode.modelOverride:
            model_override = ModelOverride.model_validate(snode.modelOverride)

        tool_filter = None
        if snode.toolFilter:
            tool_filter = ToolFilter.model_validate(snode.toolFilter)

        nodes[node_id] = ScenarioNode(
            node_id=node_id,
            question=q,
            model_override=model_override,
            tool_filter=tool_filter,
            state_update=state_update_fn,
            state_update_source=snode.stateUpdateSource,
            metadata=snode.metadata or {},
        )

    edges: list[ScenarioEdge] = []
    for sedge in schema.edges:
        condition_callable = None
        if sedge.conditionSource:
            condition_callable = _compile_callable_string(sedge.conditionSource)

        edges.append(
            ScenarioEdge(
                source=sedge.source,
                target=sedge.target,
                condition=_deserialize_condition(sedge.condition),
                condition_callable=condition_callable,
                condition_source=sedge.conditionSource,
            )
        )

    outcome_criteria: list[ScenarioOutcomeCriterion] = []
    for soutcome in schema.outcomeCriteria:
        check = None
        if soutcome.check:
            check = _deserialize_outcome_check(soutcome.check)

        evaluate_fn = None
        if soutcome.evaluateSource:
            evaluate_fn = _compile_callable_string(soutcome.evaluateSource)

        outcome_criteria.append(
            ScenarioOutcomeCriterion(
                name=soutcome.name,
                description=soutcome.description,
                check=check,
                evaluate=evaluate_fn,
                evaluate_source=soutcome.evaluateSource,
            )
        )

    return ScenarioDefinition(
        name=schema.name,
        description=schema.description,
        nodes=nodes,
        edges=edges,
        entry_node=schema.entryNode,
        outcome_criteria=outcome_criteria,
        metadata=schema.metadata or {},
    )
