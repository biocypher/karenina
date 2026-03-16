"""Conversion functions between ScenarioDefinition and SchemaOrg checkpoint format.

Handles serialization (scenario_to_schema_org) for scenario benchmark
persistence. The reverse direction (schema_org_to_scenario) is implemented
separately.
"""

from __future__ import annotations

import logging
from typing import Any

from karenina.schemas.checkpoint import (
    SchemaOrgAnswer,
    SchemaOrgQuestion,
    SchemaOrgScenario,
    SchemaOrgScenarioEdge,
    SchemaOrgScenarioNode,
    SchemaOrgScenarioOutcome,
    SchemaOrgSoftwareSourceCode,
)
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
