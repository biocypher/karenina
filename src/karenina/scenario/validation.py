"""Graph validation for scenario definitions.

Checks structural integrity of a scenario graph before freezing it into
a ScenarioDefinition: reachability, edge validity, and fallback coverage.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict

from karenina.schemas.scenario.types import END, ScenarioEdge, ScenarioNode

logger = logging.getLogger(__name__)


def validate_scenario_graph(
    nodes: dict[str, ScenarioNode],
    edges: list[ScenarioEdge],
    entry_node: str,
) -> None:
    """Validate structural integrity of a scenario graph.

    Checks:
    1. Edge sources and targets reference existing nodes (or END).
    2. No orphan nodes (unreachable from entry via BFS).
    3. Every node with conditional edges also has an unconditional fallback.
    4. Implicit terminals (nodes with no outbound edges) are valid.

    Args:
        nodes: Mapping of node_id to ScenarioNode.
        edges: List of scenario edges.
        entry_node: The entry node id.

    Raises:
        ValueError: If any validation check fails.
    """
    node_ids = set(nodes.keys())
    valid_targets = node_ids | {END}

    # 1. Check edge references
    for edge in edges:
        if edge.source not in node_ids:
            raise ValueError(f"Edge source '{edge.source}' is not a known node. Known nodes: {sorted(node_ids)}")
        if edge.target not in valid_targets:
            raise ValueError(f"Edge target '{edge.target}' is not a known node or END. Known nodes: {sorted(node_ids)}")

    # 2. Reachability: BFS from entry_node
    outbound: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.target != END:
            outbound[edge.source].append(edge.target)

    reachable: set[str] = set()
    queue = [entry_node]
    while queue:
        current = queue.pop(0)
        if current in reachable:
            continue
        reachable.add(current)
        for target in outbound.get(current, []):
            if target not in reachable:
                queue.append(target)

    orphans = node_ids - reachable
    if orphans:
        raise ValueError(f"Orphan nodes not reachable from entry '{entry_node}': {sorted(orphans)}")

    # 3. Conditional fallback check
    # Group edges by source, then check nodes that have conditional edges
    edges_by_source: dict[str, list[ScenarioEdge]] = defaultdict(list)
    for edge in edges:
        edges_by_source[edge.source].append(edge)

    for source, source_edges in edges_by_source.items():
        has_conditional = any(e.condition is not None or e.condition_callable is not None for e in source_edges)
        has_unconditional = any(e.condition is None and e.condition_callable is None for e in source_edges)
        if has_conditional and not has_unconditional:
            raise ValueError(
                f"Node '{source}' has conditional edges but no unconditional fallback. "
                "Add an unconditional edge (when=None) as a default path."
            )

    # 4. Multiple unconditional edges check
    for source, source_edges in edges_by_source.items():
        unconditional = [e for e in source_edges if e.condition is None and e.condition_callable is None]
        if len(unconditional) > 1:
            targets = [e.target for e in unconditional]
            warnings.warn(
                f"Node '{source}' has {len(unconditional)} unconditional edges "
                f"(targets: {targets}). Only the first will be used; "
                f"the rest are silently ignored. This is likely a graph error.",
                UserWarning,
                stacklevel=2,
            )
