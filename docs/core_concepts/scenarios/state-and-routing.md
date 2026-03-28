---
jupyter:
  jupytext:
    formats: docs/core_concepts/scenarios//md,docs/notebooks/core_concepts/scenarios//ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# State and Routing

`ScenarioState` is the mutable runtime object that accumulates data as a scenario executes. Edge conditions query this state via dot-path expressions to determine which node to visit next. The edge resolution algorithm evaluates conditions in definition order, taking the first match.

This page explains the structure of `ScenarioState`, how the engine updates it after each turn, and which patterns cover common routing needs. For defining the graph structure itself, see [Building Scenarios](building-scenarios.md). For asserting on the completed execution, see [Outcome Criteria](outcome-criteria.md).

```python tags=["hide-cell"]
# Mock setup for documentation: allows notebook to run without API keys.
# This cell is hidden in rendered documentation.
# Rather than importing the real builder (which triggers the full karenina
# package init chain and requires installed dependencies), we define lightweight
# stand-ins that expose the same public API used in the code cells below.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Minimal primitives
# ---------------------------------------------------------------------------

class BooleanMatch:
    type: str = "boolean_match"
    def check(self, value: Any, expected: Any) -> bool:
        return bool(value) == bool(expected)

class ExactMatch:
    type: str = "exact_match"
    def check(self, value: Any, expected: Any) -> bool:
        return value == expected

class NumericExact:
    type: str = "numeric_exact"
    def check(self, value: Any, expected: Any) -> bool:
        return value == expected


# ---------------------------------------------------------------------------
# StateCheck and ScenarioEdge mocks
# ---------------------------------------------------------------------------

class StateCheck:
    type: str = "state_check"
    def __init__(self, field: str, expected: Any = None, verify_with: Any = None) -> None:
        self.field = field
        self.expected = expected
        self.verify_with = verify_with or BooleanMatch()


class ScenarioEdge:
    def __init__(self, source: str, target: str,
                 condition: StateCheck | list[StateCheck] | None = None,
                 condition_callable: Any | None = None,
                 condition_source: str | None = None) -> None:
        self.source = source
        self.target = target
        self.condition = condition
        self.condition_callable = condition_callable
        self.condition_source = condition_source


# ---------------------------------------------------------------------------
# ScenarioState mock
# ---------------------------------------------------------------------------

END: str = "__end__"


@dataclass
class TurnRecord:
    node_id: str
    question_text: str
    raw_response: str
    verify_result: bool | None
    parsed_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioState:
    turn: int
    current_node: str
    verify_result: bool | None
    parsed: dict[str, Any]
    node_visits: dict[str, int]
    history: list[TurnRecord]
    accumulated: dict[str, Any]
    node_results: dict[str, dict[str, Any]]


# ---------------------------------------------------------------------------
# Edge resolution mock
# ---------------------------------------------------------------------------

def _resolve_dot_path(path: str, state: ScenarioState) -> Any:
    parts = path.split(".", 1)
    root = parts[0]
    if root == "verify_result":
        return state.verify_result
    if root == "turn":
        return state.turn
    if root == "current_node":
        return state.current_node
    if len(parts) == 2:
        sub_key = parts[1]
        if root == "parsed":
            return state.parsed.get(sub_key)
        if root == "accumulated":
            return state.accumulated.get(sub_key)
        if root == "node_visits":
            return state.node_visits.get(sub_key, 0)
        if root == "node_results":
            sub_parts = sub_key.split(".", 1)
            node_key = sub_parts[0]
            node_data = state.node_results.get(node_key, {})
            if len(sub_parts) == 1:
                return node_data
            remaining = sub_parts[1]
            rp = remaining.split(".", 1)
            value = node_data.get(rp[0])
            if len(rp) == 2 and isinstance(value, dict):
                return value.get(rp[1])
            return value
    return None


def evaluate_state_check(check: StateCheck, state: ScenarioState) -> bool:
    resolved = _resolve_dot_path(check.field, state)
    return check.verify_with.check(resolved, check.expected)


def _edge_matches(edge: ScenarioEdge, state: ScenarioState) -> bool:
    if edge.condition_callable is not None:
        try:
            return bool(edge.condition_callable(state))
        except Exception:
            return False
    condition = edge.condition
    if condition is None:
        return True
    if isinstance(condition, list):
        return all(evaluate_state_check(c, state) for c in condition)
    return evaluate_state_check(condition, state)


def resolve_next_node(edges: list[ScenarioEdge], state: ScenarioState) -> str | None:
    if not edges:
        return None
    fallback_target: str | None = None
    for edge in edges:
        is_unconditional = edge.condition is None and edge.condition_callable is None
        if is_unconditional:
            if fallback_target is None:
                fallback_target = edge.target
            continue
        if _edge_matches(edge, state):
            return edge.target
    return fallback_target


# ---------------------------------------------------------------------------
# Scenario builder mock (routing methods only)
# ---------------------------------------------------------------------------

class Scenario:
    """Lightweight mock for state and routing documentation examples."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._edges: list[ScenarioEdge] = []

    def add_edge(self, source: str, target: str, *,
                 when: dict[str, Any] | None = None,
                 when_callable: Any | None = None) -> None:
        if when is not None:
            conditions = []
            for f, v in when.items():
                if isinstance(v, bool):
                    prim = BooleanMatch()
                elif isinstance(v, str):
                    prim = ExactMatch()
                else:
                    prim = NumericExact()
                conditions.append(StateCheck(field=f, expected=v, verify_with=prim))
            condition: StateCheck | list[StateCheck] | None = (
                conditions[0] if len(conditions) == 1 else conditions
            )
        else:
            condition = None
        self._edges.append(ScenarioEdge(
            source=source,
            target=target,
            condition=condition,
            condition_callable=when_callable,
        ))

    def edges_from(self, node_id: str) -> list[ScenarioEdge]:
        return [e for e in self._edges if e.source == node_id]


print("Mock setup complete.")
```

## 1. What It Is

`ScenarioState` is the mutable runtime object that accumulates data as a scenario executes. It is created at the start of execution and updated after every turn. After each turn completes, the engine writes the turn's `verify_result`, the parsed template fields, the updated visit count for the current node, and a structured entry in `node_results`. Edge conditions query this object using dot-path strings to determine which node to visit next.

State is not inspected manually during normal use. You interact with it indirectly: by writing dot-path conditions in `add_edge()` calls and by reading `final_state` from the `ScenarioExecutionResult` returned by the runner.

## 2. Core Idea

State accumulates automatically; you define conditions declaratively.

After each turn, the engine updates:

- `verify_result`: the boolean template verification result from the latest turn (or `None` if the template did not run)
- `parsed`: the dictionary of parsed template fields from the latest turn
- `node_visits[node_id]`: incremented by one for the current node
- `node_results[node_id]`: a structured dict containing `verify_result`, `parsed` (template fields), and `rubric` (trait scores) for that node (last-write-wins on revisits)

Edge conditions query these using dot-path strings:

| Path string | Resolved value |
|---|---|
| `"verify_result"` | Latest turn's template verification result |
| `"parsed.drug"` | `parsed["drug"]` from the latest turn |
| `"node_visits.retry"` | Number of times the `retry` node has been visited |
| `"node_results.ask.verify_result"` | Template result from the most recent `ask` turn |

The first conditional edge whose condition matches the current state wins. If no conditional edge matches, the first unconditional edge is used as a fallback. If there are no edges at all, execution terminates.

## 3. Anatomy

`ScenarioState` is a dataclass with the following fields:

| Field | Type | Description |
|---|---|---|
| `turn` | `int` | Current turn number (0-indexed) |
| `current_node` | `str` | Node being executed |
| `verify_result` | `bool \| None` | Template verification result from the latest turn |
| `parsed` | `dict[str, Any]` | Parsed template fields from the latest turn |
| `node_visits` | `dict[str, int]` | Visit count per node |
| `history` | `list[TurnRecord]` | All previous turn records |
| `accumulated` | `dict[str, Any]` | Custom state populated via `state_update` callbacks |
| `node_results` | `dict[str, dict]` | Auto-populated per-node results (verify_result, parsed, rubric) |

`TurnRecord` captures one turn's full output. Its fields are described in the Reference section.

## 4. How It Works

Each turn follows this sequence:

1. Set `state.current_node` to the current node id.
2. Execute the node: run the answering model, then the verification pipeline.
3. Write `state.verify_result` with the template result from this turn.
4. Write `state.parsed` with the parsed template fields from this turn.
5. Increment `state.node_visits[current_node]`.
6. Write `state.node_results[current_node]` with a dict containing `verify_result`, `parsed`, and `rubric`.
7. If the node has a `state_update` callback, call it with the current accumulated dict and parsed fields to produce the new accumulated dict.
8. Append a `TurnRecord` to `state.history`.

After updating state, the engine calls `resolve_next_node` with the outbound edges from the current node and the updated state.

**Edge resolution algorithm:**

1. Collect all edges where `source == current_node`.
2. Iterate in definition order.
3. Save the first unconditional edge (no `condition` and no `condition_callable`) as the fallback.
4. For each conditional edge, evaluate its condition against the current state. Return the target of the first matching edge.
5. If no conditional edge matched, return the fallback target.
6. If no fallback exists, return `None` (implicit terminal; execution ends).

```python
# Dot-path conditions passed to add_edge() become StateCheck objects internally.
# The engine evaluates them in the order the edges were added.

scenario = Scenario("drug-identification")

scenario.add_edge("ask", "retry", when={"verify_result": False})
scenario.add_edge("ask", "deep_dive", when={"parsed.confidence": "high"})
scenario.add_edge("retry", END, when={"node_visits.retry": 3})

# Simulate state after a failed turn on "ask"
state = ScenarioState(
    turn=0,
    current_node="ask",
    verify_result=False,
    parsed={"confidence": "low"},
    node_visits={"ask": 1},
    history=[],
    accumulated={},
    node_results={},
)

ask_edges = scenario.edges_from("ask")
next_node = resolve_next_node(ask_edges, state)
print(f"Next node after failed ask: {next_node}")

# Simulate state after a successful turn with high confidence
state2 = ScenarioState(
    turn=1,
    current_node="ask",
    verify_result=True,
    parsed={"confidence": "high"},
    node_visits={"ask": 2},
    history=[],
    accumulated={},
    node_results={},
)

next_node2 = resolve_next_node(ask_edges, state2)
print(f"Next node after successful ask with high confidence: {next_node2}")
```

## 5. Patterns

### a. Simple routing on verify_result

Route to a retry node if the template failed, or to an end node if it passed:

```python
scenario_a = Scenario("simple-routing")

scenario_a.add_edge("ask", "retry", when={"verify_result": False})
scenario_a.add_edge("ask", END)  # unconditional fallback: reached only if verify_result is True

scenario_a.add_edge("retry", END)

state_pass = ScenarioState(
    turn=0, current_node="ask",
    verify_result=True, parsed={},
    node_visits={"ask": 1}, history=[],
    accumulated={}, node_results={},
)
state_fail = ScenarioState(
    turn=0, current_node="ask",
    verify_result=False, parsed={},
    node_visits={"ask": 1}, history=[],
    accumulated={}, node_results={},
)

print(f"Pass path: {resolve_next_node(scenario_a.edges_from('ask'), state_pass)}")
print(f"Fail path: {resolve_next_node(scenario_a.edges_from('ask'), state_fail)}")
```

### b. Routing on parsed fields

Route based on a field extracted by the answer template. Here the template parses a `confidence` field from the response, and the scenario branches on its value:

```python
scenario_b = Scenario("parsed-routing")

scenario_b.add_edge("ask", "clarify", when={"parsed.confidence": "low"})
scenario_b.add_edge("ask", END)  # fallback for any other confidence value

state_low = ScenarioState(
    turn=0, current_node="ask",
    verify_result=True, parsed={"confidence": "low"},
    node_visits={"ask": 1}, history=[],
    accumulated={}, node_results={},
)
state_high = ScenarioState(
    turn=0, current_node="ask",
    verify_result=True, parsed={"confidence": "high"},
    node_visits={"ask": 1}, history=[],
    accumulated={}, node_results={},
)

print(f"Low confidence path: {resolve_next_node(scenario_b.edges_from('ask'), state_low)}")
print(f"High confidence path: {resolve_next_node(scenario_b.edges_from('ask'), state_high)}")
```

### c. Custom accumulated state via state_update

`ScenarioNode` accepts an optional `state_update` callable. After each turn on that node, the engine calls `state_update(accumulated, parsed_fields)` and replaces `accumulated` with the returned dict. This allows tracking values across revisits, such as a running count.

Before calling the callback, the engine takes a `deepcopy` snapshot of `accumulated`. If the callback raises an exception, the snapshot is restored and the error is logged at `warning` level. This guarantees that a buggy `state_update` cannot corrupt accumulated state for subsequent turns.

```python
# In real usage, state_update is attached to a ScenarioNode.
# Here we demonstrate the accumulation pattern directly.

def track_attempts(accumulated: dict, parsed_fields: dict) -> dict:
    """Increment attempt counter on each visit."""
    return {**accumulated, "attempts": accumulated.get("attempts", 0) + 1}

# Simulate three visits
acc: dict = {}
for _ in range(3):
    acc = track_attempts(acc, {})

print(f"Accumulated after 3 visits: {acc}")

# An edge can then route on accumulated.attempts:
scenario_c = Scenario("accumulated-routing")
scenario_c.add_edge("probe", END, when={"accumulated.attempts": 3})
scenario_c.add_edge("probe", "probe")  # fallback: loop back

state_3 = ScenarioState(
    turn=2, current_node="probe",
    verify_result=None, parsed={},
    node_visits={"probe": 3},
    history=[],
    accumulated={"attempts": 3},
    node_results={},
)
print(f"At 3 attempts, next node: {resolve_next_node(scenario_c.edges_from('probe'), state_3)}")
```

### d. Cross-node result lookups via node_results

`node_results` lets edges on a later node reference results from an earlier node. This is useful in multi-branch scenarios where a final synthesis node needs to know how an earlier node resolved:

```python
scenario_d = Scenario("cross-node-routing")

# The "synthesize" node routes based on whether the earlier "ask" node passed.
scenario_d.add_edge("synthesize", "explain", when={"node_results.ask.verify_result": False})
scenario_d.add_edge("synthesize", END)

state_ask_failed = ScenarioState(
    turn=2, current_node="synthesize",
    verify_result=True, parsed={},
    node_visits={"ask": 1, "synthesize": 1},
    history=[],
    accumulated={},
    node_results={
        "ask": {"verify_result": False, "parsed": {}, "rubric": {}},
    },
)
state_ask_passed = ScenarioState(
    turn=2, current_node="synthesize",
    verify_result=True, parsed={},
    node_visits={"ask": 1, "synthesize": 1},
    history=[],
    accumulated={},
    node_results={
        "ask": {"verify_result": True, "parsed": {}, "rubric": {}},
    },
)

print(f"ask failed: next is {resolve_next_node(scenario_d.edges_from('synthesize'), state_ask_failed)}")
print(f"ask passed: next is {resolve_next_node(scenario_d.edges_from('synthesize'), state_ask_passed)}")
```

## 6. Reference

### Dot-path resolution

| Path | Resolves to |
|---|---|
| `"verify_result"` | `state.verify_result` |
| `"turn"` | `state.turn` |
| `"current_node"` | `state.current_node` |
| `"parsed.<field>"` | `state.parsed.get(field)` |
| `"accumulated.<field>"` | `state.accumulated.get(field)` |
| `"node_visits.<node_id>"` | `state.node_visits.get(node_id, 0)` |
| `"node_results.<node>"` | `state.node_results.get(node, {})` |
| `"node_results.<node>.verify_result"` | `state.node_results[node]["verify_result"]` |
| `"node_results.<node>.parsed.<field>"` | `state.node_results[node]["parsed"][field]` |
| `"node_results.<node>.rubric.<trait>"` | `state.node_results[node]["rubric"][trait]` |

Missing keys return `None`, except `node_visits.<node_id>`, which returns `0`.

### TurnRecord fields

| Field | Type | Description |
|---|---|---|
| `node_id` | `str` | The node that produced this turn |
| `question_text` | `str` | The rendered question text sent to the model |
| `raw_response` | `str` | The model's raw response text |
| `parsed_fields` | `dict[str, Any]` | Template fields parsed from the response |
| `verify_result` | `bool \| None` | Template verification result for this turn |

### ScenarioExecutionResult fields

| Field | Type | Description |
|---|---|---|
| `scenario_id` | `str` | Identifier for the scenario |
| `status` | `"completed" \| "limit_reached" \| "error"` | How execution terminated |
| `path` | `list[str]` | Ordered list of node ids visited |
| `turn_count` | `int` | Total number of turns executed |
| `history` | `list[TurnRecord]` | Full turn history |
| `turn_results` | `list[VerificationResult]` | Per-turn verification result objects |
| `final_state` | `ScenarioState` | State as of the last turn |
| `outcome_results` | `dict[str, bool \| int \| float]` | Evaluated outcome criteria, keyed by name |

### Turn limit

`VerificationConfig.scenario_turn_limit` (default: `20`) sets the maximum number of turns before the runner terminates the scenario with `status="limit_reached"`. Set this field in your `VerificationConfig` to override the default.

### Evaluation mode

Scenarios auto-detect evaluation mode per turn based on whether a rubric is present (per-question rubric or global rubric). Setting `evaluation_mode='rubric_only'` on `VerificationConfig` has no effect in scenarios: the ScenarioManager ignores it and emits a `UserWarning` explaining why. If you need rubric-only evaluation, attach rubrics to your questions without answer templates and the auto-detection will select `template_and_rubric` or `template_only` as appropriate.

## 7. Next Steps

- [Sycophancy Tutorial](../../../workflows/scenarios/sycophancy-tutorial.md): end-to-end walkthrough of a sycophancy resistance scenario that uses state-driven routing
- Scenario Internals: contributor-level detail on the execution engine is in the karenina-guide skill reference at `references/advanced/scenario-internals.md`, not a separate docs page
- [Building Scenarios](building-scenarios.md): constructing the graph, adding nodes and edges, serialization
