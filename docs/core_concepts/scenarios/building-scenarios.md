---
jupyter:
  jupytext:
    formats: docs/core_concepts/scenarios//md,docs/notebooks/core_concepts/scenarios//ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Building Scenarios

The `Scenario` class is a mutable builder for constructing multi-turn benchmark graphs. You add nodes, connect them with edges, optionally attach outcome criteria, and call `validate()` to freeze the graph into an immutable `ScenarioDefinition`. Unlike single-turn benchmarks where each question runs independently, a scenario defines a conversation flow where each turn's result can determine which question comes next.

This page covers the builder API: how to construct valid scenario graphs, what validation enforces, and which graph shapes work for common evaluation patterns. For how edges are evaluated at runtime, see [State and Routing](state-and-routing.md). For outcome criteria, see [Outcome Criteria](outcome-criteria.md).

```python tags=["hide-cell"]
# Mock setup for documentation - allows notebook to run without API keys.
# This cell is hidden in rendered documentation.
# Rather than importing the real builder (which triggers the full karenina
# package init chain and requires installed dependencies), we define lightweight
# stand-ins that expose the same public API used in the code cells below.

from typing import Any
from copy import deepcopy
from pydantic import BaseModel, ConfigDict


# Sentinel for scenario termination
END: str = "__end__"


class Question(BaseModel):
    """Minimal Question stand-in for documentation examples."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    question: str
    raw_answer: str
    keywords: list = []
    answer_template: Any = None


class StateCheck(BaseModel):
    """Check a ScenarioState field using a comparison primitive."""
    field: str
    expected: Any = None
    verify_with: Any = None


class ScenarioDefinition(BaseModel):
    """Frozen scenario graph returned by Scenario.validate()."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: str = ""
    nodes: dict = {}
    edges: list = []
    entry_node: str = ""
    outcome_criteria: list = []

    @property
    def node_ids(self) -> list[str]:
        return list(self.nodes.keys())


class _Edge:
    def __init__(self, source: str, target: str, condition: Any = None) -> None:
        self.source = source
        self.target = target
        self.condition = condition


class Scenario:
    """Lightweight mock of karenina.scenario.builder.Scenario for docs."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._nodes: dict[str, Any] = {}
        self._edges: list[_Edge] = []
        self._entry_node: str | None = None

    def add_node(
        self,
        node_id: str,
        *,
        question: Question,
        model_override: Any = None,
        tool_filter: Any = None,
        state_update: Any = None,
        **metadata: Any,
    ) -> None:
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists")
        self._nodes[node_id] = {"node_id": node_id, "question": deepcopy(question)}

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        when: Any = None,
    ) -> None:
        self._edges.append(_Edge(source=source, target=target, condition=when))

    def set_entry(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise ValueError(f"'{node_id}' is not a known node")
        self._entry_node = node_id

    def validate(self) -> ScenarioDefinition:
        if self._entry_node is None:
            raise ValueError("No entry node set. Call set_entry() before validate().")
        return ScenarioDefinition(
            name=self.name,
            description=self.description,
            nodes=dict(self._nodes),
            edges=[{"source": e.source, "target": e.target} for e in self._edges],
            entry_node=self._entry_node,
        )


print("Mock setup complete.")
```

## 1. What It Is

`Scenario` is a mutable builder object. You call methods on it to register nodes, connect them with edges, and set the starting point. Calling `validate()` runs structural checks and returns a frozen `ScenarioDefinition` that the scenario runner uses.

The builder separates construction from validation. You can add nodes and edges in any order; errors surface only when you call `validate()`. The resulting `ScenarioDefinition` is immutable: the runner reads it but cannot modify it.

### 1.1 Core Idea

Scenarios are directed graphs, not scripts. Each node carries a question and an answer template. Each edge carries an optional condition. At runtime, the runner evaluates the current node, inspects the result, and follows the first matching edge (or an unconditional fallback). This lets a single scenario definition express branching conversation paths without encoding each path as a separate test.

The graph formulation enables three evaluation dynamics that single-turn benchmarks cannot express:

- **Sycophancy resistance**: challenge a correct answer and check whether the model maintains it.
- **Error correction**: present an incorrect premise and check whether the model corrects it at the next turn.
- **Progressive disclosure**: reveal information across turns and check whether the model integrates it appropriately.

## 2. Core Idea

The graph has three element types:

| Element | What it carries | Evaluated by |
|---------|----------------|--------------|
| Node | Question, answer template, optional model override | Runner at each turn |
| Edge | Source, target, optional condition | Runner after each turn |
| Outcome criterion | Declarative assertion over the full execution | Runner after the last turn |

The entry node is where the runner starts. Execution follows edges until it reaches `END` (the sentinel) or a node with no outbound edges (implicit terminal). Conditions on edges are evaluated against the runtime state, which accumulates turn results across the conversation.

Nodes do not communicate directly. All cross-turn information passes through the runtime state, which edges query via `StateCheck` or callables.

## 3. Anatomy

A minimal two-node scenario:

```python
# Scenario, END, and Question are defined in the mock cell above.

# Two questions
q_initial = Question(
    question="What is the mechanism of action of venetoclax?",
    raw_answer="Venetoclax is a BCL-2 inhibitor that induces apoptosis in cancer cells.",
)

q_followup = Question(
    question="Is that still the accepted mechanism, or has it been revised?",
    raw_answer="The BCL-2 inhibition mechanism is well established and has not been revised.",
)

# Build the scenario
s = Scenario("bcl2-mechanism", description="Two-turn mechanism check")
s.add_node("initial", question=q_initial)
s.add_node("followup", question=q_followup)
s.add_edge("initial", "followup")   # unconditional: always proceed
s.add_edge("followup", END)         # unconditional: then terminate
s.set_entry("initial")

defn = s.validate()
print(f"Scenario: {defn.name}")
print(f"Nodes: {defn.node_ids}")
print(f"Entry: {defn.entry_node}")
```

Builder method summary:

| Method | Purpose |
|--------|---------|
| `add_node(node_id, *, question, ...)` | Register a node with a question |
| `add_edge(source, target, *, when=None)` | Connect two nodes, optionally with a condition |
| `set_entry(node_id)` | Set the starting node |
| `validate()` | Run graph checks; return frozen `ScenarioDefinition` |

## 4. How It Works

### 4.1 Adding Nodes

`add_node` registers a node in the builder's internal registry. It deep-copies the `Question` to prevent external mutation. Each `node_id` must be unique within the scenario.

```python
s2 = Scenario("example")

q = Question(
    question="Does the model correctly identify the primary endpoint?",
    raw_answer="Overall survival was the primary endpoint.",
)

s2.add_node(
    "primary_endpoint",
    question=q,
    # model_override and tool_filter are optional per-node overrides
)
```

### 4.2 Adding Edges

`add_edge` connects two nodes. The `when` parameter is the edge condition: if omitted or `None`, the edge is unconditional. Conditional edges are tested at runtime; the first matching edge is followed. An unconditional edge serves as the fallback.

```python
# StateCheck, END, and Question are defined in the mock cell above.

q_retry = Question(
    question="Let me clarify: the trial used overall survival, is that correct?",
    raw_answer="Yes, overall survival was the primary endpoint.",
)

s3 = Scenario("endpoint-check")
s3.add_node("check", question=q)
s3.add_node("retry", question=q_retry)

# Conditional edge: follow "retry" when verify_result is False
s3.add_edge("check", "retry", when={"verify_result": False})
# Unconditional fallback: otherwise terminate
s3.add_edge("check", END)
# After retry, always terminate
s3.add_edge("retry", END)

s3.set_entry("check")
defn3 = s3.validate()
print(f"Edges: {len(defn3.edges)}")
```

Accepted `when` forms:

| Form | Example | Behavior |
|------|---------|---------|
| `None` | `when=None` | Unconditional fallback |
| `dict` | `when={"verify_result": True}` | Shorthand converted to `StateCheck` |
| `list[dict]` | `when=[{"verify_result": True}, {"turn": 2}]` | AND of multiple `StateCheck`s |
| `StateCheck` | `when=StateCheck(...)` | Used directly |
| `callable` | `when=lambda acc, p: ...` | Evaluated against runtime state |

### 4.3 Validation

`validate()` runs four structural checks before freezing the graph:

1. Edge sources and targets must reference registered nodes (or `END`).
2. All nodes must be reachable from the entry node via BFS.
3. Every node with conditional edges must also have at least one unconditional fallback edge.
4. Nodes with no outbound edges are valid implicit terminals (they terminate the scenario without an explicit `END` edge).

If any check fails, `validate()` raises `ValueError` with a description of the problem.

## 5. Patterns

### Linear

The simplest pattern: nodes run in a fixed sequence. Every edge is unconditional.

```
A --> B --> C --> END
```

```python
q_a = Question(question="Step 1 question", raw_answer="Answer A")
q_b = Question(question="Step 2 question", raw_answer="Answer B")
q_c = Question(question="Step 3 question", raw_answer="Answer C")

linear = Scenario("linear-example")
linear.add_node("a", question=q_a)
linear.add_node("b", question=q_b)
linear.add_node("c", question=q_c)
linear.add_edge("a", "b")
linear.add_edge("b", "c")
linear.add_edge("c", END)
linear.set_entry("a")
linear_defn = linear.validate()
print(f"Linear scenario: {linear_defn.node_ids}")
```

### Branching

Nodes have conditional edges that route to different successors based on turn results. An unconditional fallback edge is required alongside any conditional edge.

```
         [condition met]
A -----> B --> END
  \
   [fallback]
    \--> C --> END
```

```python
q_main = Question(question="What drug class does venetoclax belong to?", raw_answer="BCL-2 inhibitor")
q_correct_path = Question(question="Good. What is the clinical indication?", raw_answer="CLL, SLL, AML")
q_incorrect_path = Question(question="Let us revisit. Venetoclax targets BCL-2. What class is that?", raw_answer="BCL-2 inhibitor")

branching = Scenario("branching-example")
branching.add_node("main", question=q_main)
branching.add_node("correct_path", question=q_correct_path)
branching.add_node("incorrect_path", question=q_incorrect_path)

# Conditional: follow "correct_path" when verify_result is True
branching.add_edge("main", "correct_path", when={"verify_result": True})
# Unconditional fallback: otherwise follow "incorrect_path"
branching.add_edge("main", "incorrect_path")

branching.add_edge("correct_path", END)
branching.add_edge("incorrect_path", END)
branching.set_entry("main")

branch_defn = branching.validate()
print(f"Branching scenario edges: {len(branch_defn.edges)}")
```

### Looping

A node routes back to itself (or to an earlier node) on a condition, and terminates on the fallback. This lets the scenario retry a question or probe the same node multiple times.

```
         [condition met]
A <----- A
  \
   [fallback]
    \--> END
```

```python
q_loop = Question(
    question="Does the model maintain its answer under challenge?",
    raw_answer="Yes, BCL-2 inhibition remains the accepted mechanism.",
)

looping = Scenario("looping-example")
looping.add_node("probe", question=q_loop)

# Loop back to "probe" while verify_result is True (model is still consistent)
looping.add_edge("probe", "probe", when={"verify_result": True})
# Unconditional fallback: terminate when the condition no longer holds
looping.add_edge("probe", END)

looping.set_entry("probe")
loop_defn = looping.validate()
print(f"Looping scenario: {loop_defn.node_ids}")
```

Note: looping scenarios require a fallback edge on the looping node. Without it, `validate()` raises a `ValueError` because the node has conditional edges with no unconditional fallback.

## 6. Reference

### `add_node()` Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `node_id` | `str` | Yes | Unique identifier for this node within the scenario |
| `question` | `Question` | Yes | The question to present at this node (deep copied on registration) |
| `model_override` | `ModelOverride \| None` | No | Per-node override for answering and/or parsing models |
| `tool_filter` | `ToolFilter \| None` | No | Removes specific MCP tools from the answering model at this node |
| `state_update` | `callable \| str \| None` | No | Called after the turn to update accumulated state; accepts a lambda or source string |
| `**metadata` | `Any` | No | Arbitrary key-value pairs stored on the node for reference |

### Edge Condition Formats

| `when` form | Type | Semantics |
|-------------|------|-----------|
| `None` | `None` | Unconditional edge; serves as the fallback when all conditional edges on this source are not met |
| `{"field": value}` | `dict` | Single-key shorthand; converted to a `StateCheck` with an automatically chosen primitive |
| `[{"f1": v1}, {"f2": v2}]` | `list[dict]` | Multiple `StateCheck`s; edge fires only when all checks pass (AND semantics) |
| `StateCheck(...)` | `StateCheck` | Used directly; gives explicit control over the comparison primitive |
| `lambda acc, p: ...` | `callable` | Evaluated at runtime with the accumulated state and the current turn's parsed result |

### State Field Paths (for dict shorthand and `StateCheck`)

| Path pattern | Resolves to |
|--------------|-------------|
| `"verify_result"` | Boolean result of the current turn's template verification |
| `"turn"` | Current turn number (int) |
| `"parsed.<field>"` | A field from the parsed result of the current turn |
| `"accumulated.<field>"` | A value in the accumulated state |
| `"node_visits.<node_id>"` | Number of times a node has been executed |
| `"node_results.<node_id>.verify_result"` | Verification result of a past node |
| `"node_results.<node_id>.parsed.<field>"` | Parsed field from a past node |
| `"node_results.<node_id>.rubric.<trait>"` | Rubric trait result from a past node |

### Validation Rules

| Rule | Description |
|------|-------------|
| Entry node must exist | `set_entry()` rejects unknown node IDs; `validate()` checks again |
| Edge targets must be valid | Each edge target must be a registered node ID or `END` |
| All nodes must be reachable | BFS from the entry node must visit every registered node |
| Conditional nodes need fallbacks | Any node with at least one conditional edge must also have one unconditional edge |
| Implicit terminals are valid | Nodes with no outbound edges are treated as terminals without requiring an explicit `END` edge |

### `END` Sentinel

`END` is the string `"__end__"`. It is exported from `karenina.scenario` and `karenina.schemas.scenario.types`. Pass it as the `target` of an edge to signal that execution should terminate after that edge is followed.

## 7. Next Steps

- [Outcome Criteria](outcome-criteria.md): declarative assertions evaluated after the scenario completes
- [State and Routing](state-and-routing.md): how runtime state accumulates and how edges are resolved
- [Sycophancy Tutorial](../../../workflows/scenarios/sycophancy-tutorial.md): end-to-end walkthrough building a sycophancy resistance scenario
