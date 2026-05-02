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

# Building Scenarios

The `Scenario` class is a mutable builder for constructing multi-turn benchmark graphs. You add nodes, connect them with edges, optionally attach outcome criteria, and call `validate()` to freeze the graph into an immutable `ScenarioDefinition`. Unlike single-turn benchmarks where each question runs independently, a scenario defines a conversation flow where each turn's result can determine which question comes next.

This page covers the builder API: how to construct valid scenario graphs, what validation enforces, and which graph shapes work for common evaluation patterns. For how edges are evaluated at runtime, see [State and Routing](state-and-routing.md). For outcome criteria, see [Outcome Criteria](outcome-criteria.md).

```python tags=["hide-cell"]
# Mock setup for documentation: allows notebook to run without API keys.
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
        self._outcome_criteria: list[Any] = []

    def add_node(
        self,
        node_id: str,
        *,
        question: Question,
        model_override: Any = None,
        tool_filter: Any = None,
        state_update: Any = None,
        agent_identity: str | None = None,
        **metadata: Any,
    ) -> None:
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists")
        if agent_identity == "__user__":
            raise ValueError("agent_identity='__user__' is reserved")
        self._nodes[node_id] = {
            "node_id": node_id,
            "question": deepcopy(question),
            "agent_identity": agent_identity,
        }

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        when: Any = None,
        handover: Any = None,
    ) -> None:
        if isinstance(handover, str):
            _KNOWN_STRATEGIES = {
                "transcript_prepend",
                "transcript_append",
                "transcript_materialize",
            }
            if handover not in _KNOWN_STRATEGIES:
                raise ValueError(f"Unknown handover strategy '{handover}'")
        edge = _Edge(source=source, target=target, condition=when)
        edge.handover = handover
        self._edges.append(edge)

    def add_outcome(self, name: str, check: Any, *, description: str = "") -> None:
        self._outcome_criteria = getattr(self, "_outcome_criteria", [])
        self._outcome_criteria.append(
            {"name": name, "description": description, "check": check}
        )

    def add_outcome_criterion(self, criterion: Any) -> None:
        self._outcome_criteria = getattr(self, "_outcome_criteria", [])
        self._outcome_criteria.append(criterion)

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
            outcome_criteria=list(self._outcome_criteria),
        )


print("Mock setup complete.")
```

## 1. What It Is

`Scenario` is a mutable builder object. You call methods on it to register nodes, connect them with edges, and set the starting point. Calling `validate()` runs structural checks and returns a frozen `ScenarioDefinition` that the scenario runner uses.

The builder separates construction from validation. You can add nodes and edges in any order; errors surface only when you call `validate()`. The resulting `ScenarioDefinition` is immutable: the runner reads it but cannot modify it.

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
print(f"Nodes: {list(defn.nodes.keys())}")
print(f"Entry: {defn.entry_node}")
```

Builder method summary:

| Method | Purpose |
|--------|---------|
| `add_node(node_id, *, question, ...)` | Register a node with a question (optionally with `agent_identity`, `state_update`, `model_override`, `tool_filter`) |
| `add_edge(source, target, *, when=None, handover=None)` | Connect two nodes, optionally with a condition and a handover strategy |
| `add_outcome(name, check, *, description="")` | Sugar for attaching a declarative outcome criterion |
| `add_outcome_criterion(criterion)` | Attach a pre-built `ScenarioOutcomeCriterion` (escape hatch for callable outcomes) |
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
| `callable` | `when=lambda state: ...` | Evaluated against runtime `ScenarioState` |

### 4.3 Validation

`validate()` enforces three hard-fail structural checks before freezing the graph, plus one explicit allowance and one warning:

1. Edge sources and targets must reference registered nodes (or `END`).
2. All nodes must be reachable from the entry node via BFS.
3. Every node with conditional edges must also have at least one unconditional fallback edge.
4. *(Allowance, not a failure)* Nodes with no outbound edges are valid implicit terminals; they terminate the scenario without an explicit `END` edge.
5. *(Warning, not a failure)* If a node has multiple unconditional edges, a `UserWarning` is emitted. Only the first unconditional edge will be used; the rest are silently ignored. This almost always indicates a graph construction error.

If any of the three hard-fail checks (1, 2, or 3) fails, `validate()` raises `ValueError` with a description of the problem.

### 4.4 Agent Identity (Multi-Agent Scenarios)

Each node can declare an `agent_identity` label. The label tags every message the node's agent produces in the scenario's tagged-message log (`TaggedMessage.agent_id`), and it controls when the runtime fires a handover. When an edge crosses a node-to-node boundary whose `agent_identity` changes, the matched edge's `handover` strategy decides how the next agent sees the prior conversation. See [Handover](handover.md) for the four strategies and the `TaggedMessage` shape, and [State and Routing](state-and-routing.md) for how the runtime emits tagged messages each turn.

```python
s_multi = Scenario("multi-agent")
s_multi.add_node("triage_question", question=q, agent_identity="triage")
s_multi.add_node("specialist_question", question=q, agent_identity="specialist")
```

The string `"__user__"` is reserved for scenario-internal user prompts and cannot be used here. Nodes that do not set `agent_identity` execute without an identity tag, and edges between such nodes require no handover.

### 4.5 Tool Filter (Restricting MCP Tools per Node)

`tool_filter` is an optional per-node setting that removes specific MCP tools from the answering model's tool set when this node runs. It is built as a `ToolFilter` containing one or more `ToolFilterEntry` records: each entry names a server, and optionally a tool on that server. Omitting the `tool` field removes every tool from the named server. This is most useful when a scenario wants the same model to behave differently at successive turns: e.g., grant search tools at turn 0, then strip them at turn 1 to test recall under tool removal.

```python
# ToolFilter and ToolFilterEntry are real Pydantic schemas; the mock cell
# above does not import them, so we sketch the call site only.

# from karenina.schemas.scenario.types import ToolFilter, ToolFilterEntry
#
# tool_filter = ToolFilter(remove=[
#     ToolFilterEntry(server="pubmed", tool="search_articles"),
#     ToolFilterEntry(server="filesystem"),  # remove every tool from this server
# ])
#
# s_filter = Scenario("tool-controlled")
# s_filter.add_node("with_tools", question=q)  # full MCP tool set available
# s_filter.add_node("without_tools", question=q, tool_filter=tool_filter)
print("ToolFilter / ToolFilterEntry: karenina.schemas.scenario.types")
```

`ToolFilter` is fully serialized in the SchemaOrg checkpoint and round-trips through `scenario_to_schema_org` / `schema_org_to_scenario`. For how MCP tools are wired into the answering model in the first place, see the [MCP Integration](../../notebooks/advanced/mcp-integration.ipynb) page.

### 4.6 Handover Strategies

When an edge crosses an `agent_identity` boundary (or whenever you want explicit control over how the next agent sees prior turns), `add_edge` accepts a `handover` parameter. Three string strategies are recognised:

| Strategy | What it does |
|----------|--------------|
| `"transcript_prepend"` | Prepends the prior agent's transcript before the next agent's first message |
| `"transcript_append"` | Appends the prior agent's transcript after the next agent's prompt |
| `"transcript_materialize"` | Materializes the prior agent's tagged trace as a flattened transcript fed to the next agent |

A callable `(messages, state) -> messages` may also be passed. Callable handovers are not serialized to checkpoints and emit a `UserWarning`; prefer string strategies when you intend to persist the scenario.

```python
s_handoff = Scenario("with-handover")
s_handoff.add_node("first", question=q, agent_identity="agent_a")
s_handoff.add_node("second", question=q, agent_identity="agent_b")
s_handoff.add_edge("first", "second", handover="transcript_append")
```

The full semantics of each strategy (what the rendered transcript looks like, how `transcript_materialize` writes traces under `<turn_dir>/traces/`, and how callable handovers receive `tagged_messages` and `state`) live in [Handover](handover.md).

### 4.7 Attaching Outcome Criteria

Outcome criteria run after every turn completes and assert over the full execution. Two methods attach them to a scenario:

- `add_outcome(name, check, *, description="")`: sugar that wraps a declarative `OutcomeNode` (e.g., `TurnCheck`, `ResultCheck`, `AllOf`, `CountTurns`) into a `ScenarioOutcomeCriterion`. This is the primary path; fully serializable.
- `add_outcome_criterion(criterion)`: attach a pre-built `ScenarioOutcomeCriterion` directly. Use this when you need the callable `evaluate=` escape hatch.

Outcome criteria are explained in depth on the [Outcome Criteria](outcome-criteria.md) page; this section only documents the attachment surface on the builder.

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
print(f"Linear scenario: {list(linear_defn.nodes.keys())}")
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
print(f"Looping scenario: {list(loop_defn.nodes.keys())}")
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
| `agent_identity` | `str \| None` | No | Identity label for this node's agent. The reserved value `"__user__"` is rejected. Used to drive transcript handover between agents. |
| `**metadata` | `Any` | No | Arbitrary key-value pairs stored on the node for reference |

### `add_edge()` Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source` | `str` | Yes | Source node id (must be a registered node) |
| `target` | `str` | Yes | Target node id, or `END` |
| `when` | `dict \| list[dict] \| StateCheck \| callable \| str \| None` | No | Edge condition (see Edge Condition Formats) |
| `handover` | `str \| callable \| None` | No | Transcript handover strategy. String values: `"transcript_prepend"`, `"transcript_append"`, `"transcript_materialize"`. Callable form `(messages, state) -> messages` is allowed but is not serialized to checkpoints. |

### Outcome Attachment

| Method | Signature | Use when |
|--------|-----------|----------|
| `add_outcome` | `add_outcome(name, check, *, description="")` | You have a declarative `OutcomeNode` (TurnCheck, ResultCheck, AllOf, etc.). Fully serializable. |
| `add_outcome_criterion` | `add_outcome_criterion(criterion: ScenarioOutcomeCriterion)` | You need the `evaluate=` callable escape hatch on `ScenarioOutcomeCriterion`. |

### Edge Condition Formats

| `when` form | Type | Semantics |
|-------------|------|-----------|
| `None` | `None` | Unconditional edge; serves as the fallback when all conditional edges on this source are not met |
| `{"field": value}` | `dict` | Single-key shorthand; converted to a `StateCheck` with an automatically chosen primitive |
| `[{"f1": v1}, {"f2": v2}]` | `list[dict]` | Multiple `StateCheck`s; edge fires only when all checks pass (AND semantics) |
| `StateCheck(...)` | `StateCheck` | Used directly; gives explicit control over the comparison primitive |
| `lambda state: ...` | `callable` | Evaluated at runtime with the full `ScenarioState`; returns `bool` |

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
- [Sycophancy Tutorial](../../../notebooks/scenarios/sycophancy-tutorial.ipynb): end-to-end walkthrough building a sycophancy resistance scenario
