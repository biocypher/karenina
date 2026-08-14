# Scenario Edge Cases

Reference for edge cases, validation rules, and runtime behavior in karenina
scenario evaluation.

Source: `karenina/src/karenina/scenario/validation.py`,
`karenina/src/karenina/scenario/edge_resolution.py`,
`karenina/src/karenina/scenario/manager.py`.

## Dead-End Detection

A node with no outgoing edges at all is an **implicit terminal**: the scenario
ends when execution reaches it. This is valid. The scenario status is
`"completed"`.

A node with conditional edges but no unconditional fallback is **invalid**.
Validation catches this and raises `ValueError`:

```
Node 'retry' has conditional edges but no unconditional fallback.
Add an unconditional edge (when=None) as a default path.
```

The fix: add an unconditional edge from that node (usually to `END`).

## Orphan Node Detection

Validation runs BFS from the entry node through all edges (excluding `END`
targets). Any node not reachable from the entry is an **orphan** and causes
validation to fail:

```
Orphan nodes not reachable from entry 'start': ['unused_node']
```

This catches copy-paste errors (adding a node but forgetting to wire it in)
and leftover nodes from graph refactoring.

## Field Reference Resolution

Edge conditions and outcome criteria resolve field paths against `ScenarioState`
using dot-path syntax. Resolution rules:

| Path | Resolves to | Missing behavior |
|------|-------------|-----------------|
| `verify_result` | `state.verify_result` | `None` |
| `turn` | `state.turn` (int, 0-indexed) | Always present |
| `current_node` | `state.current_node` (str) | Always present |
| `parsed.<field>` | `state.parsed[field]` | `None` |
| `accumulated.<key>` | `state.accumulated[key]` | `None` |
| `node_visits.<node>` | `state.node_visits[node]` | **`0`** (not `None`) |
| `node_results.<node>` | Full result dict for node | `{}` |
| `node_results.<node>.verify_result` | Past node's pass/fail | `None` |
| `node_results.<node>.parsed.<field>` | Past node's parsed field | `None` |
| `node_results.<node>.rubric.<trait>` | Past node's rubric score | `None` |

Key implication: referencing a field that does not exist returns `None`, and
`None` compared to any expected value returns `False`. This means conditions
on not-yet-visited nodes silently fail rather than raising errors. Design
graphs so that conditions only reference nodes guaranteed to have been visited
by the time the edge is evaluated.

`node_visits` is the exception: it returns `0` for unvisited nodes, so
`{"node_visits.retry": {"gte": 3}}` correctly evaluates to `False` when
`retry` has not been visited.

## TurnCheck Bounds

`turn_at(index)` (a scope selector used inside `TurnCheck(scope=...)`) uses
0-based indexing into the execution history. Negative indexing is supported
(`turn_at(-1)` = last turn).

If the index is out of bounds (e.g., `turn_at(5)` on a 3-turn execution),
the check **returns `False`** rather than raising an error. This is intentional:
outcome criteria should never crash; they should report failure.

The same applies to `first_turn` and `last_turn` on an empty history (0 turns):
they return `False`.

## scenario_turn_limit (Default: 20)

`VerificationConfig.scenario_turn_limit` caps the maximum number of turns a
scenario can execute. If this limit is reached:

- The scenario stops immediately
- `ScenarioExecutionResult.status` is set to `"limit_reached"`
- All turns executed so far are preserved in `history` and `turn_results`
- Outcome criteria still evaluate against the partial execution

The default of 20 is generous for most scenarios. Set it lower for retry loops
to prevent infinite cycling:

```python
config = VerificationConfig(
    # ...models...
    scenario_turn_limit=10,  # Prevent runaway retry loops
)
```

## Guard Per-Turn Behavior

Pipeline guards (abstention check, sufficiency check, recursion limit auto-fail,
trace validation auto-fail) apply independently to each turn. Their behavior
during scenarios:

**Abstention check**: If enabled, runs on each turn. If abstention is detected,
the turn's `verify_result` is set to `False` and
`template.abstention_detected = True`. The scenario continues; edge conditions
can route based on the result.

**Sufficiency check**: Same as abstention. Per-turn evaluation, does not
terminate the scenario.

**Recursion limit auto-fail**: If an MCP agent hits its recursion limit during
a turn, that turn's `verify_result` is set to `False` and
`template.recursion_limit_reached = True`. The scenario continues.

**Trace validation auto-fail**: If the agent trace does not end with an AI
message, the turn auto-fails. The scenario continues.

**Pipeline error**: If a turn encounters an unrecoverable pipeline error
(exception during stage execution), `ScenarioExecutionResult.status` is set
to `"error"` and the scenario stops immediately. The partial execution is
preserved.

## Edge Evaluation Order

Edges from a node are evaluated in definition order. The first conditional
edge whose condition is satisfied wins. Unconditional edges are skipped
during the conditional pass and used as fallback only if no conditional edge
matches.

This means edge order matters for conditional edges:

```python
# Edge order matters: first match wins
s.add_edge("node", "path_a", when={"parsed.score": 5})  # Checked first
s.add_edge("node", "path_b", when={"parsed.score": 4})  # Checked second
s.add_edge("node", END)  # Fallback
```

If the score is 5, `path_a` is taken. If the score is 4, `path_b` is taken.
For any other score, `END` is taken.

## State Update Callables

Nodes can carry a `state_update` callable that modifies custom accumulated
state after each turn:

```python
s.add_node(
    "counter",
    question=q,
    state_update=lambda acc, parsed: {**acc, "count": acc.get("count", 0) + 1},
)
```

If a `state_update` callable raises an exception, the engine restores the
accumulated state from a `deepcopy` snapshot taken before the call. The
exception is logged at warning level and the scenario continues. This
guarantees that a callback that raises partway through mutation cannot leave
the state partially modified.

The callable receives `(accumulated_dict, parsed_fields_dict)` and must return
the new accumulated dict.

## Redundant Unconditional Edges

If a node has multiple unconditional edges (no condition), `validate()` emits
a `UserWarning`. Only the first unconditional edge is used as the fallback;
the rest are silently ignored. This is almost always a graph construction
error.

```python
# Warning: Node 'check' has 2 unconditional edges (targets: ['path_a', 'path_b']).
# Only the first will be used.
s.add_edge("check", "path_a")  # This one wins
s.add_edge("check", "path_b")  # Silently ignored
```

## ModelOverride Per-Node

Individual nodes can override the answering and/or parsing models:

```python
from karenina.schemas.scenario.types import ModelOverride

s.add_node(
    "hard_question",
    question=q,
    model_override=ModelOverride(
        answering_model=ModelConfig(
            id="strong",
            model_provider="anthropic",
            model_name="claude-sonnet-4-20250514",
        ),
    ),
)
```

Only the specified model is overridden; the other uses the base config.
