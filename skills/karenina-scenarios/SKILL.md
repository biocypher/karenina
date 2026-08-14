---
name: karenina-scenarios
description: >
  Build and run multi-turn scenario evaluations with karenina. Use when
  testing conversational AI through branching dialogue paths, sycophancy
  checks, or multi-turn reasoning. Covers scenario graphs, nodes, edges,
  outcome criteria, and TurnCheck. Invoke for any multi-turn evaluation work.
---

# Scenario Evaluation

Scenarios evaluate LLM behavior across a graph of connected conversation turns.
Unlike single-turn benchmarks (independent questions) or TaskEval (pre-recorded
outputs), scenarios run a live conversation where each turn's result determines
which question comes next. The evaluation covers the full execution: individual
responses, the path taken, and state accumulated across turns.

**Source of truth**: `karenina/src/karenina/scenario/` (builder, validation,
edge resolution, outcome evaluation, sugar functions).

**Reference files in this skill**:
- [references/scenario-edge-cases.md](references/scenario-edge-cases.md): Dead-end detection, field reference resolution, TurnCheck bounds, orphan nodes, turn limits, guard behavior

## Interactive Procedure

Follow these eight steps when building a scenario evaluation. Do not skip steps.

### Step 1: Define the Graph

Ask the user:

> Describe the conversation flow you want to evaluate. What questions does the
> model answer, and how should the next question depend on the previous answer?

Sketch a graph from the response. Identify nodes (questions), edges (transitions),
and conditions (what determines routing). Common patterns:

| Pattern | Shape | Example |
|---------|-------|---------|
| Linear | A -> B -> END | Two-step knowledge probe |
| Branching | A -> B (if correct) or A -> C (if wrong) -> END | Sycophancy detection |
| Retry loop | A -> A (if wrong, up to N) -> END (if right) | Self-correction |
| Diamond | A -> B or C -> D -> END | Multiple evaluation paths converging |

### Step 2: Define Questions and Templates per Node

For each node in the graph, create a Question with an answer template. Delegate
template creation to `karenina-template-authoring` with context:

> Scenario node "[node_name]", template evaluates [description of what this
> turn should extract and verify].

Each node requires a `Question` object. The question carries the text, ground
truth (`raw_answer`), keywords, and an `answer_template` (the BaseAnswer
subclass code as a string).

```python
from karenina.schemas.entities.question import Question

q_aspirin = Question(
    question="What is the mechanism of action of aspirin?",
    raw_answer="Irreversible inhibition of cyclooxygenase (COX) enzymes",
    answer_template="""
class AspirinMOA(BaseAnswer):
    mechanism: str = VerifiedField(
        description="The mechanism of action",
        ground_truth="Irreversible inhibition of cyclooxygenase (COX) enzymes",
        verify_with=ContainsAll(substrings=["cyclooxygenase"], normalize=["lowercase"]),
    )
""",
)
```

Primitive names (`ExactMatch`, `ContainsAll`, `BooleanMatch`, ...) are
available inside the template string without imports. Construct them with
keyword arguments — pydantic v2 models reject positional args
(`ContainsAll(substrings=["cox"], normalize=["lowercase"])`, not
`ContainsAll(["cox"])`). `ExactMatch()` is already case-insensitive (its
default normalization is lowercase + strip).

**`SemanticMatch` is not a standalone primitive.** Its `check()` raises
`NotImplementedError` outside the `embedding_check` pipeline stage (it needs an
embedding model configured at runtime). Do not use it as an ordinary
`verify_with` primitive in a template; pick a text primitive such as
`ContainsAll`, `ContainsAny`, or `ExactMatch` instead.

### Step 3: Define Edges with Conditions

Connect nodes with edges. Edges can be conditional (fire when a predicate on
the current state is satisfied) or unconditional (fallback when no conditional
edge fires).

```python
from karenina.scenario import Scenario, END

s = Scenario("aspirin-knowledge")
s.add_node("ask_moa", question=q_aspirin)
s.add_node("followup", question=q_followup)

# Conditional: proceed to followup if first answer was correct
s.add_edge("ask_moa", "followup", when={"verify_result": True})

# Unconditional fallback: end if wrong
s.add_edge("ask_moa", END)

# Followup always ends
s.add_edge("followup", END)
```

Edge conditions use dict shorthand or `StateCheck` objects. Dict keys are
dot-paths resolved against `ScenarioState`:

| Path | Resolves to |
|------|-------------|
| `verify_result` | Current turn's template pass/fail |
| `parsed.<field>` | Parsed template field from current turn |
| `accumulated.<key>` | Custom state from `state_update` callables |
| `node_visits.<node>` | Visit count for a node (0 if unvisited) |
| `node_results.<node>.verify_result` | Past node's pass/fail |
| `node_results.<node>.parsed.<field>` | Past node's parsed field |
| `node_results.<node>.rubric.<trait>` | Past node's rubric trait score |
| `turn` | Current turn number (0-indexed) |

Multiple conditions in a list are AND'd:

```python
s.add_edge("retry", "ask_again", when=[
    {"verify_result": False},
    {"node_visits.retry": {"lt": 3}},
])
```

### Step 4: Set Entry and Add END Edges

Every scenario must have an explicit entry node. Every node that can terminate
the conversation needs an edge to `END`.

```python
s.set_entry("ask_moa")
```

Validation will fail if:
- No entry node is set
- Any node is unreachable from the entry (orphan nodes)
- A node has conditional edges but no unconditional fallback

### Step 5: Define Outcome Criteria

Outcome criteria are assertions evaluated after the scenario finishes. They
check properties of the entire execution, not just individual turns.

Use the sugar functions from `karenina.scenario.sugar`:

```python
from karenina.scenario.sugar import all_of, last_turn, turn_at, any_turn
from karenina.schemas.scenario.checks import TurnCheck
from karenina.schemas.primitives.scope import TurnAt
from karenina.schemas.primitives import BooleanMatch

# Check that the last turn passed verification
s.add_outcome("final_correct", last_turn(verify_result=True))

# Check a specific turn by index: turn_at() returns a scope SELECTOR,
# so build the TurnCheck manually and pass the selector as its scope.
s.add_outcome(
    "first_on_topic",
    TurnCheck(
        scope=TurnAt(index=0),
        field="verify_result",
        expected=True,
        verify_with=BooleanMatch(),
    ),
    description="First turn stayed on topic",
)

# Compose multiple checks
s.add_outcome(
    "knowledge_retained",
    all_of(
        last_turn(verify_result=True),
        any_turn(node="ask_moa", verify_result=True),
    ),
)
```

`turn_at(index)`, `first_turn_scope()`, and `last_turn_scope()` are **scope
selectors**, not checks. They are usable only inside `TurnCheck(scope=...)` or
as the turn scopes of `cross_turn(...)`. Passing a bare selector to
`add_outcome()` fails validation: `ScenarioOutcomeCriterion.check` is a tagged
union (`turn_check`, `result_check`, `cross_turn_check`, `all_of`, `any_of`,
`at_least_n`, `count_turns`, `first_match_index`) and a selector matches none
of those tags.

Available sugar functions:

| Function | Scope | Returns |
|----------|-------|---------|
| `last_turn(**fields)` | Last turn in history | TurnCheck or AllOf |
| `first_turn(**fields)` | First turn | TurnCheck or AllOf |
| `turn_at(index)` | Specific turn (supports negative indexing) | TurnAt scope selector |
| `any_turn(node=None, **fields)` | Any matching turn | TurnCheck or AllOf |
| `all_turns(node=None, **fields)` | All matching turns | TurnCheck or AllOf |
| `status_is(expected)` | Execution status | ResultCheck |
| `turn_count_gte(n)` | Minimum turn count | ResultCheck |
| `turn_count_eq(n)` | Exact turn count | ResultCheck |
| `count_turns(node=None, verify_result=None)` | Count matching turns | CountTurns |
| `first_match_index(node=None, verify_result=None)` | Index of first match | FirstMatchIndex |
| `cross_turn(source_turn=..., source_field=..., target_turn=..., target_field=..., comparison=...)` | Compare two turns (keyword-only; scopes take selectors like `turn_at(0)`) | CrossTurnCheck |
| `all_of(*checks)` | All must pass | AllOf |
| `any_of(*checks)` | At least one must pass | AnyOf |
| `at_least_n(n, *checks)` | N or more must pass | AtLeastN |

Scope helpers (selectors only — see the note above):
- `first_turn_scope()` / `last_turn_scope()` / `turn_at(index)`

### Step 6: Add to Benchmark

Validate the scenario to freeze it into a `ScenarioDefinition`, then add it
to a benchmark:

```python
defn = s.validate()  # Raises ValueError on structural errors

from karenina.benchmark import Benchmark

benchmark = Benchmark.create(name="Aspirin Knowledge", version="1.0.0")
benchmark.add_scenario(defn)
```

### Step 7: Configure and Run

Delegate to `karenina-verification` for full configuration details. Scenarios
require BOTH answering and parsing models.

```python
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification import VerificationConfig

config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="answerer",
            model_provider="anthropic",
            model_name="claude-sonnet-4-6",
        )
    ],
    parsing_models=[
        ModelConfig(
            id="judge",
            model_provider="anthropic",
            model_name="claude-haiku-4-5",
        )
    ],
    scenario_turn_limit=20,  # Default; max turns before forced termination
)

results = benchmark.run_verification(config)
```

**Live runs need provider keys in the environment.** The repo-root `.env` is
NOT auto-loaded; export keys before launching, e.g.
`set -a; source .env; set +a`. Without a key the run fails when the first
request is built with
`TypeError: "Could not resolve authentication method. Expected either api_key
or auth_token to be set..."`, which surfaces as a scenario terminal failure
(see Step 8).

**Progressive save / resume works for scenarios.** Pass a `ProgressiveFileSink`
to `run_verification(sink=...)` or use the CLI `--progressive-save` /
`--resume` flags. Scenarios are **combo-atomic**: the unit of work is
`(scenario_id, answering_key, parsing_key, replicate)`, persisted only when
all turns of the combo complete. Interrupted combos re-run from turn 1 on
resume (no turn-level checkpointing). `Benchmark.resume_verification()`
auto-detects QA vs scenario. See `karenina-verification` for details.

### Step 8: Analyze Results

Delegate to `karenina-results` for detailed analysis. `run_verification()`
returns a `VerificationResultSet`. For scenarios, access per-scenario execution
results via `scenario_results` and any collected errors via `errors`:

```python
result_set = benchmark.run_verification(config)

# Access scenario execution results
if result_set.scenario_results:
    for scenario_result in result_set.scenario_results:
        print(f"Scenario: {scenario_result.scenario_id}")
        print(f"Status: {scenario_result.status}")  # completed, limit_reached, error, timeout
        print(f"Path taken: {scenario_result.path}")
        print(f"Turn count: {scenario_result.turn_count}")

        # outcome_results is dict[str, bool | int | float]: criterion name -> verdict.
        # Access verdicts by key; iterating yields criterion names, not objects.
        for name in scenario_result.outcome_results:
            verdict = scenario_result.outcome_results[name]
            print(f"  Outcome {name}: {verdict}")

        # Per-turn details
        for turn in scenario_result.history:
            print(f"  Turn {turn.node_id}: verify={turn.verify_result}")

# Check collected errors: (description, exception) tuples
if result_set.errors:
    for desc, exc in result_set.errors:
        print(f"Failed: {desc}: {exc}")

# Per-turn VerificationResults are also available flat
for vr in result_set.results:
    print(vr.metadata.question_id, vr.metadata.scenario_node)
```

`errors` entries are `(description, exception)` tuples covering two kinds of
failure. Raised combo exceptions (adapter construction failures, unexpected
crashes) appear as-is. Scenarios that stopped with `status='error'` ALSO
appear here: terminal pipeline failures (e.g. an adapter auth error
exhausted after retries) are wrapped in `ScenarioExecutionFailure`, whose
message names the scenario, failing node, failure category/stage, and reason —
e.g. `Scenario 'auth_fail_scenario' stopped at node 'n1' (connection in
stage 'generate_answer'): AuthenticationError: 401 invalid api key`. The
scenario still appears in `scenario_results` with `status='error'` and a
populated `terminal_failure` (`node_id`, `category`, `stage`, `reason`).
Error entries do not carry tracebacks; full tracebacks appear in the logs.

For tabular analysis, `result_set.get_scenario_results()` returns a
`ScenarioResults` container with dataframe methods: `to_dataframe()` (one row
per execution), `to_turn_dataframe()` (one row per turn), and
`to_outcome_dataframe()` (one row per outcome criterion).

---

## Gotchas

### Every node needs an outgoing edge (even to END)

Nodes with no outgoing edges are implicit terminals (the scenario ends there).
However, nodes with conditional edges MUST also have an unconditional fallback
edge. Validation enforces this.

```python
# Wrong: conditional edges with no fallback
s.add_edge("retry", "ask_again", when={"verify_result": False})
# Validation error: "Node 'retry' has conditional edges but no unconditional fallback"

# Correct: add unconditional fallback
s.add_edge("retry", "ask_again", when={"verify_result": False})
s.add_edge("retry", END)  # Fallback: end if condition doesn't match
```

### Entry must be set explicitly

`Scenario.validate()` raises `ValueError` if `set_entry()` was never called.
There is no default entry node.

### Edge conditions use dot-path on previous results

Conditions reference state fields using dot-path syntax. A nonexistent field
resolves to `None` (except `node_visits`, which returns 0 for unvisited nodes).
`None` compared with any expected value via `BooleanMatch` or `ExactMatch`
will return `False`.

### Outcome TurnCheck uses index-based scope

Inside `TurnCheck(scope=turn_at(0), ...)`, `turn_at(0)` selects the first turn
in history. Negative indexing is supported: `turn_at(-1)` selects the last
turn. If the index is out of bounds, the check fails (returns `False`).
Remember that `turn_at(...)` is a selector, not a check — it cannot be passed
to `add_outcome()` on its own.

### BOTH answering and parsing models required

Scenarios always need both models. The answering model generates responses
at each turn; the parsing model extracts structured data from responses.
Unlike TaskEval, there is no way to skip the generation step.

### `evaluation_mode='rubric_only'` is ignored in scenarios

`ScenarioManager.run` auto-detects whether to evaluate rubrics from rubric
presence on each turn. Passing `evaluation_mode="rubric_only"` does NOT raise;
it emits a `UserWarning` and the setting is ignored. There is no `"auto"` mode.
Attach rubrics and use `evaluation_mode="template_and_rubric"` when you need
rubric evaluation alongside template verification.

### Guards apply per-turn

Pipeline guards (abstention check, sufficiency check, recursion limit) apply
to each individual turn. A turn that triggers an abstention auto-fail still
produces a `VerificationResult` with `abstention_detected=True` in the
history. Edge conditions can route based on this.

### Conversation history accumulates

Each turn receives the full conversation history from previous turns. The
answering model sees all prior question-answer pairs, enabling multi-turn
reasoning. This is the core difference from single-turn benchmarks.

---

## Key Imports

```python
# Builder
from karenina.scenario import Scenario, END

# Sugar functions for outcome criteria
from karenina.scenario.sugar import (
    all_of, any_of, at_least_n,
    last_turn, first_turn, turn_at, any_turn, all_turns,
    status_is, turn_count_gte, turn_count_eq,
    count_turns, first_match_index, cross_turn,
    first_turn_scope, last_turn_scope,
)

# Check types (for type hints or advanced usage)
from karenina.schemas.scenario.checks import (
    TurnCheck, ResultCheck, CrossTurnCheck,
    CountTurns, FirstMatchIndex,
)

# Types
from karenina.schemas.scenario.types import (
    ModelOverride, ScenarioOutcomeCriterion, StateCheck,
)

# State and results (for analysis)
from karenina.schemas.scenario.state import (
    ScenarioExecutionResult, ScenarioState, TurnRecord,
)
```


## Assets

- [assets/scenario-skeleton.py](assets/scenario-skeleton.py): Two-node aspirin knowledge scenario with branching

**Full karenina docs**: The `using-karenina` skill contains the complete karenina documentation in its `references/` directory. Consult it for API details not covered here.
