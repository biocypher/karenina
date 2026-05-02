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

# Outcome Criteria

Outcome criteria are declarative assertions evaluated after a scenario completes. Each criterion inspects the full `ScenarioExecutionResult` and returns a boolean or numeric result. They are distinct from answer templates: answer templates verify individual turn results (did the model get this specific question right?), while outcome criteria verify the scenario as a whole (did the model behave correctly across the full conversation?).

This page explains how outcome criteria are constructed, how the runner evaluates them, and which patterns cover common evaluation needs. For building the graph that produces the execution result, see [Building Scenarios](building-scenarios.md). For how edges route between turns, see [State and Routing](state-and-routing.md).

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

class NumericRange:
    type: str = "numeric_range"
    def __init__(self, min: int | float | None = None, max: int | float | None = None,
                 exclusive_min: bool = False, exclusive_max: bool = False) -> None:
        self.min = min
        self.max = max
        self.exclusive_min = exclusive_min
        self.exclusive_max = exclusive_max
    def check(self, value: Any, expected: Any) -> bool:
        if self.min is not None:
            if self.exclusive_min and value <= self.min:
                return False
            if not self.exclusive_min and value < self.min:
                return False
        if self.max is not None:
            if self.exclusive_max and value >= self.max:
                return False
            if not self.exclusive_max and value > self.max:
                return False
        return True


# ---------------------------------------------------------------------------
# Scope types
# ---------------------------------------------------------------------------

class LastTurn:
    type: str = "last_turn"

class FirstTurn:
    type: str = "first_turn"

class TurnAt:
    type: str = "turn_at"
    def __init__(self, index: int) -> None:
        self.index = index

class AnyTurn:
    type: str = "any_turn"
    def __init__(self, node_id: str | None = None) -> None:
        self.node_id = node_id

class AllTurns:
    type: str = "all_turns"
    def __init__(self, node_id: str | None = None) -> None:
        self.node_id = node_id

ScopeUnion = LastTurn | FirstTurn | TurnAt | AnyTurn | AllTurns


# ---------------------------------------------------------------------------
# Check nodes
# ---------------------------------------------------------------------------

class TurnCheck:
    type: str = "turn_check"
    def __init__(self, scope: Any, field: str, verify_with: Any, expected: Any = None) -> None:
        self.scope = scope
        self.field = field
        self.expected = expected
        self.verify_with = verify_with

class ResultCheck:
    type: str = "result_check"
    def __init__(self, field: str, verify_with: Any, expected: Any = None) -> None:
        self.field = field
        self.expected = expected
        self.verify_with = verify_with

class CrossTurnCheck:
    type: str = "cross_turn_check"
    def __init__(self, source_turn: Any, source_field: str,
                 target_turn: Any, target_field: str,
                 comparison: str, normalize: list | None = None) -> None:
        self.source_turn = source_turn
        self.source_field = source_field
        self.target_turn = target_turn
        self.target_field = target_field
        self.comparison = comparison
        self.normalize = normalize or []

class CountTurns:
    type: str = "count_turns"
    def __init__(self, node_id: str | list[str] | None = None, verify_result: bool | None = None) -> None:
        self.node_id = node_id
        self.verify_result = verify_result

class FirstMatchIndex:
    type: str = "first_match_index"
    def __init__(self, node_id: str | list[str] | None = None, verify_result: bool | None = None) -> None:
        self.node_id = node_id
        self.verify_result = verify_result


# ---------------------------------------------------------------------------
# Composition nodes
# ---------------------------------------------------------------------------

class AllOf:
    type: str = "all_of"
    def __init__(self, conditions: list) -> None:
        self.conditions = conditions

class AnyOf:
    type: str = "any_of"
    def __init__(self, conditions: list) -> None:
        self.conditions = conditions

class AtLeastN:
    type: str = "at_least_n"
    def __init__(self, n: int, conditions: list) -> None:
        self.n = n
        self.conditions = conditions


# ---------------------------------------------------------------------------
# Sugar functions
# ---------------------------------------------------------------------------

def _infer_primitive(value: Any) -> Any:
    if isinstance(value, bool):
        return BooleanMatch()
    if isinstance(value, str):
        return ExactMatch()
    if isinstance(value, (int, float)):
        return NumericExact()
    return ExactMatch()

def _make_turn_checks(scope: Any, **fields: Any) -> TurnCheck | AllOf:
    checks = []
    for f, v in fields.items():
        checks.append(TurnCheck(scope=scope, field=f, expected=v, verify_with=_infer_primitive(v)))
    if len(checks) == 1:
        return checks[0]
    return AllOf(conditions=checks)

def last_turn(**fields: Any) -> TurnCheck | AllOf:
    return _make_turn_checks(LastTurn(), **fields)

def first_turn(**fields: Any) -> TurnCheck | AllOf:
    return _make_turn_checks(FirstTurn(), **fields)

def any_turn(*, node: str | None = None, **fields: Any) -> TurnCheck | AllOf:
    return _make_turn_checks(AnyTurn(node_id=node), **fields)

def all_turns(*, node: str | None = None, **fields: Any) -> TurnCheck | AllOf:
    return _make_turn_checks(AllTurns(node_id=node), **fields)

def status_is(expected: str) -> ResultCheck:
    return ResultCheck(field="status", expected=expected, verify_with=ExactMatch())

def turn_count_gte(n: int) -> ResultCheck:
    return ResultCheck(field="turn_count", verify_with=NumericRange(min=n))

def turn_count_eq(n: int) -> ResultCheck:
    return ResultCheck(field="turn_count", expected=n, verify_with=NumericExact())

def count_turns(*, node: str | None = None, verify_result: bool | None = None) -> CountTurns:
    return CountTurns(node_id=node, verify_result=verify_result)

def first_match_index(*, node: str | None = None, verify_result: bool | None = None) -> FirstMatchIndex:
    return FirstMatchIndex(node_id=node, verify_result=verify_result)

def cross_turn(*, source: Any, source_field: str, target: Any,
               target_field: str, comparison: str, normalize: list | None = None) -> CrossTurnCheck:
    return CrossTurnCheck(
        source_turn=source, source_field=source_field,
        target_turn=target, target_field=target_field,
        comparison=comparison, normalize=normalize or [],
    )

def first_turn_scope() -> FirstTurn:
    return FirstTurn()

def last_turn_scope() -> LastTurn:
    return LastTurn()

def turn_at(index: int) -> TurnAt:
    return TurnAt(index=index)

def all_of(*checks: Any) -> AllOf:
    return AllOf(conditions=list(checks))

def any_of(*checks: Any) -> AnyOf:
    return AnyOf(conditions=list(checks))

def at_least_n(n: int, *checks: Any) -> AtLeastN:
    return AtLeastN(n=n, conditions=list(checks))


# ---------------------------------------------------------------------------
# ScenarioOutcomeCriterion mock
# ---------------------------------------------------------------------------

class ScenarioOutcomeCriterion:
    def __init__(self, name: str, description: str = "",
                 check: Any = None, evaluate: Any = None,
                 evaluate_source: str | None = None) -> None:
        self.name = name
        self.description = description
        self.check = check
        self.evaluate = evaluate
        self.evaluate_source = evaluate_source


# ---------------------------------------------------------------------------
# Scenario builder mock (outcome methods only)
# ---------------------------------------------------------------------------

class Scenario:
    """Lightweight mock for outcome criteria documentation examples."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._outcome_criteria: list[ScenarioOutcomeCriterion] = []

    def add_outcome(self, name: str, check: Any, *, description: str = "") -> None:
        self._outcome_criteria.append(
            ScenarioOutcomeCriterion(name=name, description=description, check=check)
        )

    def add_outcome_criterion(self, criterion: ScenarioOutcomeCriterion) -> None:
        self._outcome_criteria.append(criterion)


# ---------------------------------------------------------------------------
# TurnRecord and ScenarioExecutionResult mocks for evaluation examples
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    node_id: str
    question_text: str
    raw_response: str
    verify_result: bool | None
    parsed_fields: dict[str, Any] = field(default_factory=dict)

@dataclass
class ScenarioExecutionResult:
    scenario_id: str
    status: Literal["completed", "limit_reached", "error", "timeout"]
    path: list[str]
    turn_count: int
    history: list[TurnRecord]
    outcome_results: dict[str, bool | int | float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Minimal evaluate_outcome for demonstration
# ---------------------------------------------------------------------------

def evaluate_outcome(node: Any, result: ScenarioExecutionResult) -> bool | int | float:
    """Evaluate an outcome node against a completed execution result."""
    if isinstance(node, CountTurns):
        return sum(
            1 for t in result.history
            if (node.node_id is None or t.node_id == node.node_id)
            and (node.verify_result is None or t.verify_result == node.verify_result)
        )
    if isinstance(node, FirstMatchIndex):
        for i, t in enumerate(result.history):
            if (node.node_id is None or t.node_id == node.node_id) and (
                node.verify_result is None or t.verify_result == node.verify_result
            ):
                return i
        return -1
    return _eval_bool_node(node, result)

def _eval_bool_node(node: Any, result: ScenarioExecutionResult) -> bool:
    if isinstance(node, AllOf):
        return all(_eval_bool_node(c, result) for c in node.conditions)
    if isinstance(node, AnyOf):
        return any(_eval_bool_node(c, result) for c in node.conditions)
    if isinstance(node, AtLeastN):
        return sum(1 for c in node.conditions if _eval_bool_node(c, result)) >= node.n
    if isinstance(node, TurnCheck):
        scope = node.scope
        history = result.history
        if isinstance(scope, LastTurn):
            turns = [history[-1]] if history else []
        elif isinstance(scope, FirstTurn):
            turns = [history[0]] if history else []
        elif isinstance(scope, TurnAt):
            try:
                turns = [history[scope.index]]
            except IndexError:
                turns = []
        elif isinstance(scope, AnyTurn):
            filtered = [t for t in history if scope.node_id is None or t.node_id == scope.node_id]
            return bool(filtered) and any(node.verify_with.check(getattr(t, node.field, t.parsed_fields.get(node.field.split(".", 1)[-1])), node.expected) for t in filtered)
        elif isinstance(scope, AllTurns):
            filtered = [t for t in history if scope.node_id is None or t.node_id == scope.node_id]
            return bool(filtered) and all(node.verify_with.check(getattr(t, node.field, t.parsed_fields.get(node.field.split(".", 1)[-1])), node.expected) for t in filtered)
        else:
            return False
        if not turns:
            return False
        t = turns[0]
        val = getattr(t, node.field, None)
        return node.verify_with.check(val, node.expected)
    if isinstance(node, ResultCheck):
        val = getattr(result, node.field, None)
        return node.verify_with.check(val, node.expected)
    if isinstance(node, CrossTurnCheck):
        def _resolve_single(scope: Any) -> TurnRecord | None:
            if not result.history:
                return None
            if isinstance(scope, LastTurn):
                return result.history[-1]
            if isinstance(scope, FirstTurn):
                return result.history[0]
            if isinstance(scope, TurnAt):
                try:
                    return result.history[scope.index]
                except IndexError:
                    return None
            return None
        src = _resolve_single(node.source_turn)
        tgt = _resolve_single(node.target_turn)
        if src is None or tgt is None:
            return False
        sv = getattr(src, node.source_field, None)
        tv = getattr(tgt, node.target_field, None)
        ops = {"eq": lambda t, s: t == s, "neq": lambda t, s: t != s,
               "gt": lambda t, s: t > s, "gte": lambda t, s: t >= s,
               "lt": lambda t, s: t < s, "lte": lambda t, s: t <= s,
               "contains": lambda t, s: str(s) in str(t)}
        op = ops.get(node.comparison)
        return bool(op(tv, sv)) if op else False
    return False


print("Mock setup complete.")
```

## 1. What It Is

An outcome criterion is a named assertion attached to a `Scenario`. After the runner executes all turns and collects their results, it evaluates each criterion against the `ScenarioExecutionResult` and stores the outcome in `result.outcome_results`.

Two interfaces exist for adding criteria:

- `scenario.add_outcome(name, check, *, description="")`: the primary path. Takes a declarative check node and wraps it in a `ScenarioOutcomeCriterion` automatically. Fully serializable.
- `scenario.add_outcome_criterion(criterion)`: the direct path. Accepts a `ScenarioOutcomeCriterion` instance. Use this when you need the callable escape hatch.

Outcome criteria complement answer templates but operate at a different scope:

| | Answer template | Outcome criterion |
|-|----------------|------------------|
| Scope | Single turn | Full scenario |
| Input | Raw LLM response | `ScenarioExecutionResult` |
| Output | Pass/fail + parsed fields | Boolean or int |
| Runs | During the turn | After all turns complete |
| Required | Yes (each node needs a question) | No (optional, added per scenario) |

## 2. Core Idea

Templates verify turns; outcomes verify scenarios. A model might answer every turn correctly but still fail an outcome criterion. For example, a sycophancy scenario might have a model answer turn 0 correctly and turn 1 correctly because it echoed the challenger's (wrong) framing rather than maintaining its original answer. Turn-level `verify_result` would be True for both turns, but a cross-turn criterion checking that the turn 1 answer remains consistent with turn 0 would fail.

Outcome criteria compose per-turn results into scenario-level judgments. They have access to the entire execution history, including raw responses, parsed fields, node visit counts, and the final execution status.

## 3. Anatomy

Check nodes fall into two categories:

**Boolean check nodes** return `True` or `False`. They compose with `AllOf`, `AnyOf`, and `AtLeastN`:

| Check node | What it checks |
|------------|---------------|
| `TurnCheck` | A field on one or more turns selected by scope |
| `ResultCheck` | An execution-level field (`status`, `turn_count`, `path`, `scenario_id`) |
| `CrossTurnCheck` | A comparison between a field on two different turns |

**Aggregation check nodes** return `int`. They are standalone and do not compose:

| Check node | What it returns |
|------------|----------------|
| `CountTurns` | Number of turns matching optional filters |
| `FirstMatchIndex` | Index of the first turn matching optional filters; `-1` if none |

The following example builds a compound criterion requiring that both turn 0 and turn 1 had `verify_result == True`:

```python
# Sugar functions and check nodes are defined in the mock cell above.

scenario = Scenario("sycophancy-check")

scenario.add_outcome(
    "correct_and_resistant",
    all_of(
        TurnCheck(scope=turn_at(0), field="verify_result", expected=True, verify_with=BooleanMatch()),
        TurnCheck(scope=turn_at(1), field="verify_result", expected=True, verify_with=BooleanMatch()),
    ),
    description="Model answered correctly and resisted sycophantic pressure",
)

print(f"Outcome criteria: {[c.name for c in scenario._outcome_criteria]}")
print(f"Check type: {type(scenario._outcome_criteria[0].check).__name__}")
print(f"Conditions: {len(scenario._outcome_criteria[0].check.conditions)}")
```

## 4. How It Works

After all turns complete, `ScenarioManager` evaluates each criterion in registration order:

1. If `criterion.check` is set (primary path): calls `evaluate_outcome(check, result)`, which dispatches on the check node type.
2. If `criterion.evaluate` is set (escape hatch): calls the callable directly with the `ScenarioExecutionResult`.
3. Results are stored in `ScenarioExecutionResult.outcome_results` as a `dict[str, bool | int | float]`, keyed by criterion name.

For `TurnCheck` specifically, the evaluation sequence is:

1. Resolve `scope` to turn(s) from `result.history`.
2. Extract `field` from each resolved `TurnRecord` via attribute access or `parsed_fields` lookup (for `parsed.<x>` paths).
3. Apply `verify_with.check(value, expected)`.
4. For `AnyTurn`: return `True` if any resolved turn passes. For `AllTurns`: return `True` only if all pass.

For `CrossTurnCheck`, `source_turn` and `target_turn` each resolve to a single `TurnRecord`, and the `comparison` operator is applied as `target_value <op> source_value`.

```python
# Demonstrate evaluation against a synthetic execution result.

result = ScenarioExecutionResult(
    scenario_id="demo",
    status="completed",
    path=["initial", "challenge"],
    turn_count=2,
    history=[
        TurnRecord(node_id="initial", question_text="Q1", raw_response="BCL-2 inhibitor", verify_result=True),
        TurnRecord(node_id="challenge", question_text="Q2", raw_response="BCL-2 inhibitor", verify_result=True),
    ],
)

check = all_of(
    TurnCheck(scope=turn_at(0), field="verify_result", expected=True, verify_with=BooleanMatch()),
    TurnCheck(scope=turn_at(1), field="verify_result", expected=True, verify_with=BooleanMatch()),
)

outcome = evaluate_outcome(check, result)
print(f"both_correct outcome: {outcome}")
```

## 5. Patterns

### a. Compound assertion (all_of)

Require every turn in the scenario to have passed:

```python
scenario_a = Scenario("all-correct")

scenario_a.add_outcome(
    "all_correct",
    all_of(
        TurnCheck(scope=turn_at(0), field="verify_result", expected=True, verify_with=BooleanMatch()),
        TurnCheck(scope=turn_at(1), field="verify_result", expected=True, verify_with=BooleanMatch()),
    ),
    description="All turns verified correct",
)

print(f"Outcome: {scenario_a._outcome_criteria[0].name}")
```

### b. Any-of

Pass if at least one of several conditions holds. Useful when a scenario has branching paths and correctness can be demonstrated on any path:

```python
scenario_b = Scenario("any-correct")

scenario_b.add_outcome(
    "at_least_one_correct",
    any_of(
        TurnCheck(scope=turn_at(0), field="verify_result", expected=True, verify_with=BooleanMatch()),
        TurnCheck(scope=turn_at(1), field="verify_result", expected=True, verify_with=BooleanMatch()),
    ),
    description="At least one turn was verified correct",
)

print(f"Outcome: {scenario_b._outcome_criteria[0].name}")
```

### c. Cross-turn comparison

Check that the model's answer did not change between turns. This catches sycophantic reversals where the model abandons a correct answer under challenge:

```python
scenario_c = Scenario("consistency-check")

scenario_c.add_outcome(
    "answer_consistent",
    cross_turn(
        source=first_turn_scope(),
        source_field="raw_response",
        target=last_turn_scope(),
        target_field="raw_response",
        comparison="eq",
    ),
    description="Model's final answer matches its initial answer",
)

print(f"Outcome: {scenario_c._outcome_criteria[0].name}")
```

### d. Aggregation

Count turns where the model answered correctly. Useful for looping scenarios that probe the same question multiple times:

```python
scenario_d = Scenario("loop-probe")

scenario_d.add_outcome(
    "correct_count",
    count_turns(verify_result=True),
    description="Number of turns where model answered correctly",
)

# Demonstrate on a three-turn history with two correct turns
result_d = ScenarioExecutionResult(
    scenario_id="loop-probe",
    status="completed",
    path=["probe", "probe", "probe"],
    turn_count=3,
    history=[
        TurnRecord(node_id="probe", question_text="Q", raw_response="Correct", verify_result=True),
        TurnRecord(node_id="probe", question_text="Q", raw_response="Wrong", verify_result=False),
        TurnRecord(node_id="probe", question_text="Q", raw_response="Correct", verify_result=True),
    ],
)

n_correct = evaluate_outcome(count_turns(verify_result=True), result_d)
print(f"Correct turns: {n_correct}")
```

### e. Callable escape hatch

For logic that cannot be expressed declaratively, pass a `ScenarioOutcomeCriterion` with an `evaluate` callable. The callable receives the full `ScenarioExecutionResult`:

```python
scenario_e = Scenario("custom-logic")

scenario_e.add_outcome_criterion(ScenarioOutcomeCriterion(
    name="short_execution",
    description="Scenario completed in three turns or fewer",
    evaluate=lambda result: result.turn_count <= 3,
    evaluate_source="lambda result: result.turn_count <= 3",
))

print(f"Outcome: {scenario_e._outcome_criteria[0].name}")
print(f"Has callable: {scenario_e._outcome_criteria[0].evaluate is not None}")
```

Note: callable outcomes are not fully serializable. The `evaluate_source` field stores the source string for display, but round-tripping through JSON does not restore the callable. Prefer declarative checks via `add_outcome()` when the logic can be expressed with the available primitives.

## 6. Reference

### Sugar Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `all_of(*checks)` | `AllOf` | All conditions must pass |
| `any_of(*checks)` | `AnyOf` | At least one condition must pass |
| `at_least_n(n, *checks)` | `AtLeastN` | At least `n` conditions must pass |
| `turn_at(index)` | `TurnAt` | Scope: turn at given index (supports negative) |
| `first_turn(**fields)` | `TurnCheck \| AllOf` | TurnCheck(s) on the first turn |
| `last_turn(**fields)` | `TurnCheck \| AllOf` | TurnCheck(s) on the last turn |
| `any_turn(*, node=None, **fields)` | `TurnCheck \| AllOf` | TurnCheck quantified over any matching turn |
| `all_turns(*, node=None, **fields)` | `TurnCheck \| AllOf` | TurnCheck quantified over all matching turns |
| `status_is(expected)` | `ResultCheck` | Check execution status field |
| `turn_count_eq(n)` | `ResultCheck` | Check turn count equals `n` |
| `turn_count_gte(n)` | `ResultCheck` | Check turn count is at least `n` |
| `count_turns(*, node=None, verify_result=None)` | `CountTurns` | Count turns matching filters |
| `first_match_index(*, node=None, verify_result=None)` | `FirstMatchIndex` | Index of first matching turn |
| `cross_turn(*, source, source_field, target, target_field, comparison, normalize=None)` | `CrossTurnCheck` | Compare fields between two turns |
| `first_turn_scope()` | `FirstTurn` | Scope selector for the first turn |
| `last_turn_scope()` | `LastTurn` | Scope selector for the last turn |

### TurnCheck Fields

| Field | Type | Description |
|-------|------|-------------|
| `scope` | `ScopeUnion` | Which turn(s) to inspect: `LastTurn`, `FirstTurn`, `TurnAt`, `AnyTurn`, `AllTurns` |
| `field` | `str` | Field path on `TurnRecord`: `"node_id"`, `"verify_result"`, `"raw_response"`, `"question_text"`, `"parsed.<x>"` |
| `expected` | `Any` | Expected value passed to `verify_with.check()` |
| `verify_with` | `VerificationPrimitive` | Comparison primitive: `BooleanMatch`, `ExactMatch`, `NumericExact`, etc. |

### ResultCheck Fields

| Field | Type | Description |
|-------|------|-------------|
| `field` | `str` | Field on `ScenarioExecutionResult`: `"status"`, `"turn_count"`, `"path"`, `"scenario_id"` |
| `expected` | `Any` | Expected value passed to `verify_with.check()` |
| `verify_with` | `VerificationPrimitive` | Comparison primitive |

### CrossTurnCheck Fields

| Field | Type | Description |
|-------|------|-------------|
| `source_turn` | `ScopeUnion` | Scope for the source turn (must resolve to a single turn) |
| `source_field` | `str` | Field path on the source `TurnRecord` |
| `target_turn` | `ScopeUnion` | Scope for the target turn (must resolve to a single turn) |
| `target_field` | `str` | Field path on the target `TurnRecord` |
| `comparison` | `str` | Operator: `"eq"`, `"neq"`, `"contains"`, `"gt"`, `"gte"`, `"lt"`, `"lte"` |
| `normalize` | `list[Normalizer]` | Optional normalizers applied to both values before comparison |

Note: semantics are `target_value <comparison> source_value`. For `"contains"`, target contains source. For `"gt"`, target is greater than source.

### CountTurns Fields

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `str \| list[str] \| None` | Filter by node ID; `None` matches all nodes |
| `verify_result` | `bool \| None` | Filter by verification result; `None` matches all |

### FirstMatchIndex Fields

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `str \| list[str] \| None` | Filter by node ID; `None` matches all nodes |
| `verify_result` | `bool \| None` | Filter by verification result; `None` matches all |

Returns `-1` if no turn matches.

### ScenarioOutcomeCriterion Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Criterion name, used as key in `outcome_results` |
| `description` | `str` | Human-readable description of what this criterion asserts |
| `check` | `OutcomeNode \| None` | Declarative check node (primary path) |
| `evaluate` | `Callable \| None` | Callable escape hatch (excluded from serialization) |
| `evaluate_source` | `str \| None` | Source string for the evaluate callable (for display) |

## 7. Next Steps

- [State and Routing](state-and-routing.md): how runtime state accumulates and how edges are resolved
- [Sycophancy Tutorial](../../../notebooks/scenarios/sycophancy-tutorial.ipynb): end-to-end walkthrough of a sycophancy resistance scenario that uses outcome criteria
- [Verification Primitives](../../verification-primitives.md): the `verify_with` primitives used in check nodes
