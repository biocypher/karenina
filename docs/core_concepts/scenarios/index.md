# Scenarios

Scenario benchmarks evaluate LLM behavior across a graph of connected conversation turns. Unlike single-turn benchmarks (which evaluate isolated questions, each independent of the others) and [TaskEval](../../notebooks/core_concepts/task-eval.ipynb) (which evaluates pre-recorded outputs without live LLM calls), scenarios run a live conversation where each turn's result can determine which question comes next. The evaluation covers the full execution: not just individual responses, but the path taken and the state accumulated across turns.

## Building Blocks

```
                   ┌─────────────────────────────────────────┐
                   │           SCENARIO GRAPH                │
                   │                                         │
                   │   ┌──────────┐      edge (condition)   │
                   │   │  Node A  │ ──────────────────────► │
                   │   │          │                          │
                   │   │ question │      edge (fallback)    │
                   │   │ template │ ──────────────────────► │
                   │   └──────────┘                          │
                   │        │                                │
                   │        ▼                                │
                   │   ┌──────────┐                          │
                   │   │  Node B  │                          │
                   │   │          │                          │
                   │   │ question │                          │
                   │   │ template │                          │
                   │   └──────────┘                          │
                   │        │                                │
                   │        ▼                                │
                   │  ┌───────────────────────────────────┐  │
                   │  │       Outcome Criteria            │  │
                   │  │  (assertions over full execution) │  │
                   │  └───────────────────────────────────┘  │
                   └─────────────────────────────────────────┘
```

Each scenario consists of three building block types: **nodes** (the questions and answer templates), **edges** (connections between nodes with optional routing conditions), and **outcome criteria** (assertions evaluated after the scenario finishes).

## When to Use Each Evaluation Mode

| Need | Mode | Why |
|------|------|-----|
| Evaluate isolated factual questions | Benchmark (single-turn) | Each question is independent; no conversation context is needed |
| Test multi-turn conversation dynamics | Scenarios | Branching paths, conversation history, and cross-turn assertions |
| Evaluate pre-recorded LLM outputs | [TaskEval](../../notebooks/core_concepts/task-eval.ipynb) | No live LLM calls; offline evaluation of existing traces |

### Nodes

A node carries a [Question](../questions-and-benchmarks/index.md) and its [answer template](../../notebooks/core_concepts/answer-templates.ipynb). The template controls how the model's response is parsed and verified at that turn, exactly as in a single-turn benchmark. Nodes can also specify per-node model overrides, allowing different turns to use different models within the same scenario.

### Edges

Edges connect nodes and determine the path the scenario takes at runtime. An edge can carry a condition (a predicate evaluated against the current turn's result and accumulated state); when the condition is satisfied, execution follows that edge. Unconditional edges serve as fallbacks when no conditional edge fires.

### Outcome Criteria

Outcome criteria are declarative assertions evaluated after the scenario completes. They can check properties of individual nodes (which path was taken, what a specific turn produced) or cross-turn properties (state values accumulated over the conversation). See [Outcome Criteria](../../notebooks/core_concepts/scenarios/outcome-criteria.ipynb) for the full API, including check nodes and sugar functions.

## Next Steps

- [Building Scenarios](../../notebooks/core_concepts/scenarios/building-scenarios.ipynb): builder API, graph construction
- [Outcome Criteria](../../notebooks/core_concepts/scenarios/outcome-criteria.ipynb): check nodes, sugar functions
- [State and Routing](../../notebooks/core_concepts/scenarios/state-and-routing.ipynb): runtime state, edge resolution
- [Sycophancy Tutorial](../../notebooks/scenarios/sycophancy-tutorial.ipynb): end-to-end walkthrough

Scenario runs render as single files with per-turn sections in [Error Analysis](../../workflows/analyzing-results/error-analysis.md).
