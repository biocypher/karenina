# Running Verification

This section walks through running verification end-to-end — from loading a saved benchmark to inspecting results. Each tutorial is a complete, self-contained scenario focused on a specific verification workflow.

Choose the scenario that matches your evaluation needs, or work through them in order of increasing complexity.

## Choose Your Scenario

| Scenario | Focus Area | What You'll Learn |
|----------|-----------|-------------------|
| [Basic Verification](basic-verification.md) | Template-only evaluation | Load, configure, run, inspect — the simplest verification path with `VerificationConfig`, result iteration, CLI equivalents |
| [Full Evaluation](full-evaluation.md) | Template + rubric | Enable rubrics, abstention/sufficiency checks, embedding verification, `PromptConfig`, presets |
| [Multi-Model Comparison](multi-model-comparison.md) | Comparing models | Multiple answering models, answer caching, replicates, DataFrames, model-level grouping |
| [Deep Judgment](deep-judgment.md) | Excerpt-based reasoning | Deep judgment for templates and rubrics, excerpt extraction, hallucination risk, search validation |
| [MCP Agent Evaluation](mcp-agent-evaluation.md) | Tool-using agents | MCP tool configuration, agent middleware, trace handling, recursion limits |
| [Manual Interface](manual-interface-workflow.md) | Pre-recorded traces | Offline evaluation with pre-recorded responses, template iteration, parsing model comparison |

---

## Common Workflow

All six scenarios follow this general pattern:

```
Load benchmark
    │
    ▼
Configure verification (models, evaluation mode, features)
    │
    ▼
Run verification (all questions or a subset)
    │
    ▼
Inspect results (iterate, filter, group, summarize)
```

### Key APIs

| Operation | Method | Covered In |
|-----------|--------|------------|
| Load benchmark | `Benchmark.load("checkpoint.jsonld")` | All scenarios |
| Full configuration | `VerificationConfig(answering_models=[...], ...)` | [Basic Verification](basic-verification.md) |
| Quick configuration | `VerificationConfig.from_overrides(...)` | [Basic Verification](basic-verification.md) |
| Load from preset | `VerificationConfig.from_preset(path)` | [Full Evaluation](full-evaluation.md) |
| Run all questions | `benchmark.run_verification(config)` | All scenarios |
| Run subset | `benchmark.run_verification(config, question_ids=[...])` | [Basic Verification](basic-verification.md) |
| Iterate results | `for result in results: ...` | All scenarios |
| Summary statistics | `results.get_summary()` | [Basic Verification](basic-verification.md) |
| Filter results | `results.filter(completed_only=True)` | [Basic Verification](basic-verification.md) |
| Group by model | `results.group_by_model()` | [Multi-Model Comparison](multi-model-comparison.md) |
| Group by question | `results.group_by_question()` | [Multi-Model Comparison](multi-model-comparison.md) |
| DataFrame analysis | `results.get_template_results().to_dataframe()` | [Multi-Model Comparison](multi-model-comparison.md) |
| CLI verification | `karenina verify checkpoint.jsonld --preset ...` | All scenarios |

---

## Core Concepts

These concept pages provide the foundational knowledge that the scenarios build on:

- [Evaluation Modes](../../core_concepts/evaluation-modes.md) — How template-only, template+rubric, and rubric-only map to pipeline stages
- [Answer Templates](../../core_concepts/answer-templates.md) — Template structure, field types, `verify()` semantics
- [Rubrics](../../core_concepts/rubrics/index.md) — Trait types (LLM, regex, callable, metric), global vs per-question
- [Adapters](../../core_concepts/adapters.md) — Port/adapter architecture, available backends
- [Checkpoints](../../core_concepts/checkpoints.md) — JSON-LD format, save/load behavior

---

## Next Steps

- [Creating Benchmarks](../creating-benchmarks/index.md) — Build a benchmark to verify
- [Analyzing Results](../analyzing-results/index.md) — DataFrame analysis, export, iteration
