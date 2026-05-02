# Running Verification

This section walks through running verification end-to-end — from loading a saved benchmark to inspecting results. Each tutorial is a complete, self-contained scenario focused on a specific verification workflow.

Choose the scenario that matches your evaluation needs, or work through them in order of increasing complexity.

## Choose Your Scenario

| Scenario | Focus Area | What You'll Learn |
|----------|-----------|-------------------|
| [Basic Verification](../../notebooks/running-verification/basic-verification.ipynb) | Template-only evaluation | Load, configure, run, inspect — the simplest verification path with `VerificationConfig`, result iteration, CLI equivalents |
| [Full Evaluation](../../notebooks/running-verification/full-evaluation.ipynb) | Template + rubric | Enable rubrics, abstention/sufficiency checks, embedding verification, `PromptConfig`, presets |
| [Multi-Model Comparison](../../notebooks/running-verification/multi-model-comparison.ipynb) | Comparing models | Multiple answering models, answer caching, replicates, DataFrames, model-level grouping |
| [Deep Judgment](../../notebooks/running-verification/deep-judgment.ipynb) | Excerpt-based reasoning | Deep judgment for templates and rubrics, excerpt extraction, hallucination risk, search validation |
| [MCP Agent Evaluation](../../notebooks/running-verification/mcp-agent-evaluation.ipynb) | Tool-using agents | MCP tool configuration, agent middleware, trace handling, recursion limits |
| [Agentic Evaluation](../../notebooks/running-verification/agentic-evaluation.ipynb) | Workspace-based agents | Agentic parsing, investigation judge, `VerifiedField` primitives, agentic rubric traits, context modes |
| [Manual Interface](../../notebooks/running-verification/manual-interface-workflow.ipynb) | Pre-recorded traces | Offline evaluation with pre-recorded responses, template iteration, parsing model comparison |
| [Progressive Save](../../notebooks/running-verification/progressive-save.ipynb) | Resumable runs | Checkpoint progress incrementally, resume interrupted runs, `.state` + `.results.jsonl` sidecars, `ProgressiveFileSink` |
| [Few-Shot Configuration](../../notebooks/running-verification/few-shot-configuration.ipynb) | Example injection | Global modes (all, k-shot, custom, none), per-question overrides, `FewShotConfig`, example resolution |

---

## Common Workflow

All nine scenarios follow this general pattern:

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
| Full configuration | `VerificationConfig(answering_models=[...], ...)` | [Basic Verification](../../notebooks/running-verification/basic-verification.ipynb) |
| Quick configuration | `VerificationConfig.from_overrides(...)` | [Basic Verification](../../notebooks/running-verification/basic-verification.ipynb) |
| Load from preset | `VerificationConfig.from_preset(path)` | [Full Evaluation](../../notebooks/running-verification/full-evaluation.ipynb) |
| Run all questions | `benchmark.run_verification(config)` | All scenarios |
| Run subset | `benchmark.run_verification(config, question_ids=[...])` | [Basic Verification](../../notebooks/running-verification/basic-verification.ipynb) |
| Iterate results | `for result in results: ...` | All scenarios |
| Summary statistics | `results.get_summary()` | [Basic Verification](../../notebooks/running-verification/basic-verification.ipynb) |
| Filter results | `results.filter(completed_only=True)` | [Basic Verification](../../notebooks/running-verification/basic-verification.ipynb) |
| Group by model | `results.group_by_model()` | [Multi-Model Comparison](../../notebooks/running-verification/multi-model-comparison.ipynb) |
| Group by question | `results.group_by_question()` | [Multi-Model Comparison](../../notebooks/running-verification/multi-model-comparison.ipynb) |
| DataFrame analysis | `results.get_template_results().to_dataframe()` | [Multi-Model Comparison](../../notebooks/running-verification/multi-model-comparison.ipynb) |
| CLI verification | `karenina verify checkpoint.jsonld --preset ...` | All scenarios |

---

## Core Concepts

These concept pages provide the foundational knowledge that the scenarios build on:

- [Evaluation Modes](../../notebooks/core_concepts/evaluation-modes.ipynb) — How template-only, template+rubric, and rubric-only map to pipeline stages
- [Answer Templates](../../notebooks/core_concepts/answer-templates.ipynb) — Template structure, field types, `verify()` semantics
- [Rubrics](../../core_concepts/rubrics/index.md) — Trait types (LLM, regex, callable, metric), global vs per-question
- [Adapters](../../core_concepts/adapters.md) — Port/adapter architecture, available backends
- [Checkpoints](../../core_concepts/questions-and-benchmarks/checkpoints.md) — JSON-LD format, save/load behavior

---

## Next Steps

- [Creating Benchmarks](../creating-benchmarks/index.md) — Build a benchmark to verify
- [Analyzing Results](../analyzing-results/index.md) — DataFrame analysis, export, iteration
