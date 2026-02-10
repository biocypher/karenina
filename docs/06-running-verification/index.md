# Running Verification

This section walks through the complete workflow for executing a benchmark against LLMs — from loading a checkpoint to inspecting verification results.

## Workflow Overview

```
Load benchmark from checkpoint or database
    │
    ▼
Configure verification (models, evaluation mode, features)
    │
    ▼
Run verification ─── Python API ─── or ─── CLI
    │
    ▼
Results generated (per question × model × replicate)
    │
    ▼
Inspect, analyze, and export results
```

Each step has a dedicated page with detailed instructions and examples.

---

## Workflow Steps

### 1. Load a Benchmark

Start by loading an existing benchmark that contains questions and evaluation criteria (templates and/or rubrics):

```python
from karenina import Benchmark

benchmark = Benchmark.load("my_benchmark.jsonld")

# Inspect what's loaded
print(f"Questions: {len(benchmark.questions)}")
```

[Load and inspect a benchmark →](loading-benchmark.md)

### 2. Configure Verification

Set up `VerificationConfig` to control how verification runs — which models to use, what evaluation mode, and which optional features to enable:

```python
from karenina.schemas import VerificationConfig, ModelConfig

config = VerificationConfig(
    answering_models=[
        ModelConfig(id="claude-haiku-4-5", model_provider="anthropic",
                    model_name="claude-haiku-4-5", interface="langchain")
    ],
    parsing_models=[
        ModelConfig(id="haiku-judge", model_provider="anthropic",
                    model_name="claude-haiku-4-5", temperature=0.0,
                    interface="langchain")
    ],
    evaluation_mode="template_only"
)
```

[Configure VerificationConfig →](verification-config.md) · [Inject custom instructions via PromptConfig →](prompt-config.md)

### 3. Run Verification

Execute verification using the Python API or CLI:

**Python API:**

```python
results = benchmark.run_verification(config)
print(f"Results: {len(results)} generated")
```

**CLI:**

```bash
karenina verify my_benchmark.jsonld --preset my_preset.json
```

[Run via Python API →](python-api.md) · [Run via CLI →](cli.md)

### 4. Inspect Results

Access results through the `VerificationResultSet` returned by `run_verification()`:

```python
# Access individual results
for result in results.results:
    print(f"Q: {result.question_id} → Pass: {result.verify_result}")

# Or convert to DataFrames for analysis
df = results.get_templates().to_dataframe()
```

[Understand result structure →](../07-analyzing-results/verification-result.md) · [Analyze with DataFrames →](../07-analyzing-results/dataframe-analysis.md)

---

## Configuration Options

The verification system provides several configuration layers:

| Configuration | Purpose | Details |
|---------------|---------|---------|
| **VerificationConfig** | Models, evaluation mode, feature flags | [Tutorial](verification-config.md) · [Reference](../10-configuration-reference/verification-config.md) |
| **PromptConfig** | Custom instructions for LLM prompts | [Tutorial](prompt-config.md) · [Reference](../10-configuration-reference/prompt-config.md) |
| **Presets** | Reusable configuration bundles | [Using presets](using-presets.md) · [Creating presets](../03-configuration/presets.md) |
| **Environment variables** | API keys, paths, defaults | [Tutorial](../03-configuration/environment-variables.md) · [Reference](../10-configuration-reference/env-vars.md) |

---

## Execution Methods

Karenina supports multiple ways to run verification:

| Method | Best For | Page |
|--------|----------|------|
| **Python API** | Programmatic control, notebooks, custom workflows | [Python API](python-api.md) |
| **CLI** | Terminal use, CI/CD, scripting | [CLI](cli.md) |
| **Manual interface** | Pre-recorded traces, reproducibility, cost reduction | [Manual interface](manual-interface.md) |

---

## Optional Features

The verification pipeline supports several optional features that can be toggled via `VerificationConfig`:

### Response Quality Checks

Detect problematic responses *before* parsing to save cost:

- **Abstention detection** — Identify when a model refuses to answer
- **Sufficiency detection** — Identify incomplete or inadequate responses

Both run before the parsing stage. If detected, the pipeline skips parsing and marks the result accordingly.

[Response quality checks →](response-quality-checks.md)

### Multi-Model Evaluation

Compare performance across multiple LLMs by specifying multiple answering and/or parsing models. Karenina automatically generates all combinations and caches answers for efficiency.

[Multi-model evaluation →](multi-model.md)

### MCP-Enabled Verification

Run tool-augmented evaluation by connecting LLMs to MCP servers. Configure tool access, middleware settings, and trace handling.

[MCP-enabled verification →](mcp-verification.md)

### Database Persistence

Automatically save results to a SQLite database as verification completes. Results are immediately queryable without manual export.

[Database persistence →](database-persistence.md)

---

## What You Need

Before running verification, ensure you have:

| Requirement | Why | Where to Set Up |
|-------------|-----|-----------------|
| **Benchmark with questions** | Questions to evaluate | [Creating Benchmarks](../05-creating-benchmarks/index.md) |
| **Templates and/or rubrics** | Evaluation criteria | [Answer Templates](../core_concepts/answer-templates.md) · [Rubrics](../core_concepts/rubrics/index.md) |
| **LLM API keys** | Access to answering and parsing models | [Environment Variables](../03-configuration/environment-variables.md) |
| **Evaluation mode choice** | What to evaluate (correctness, quality, or both) | [Evaluation Modes](../core_concepts/evaluation-modes.md) |

---

## Next Steps

After running verification:

- [Analyzing Results](../07-analyzing-results/index.md) — Inspect result structure, build DataFrames, export data
- [Creating Benchmarks](../05-creating-benchmarks/index.md) — If you haven't created a benchmark yet
