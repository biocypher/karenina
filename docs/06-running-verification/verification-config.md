---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Configuring Verification

Every verification run is controlled by a `VerificationConfig` object. This page walks through the major configuration categories with practical examples. For an exhaustive field-by-field reference, see [VerificationConfig Reference](../10-configuration-reference/verification-config.md).

```python tags=["hide-cell"]
# Mock cell: patches VerificationConfig validation so examples execute without live API keys.
# This cell is hidden in the rendered documentation.
from unittest.mock import patch

from karenina.schemas.verification import VerificationConfig

_patcher_validate = patch.object(
    VerificationConfig, "_validate_config", lambda self: None
)
_patcher_validate.start()
```

## Overview

`VerificationConfig` is the central configuration object for running verification. It controls:

- **Which models** generate answers and parse responses
- **What evaluation mode** to use (template-only, template+rubric, rubric-only)
- **Which features** are enabled (abstention detection, embedding checks, deep judgment, etc.)
- **Execution settings** (async parallelism, replicate count)

Configuration flows through a precedence hierarchy:

    CLI arguments > Preset values > Environment variables > Built-in defaults

For details on this hierarchy, see [Configuration Hierarchy](../03-configuration/index.md).

## Model Configuration

Every verification run needs at least one **parsing model** (the Judge LLM that extracts structured data). Most runs also need an **answering model** (the model being evaluated).

```python
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification import VerificationConfig

# Minimal configuration: one answering model + one parsing model
config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="answering-gpt4mini",
            model_name="gpt-4.1-mini",
            model_provider="openai",
            interface="langchain",
        )
    ],
    parsing_models=[
        ModelConfig(
            id="parsing-gpt4mini",
            model_name="gpt-4.1-mini",
            model_provider="openai",
            interface="langchain",
        )
    ],
)

print(f"Answering models: {len(config.answering_models)}")
print(f"Parsing models: {len(config.parsing_models)}")
print(f"Interface: {config.answering_models[0].interface}")
```

### ModelConfig Fields

Each `ModelConfig` specifies how to connect to an LLM:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | — | Unique identifier for this model configuration (required) |
| `model_name` | `str` | — | Model identifier (e.g., `"gpt-4.1-mini"`, `"claude-sonnet-4-20250514"`) |
| `model_provider` | `str` | — | Provider name (e.g., `"openai"`, `"anthropic"`) |
| `interface` | `str` | `"langchain"` | Adapter backend (see [Adapters](../04-core-concepts/adapters.md)) |
| `temperature` | `float` | `0.1` | Sampling temperature |
| `max_tokens` | `int` | `8192` | Maximum response tokens |
| `system_prompt` | `str` | auto | Auto-set based on role (answering vs parsing) |

System prompts are automatically applied if not explicitly set:

- **Answering models**: *"You are an expert assistant. Answer the question accurately and concisely."*
- **Parsing models**: *"You are a validation assistant. Parse and validate responses against the given Pydantic template."*

### Using Different Interfaces

You can mix interfaces — for example, use Claude Agent SDK for answering (to get native tool use) and LangChain for parsing:

```python
config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="answering-claude",
            model_name="claude-sonnet-4-20250514",
            interface="claude_agent_sdk",
        )
    ],
    parsing_models=[
        ModelConfig(
            id="parsing-gpt4mini",
            model_name="gpt-4.1-mini",
            model_provider="openai",
            interface="langchain",
        )
    ],
)

print(f"Answering interface: {config.answering_models[0].interface}")
print(f"Parsing interface: {config.parsing_models[0].interface}")
```

The six available interfaces are: `langchain` (default, multi-provider), `openrouter`, `openai_endpoint`, `claude_agent_sdk`, `claude_tool`, and `manual`. See [Adapters Overview](../04-core-concepts/adapters.md) for when to use each.

## Evaluation Modes

The `evaluation_mode` field determines which pipeline stages run:

| Mode | Runs Templates | Runs Rubrics | Use Case |
|------|:-:|:-:|----------|
| `"template_only"` | Yes | No | Verify factual correctness (default) |
| `"template_and_rubric"` | Yes | Yes | Correctness + quality assessment |
| `"rubric_only"` | No | Yes | Quality-only evaluation (no templates needed) |

```python
# Template + rubric mode
config = VerificationConfig(
    answering_models=[
        ModelConfig(id="answering", model_name="gpt-4.1-mini", model_provider="openai")
    ],
    parsing_models=[
        ModelConfig(id="parsing", model_name="gpt-4.1-mini", model_provider="openai")
    ],
    evaluation_mode="template_and_rubric",
    rubric_enabled=True,
)

print(f"Evaluation mode: {config.evaluation_mode}")
print(f"Rubric enabled: {config.rubric_enabled}")
```

!!! note "Consistency requirement"
    `evaluation_mode` and `rubric_enabled` must be consistent:

    - `"template_and_rubric"` and `"rubric_only"` both require `rubric_enabled=True`
    - `"template_only"` requires `rubric_enabled=False` (the default)

For a detailed comparison of what each mode includes, see [Evaluation Modes](../04-core-concepts/evaluation-modes.md).

## Feature Flags

Feature flags enable optional pipeline stages. All are disabled by default.

### Abstention and Sufficiency Detection

These run *before* parsing, saving cost by short-circuiting evaluation for problematic responses:

| Flag | Default | Effect |
|------|---------|--------|
| `abstention_enabled` | `False` | Detect model refusals/evasions — auto-fail if detected |
| `sufficiency_enabled` | `False` | Detect incomplete responses — auto-fail if insufficient |

```python
config = VerificationConfig(
    answering_models=[
        ModelConfig(id="answering", model_name="gpt-4.1-mini", model_provider="openai")
    ],
    parsing_models=[
        ModelConfig(id="parsing", model_name="gpt-4.1-mini", model_provider="openai")
    ],
    abstention_enabled=True,
    sufficiency_enabled=True,
)

print(f"Abstention: {config.abstention_enabled}")
print(f"Sufficiency: {config.sufficiency_enabled}")
```

### Embedding Check

Semantic similarity fallback that runs *after* template verification:

| Flag | Default | Description |
|------|---------|-------------|
| `embedding_check_enabled` | `False` | Enable semantic similarity verification |
| `embedding_check_model` | `"all-MiniLM-L6-v2"` | SentenceTransformer model name |
| `embedding_check_threshold` | `0.85` | Similarity threshold (0.0–1.0) |

These can also be set via environment variables (`EMBEDDING_CHECK`, `EMBEDDING_CHECK_MODEL`, `EMBEDDING_CHECK_THRESHOLD`).

### Deep Judgment

Multi-stage parsing with excerpt extraction and reasoning. Runs after standard template verification:

| Flag | Default | Description |
|------|---------|-------------|
| `deep_judgment_enabled` | `False` | Enable deep judgment for templates |
| `deep_judgment_max_excerpts_per_attribute` | `3` | Max excerpts extracted per attribute |
| `deep_judgment_fuzzy_match_threshold` | `0.80` | Fuzzy match similarity threshold |
| `deep_judgment_excerpt_retry_attempts` | `2` | Retry attempts for excerpt extraction |
| `deep_judgment_search_enabled` | `False` | Enable web search validation for excerpts |

For rubric deep judgment, see the `deep_judgment_rubric_mode` field and [Deep Judgment for Rubrics](../11-advanced-pipeline/deep-judgment-rubrics.md).

## Rubric Settings

When rubrics are enabled (`evaluation_mode` set to `"template_and_rubric"` or `"rubric_only"`), additional settings control rubric behavior:

| Field | Default | Description |
|-------|---------|-------------|
| `rubric_evaluation_strategy` | `"batch"` | `"batch"` (all traits in one LLM call) or `"sequential"` (one-by-one) |
| `rubric_trait_names` | `None` | Optional filter — evaluate only these trait names |

```python
# Rubric-only mode evaluating specific traits
config = VerificationConfig(
    answering_models=[
        ModelConfig(id="answering", model_name="gpt-4.1-mini", model_provider="openai")
    ],
    parsing_models=[
        ModelConfig(id="parsing", model_name="gpt-4.1-mini", model_provider="openai")
    ],
    evaluation_mode="rubric_only",
    rubric_enabled=True,
    rubric_trait_names=["safety", "conciseness"],
    rubric_evaluation_strategy="sequential",
)

print(f"Mode: {config.evaluation_mode}")
print(f"Strategy: {config.rubric_evaluation_strategy}")
print(f"Trait filter: {config.rubric_trait_names}")
```

## Async Execution

Verification runs in parallel by default:

| Field | Default | Description |
|-------|---------|-------------|
| `async_enabled` | `True` | Enable parallel execution of verification tasks |
| `async_max_workers` | `2` | Number of concurrent workers |

These can also be set via environment variables (`KARENINA_ASYNC_ENABLED`, `KARENINA_ASYNC_MAX_WORKERS`).

```python
# Increase parallelism for large benchmarks
config = VerificationConfig(
    answering_models=[
        ModelConfig(id="answering", model_name="gpt-4.1-mini", model_provider="openai")
    ],
    parsing_models=[
        ModelConfig(id="parsing", model_name="gpt-4.1-mini", model_provider="openai")
    ],
    async_enabled=True,
    async_max_workers=5,
)

print(f"Async: {config.async_enabled}")
print(f"Workers: {config.async_max_workers}")
```

## Trace Filtering (MCP)

When using MCP-enabled agents, these flags control what portion of the agent trace is passed to evaluation:

| Field | Default | Description |
|-------|---------|-------------|
| `use_full_trace_for_template` | `False` | Pass full agent trace to template parsing (vs final AI message only) |
| `use_full_trace_for_rubric` | `True` | Pass full agent trace to rubric evaluation (vs final AI message only) |

The full trace is always captured and stored in `raw_llm_response` regardless of these settings. The flags only control what input the parsing/evaluation models see.

## Replicate Count

Run each question–model combination multiple times:

```python
config = VerificationConfig(
    answering_models=[
        ModelConfig(id="answering", model_name="gpt-4.1-mini", model_provider="openai")
    ],
    parsing_models=[
        ModelConfig(id="parsing", model_name="gpt-4.1-mini", model_provider="openai")
    ],
    replicate_count=3,
)

print(f"Replicates: {config.replicate_count}")
```

This produces 3 results per question per model combination, useful for measuring variance in LLM outputs.

## Using `from_overrides`

The `from_overrides` class method is the most convenient way to create a config with selective overrides. It implements the full precedence hierarchy: `overrides > base config > defaults`.

```python
# Start from defaults, override just the models
config = VerificationConfig.from_overrides(
    answering_model="gpt-4.1-mini",
    answering_provider="openai",
    answering_id="answering",
    parsing_model="gpt-4.1-mini",
    parsing_provider="openai",
    parsing_id="parsing",
    abstention=True,
    embedding_check=True,
    replicate_count=2,
)

print(f"Answering: {config.answering_models[0].model_name}")
print(f"Abstention: {config.abstention_enabled}")
print(f"Embedding check: {config.embedding_check_enabled}")
print(f"Replicates: {config.replicate_count}")
```

You can also apply overrides to a base config loaded from a preset:

```python
# Hypothetical: load preset then override specific settings
# base = VerificationConfig.from_preset(Path("presets/default.json"))
# config = VerificationConfig.from_overrides(
#     base,
#     answering_model="claude-sonnet-4-20250514",
#     answering_id="answering-claude",
#     deep_judgment=True,
# )
```

## Inspecting Configuration

`VerificationConfig` has a detailed `repr` that shows all active settings:

```python
config = VerificationConfig.from_overrides(
    answering_model="gpt-4.1-mini",
    answering_provider="openai",
    answering_id="answering",
    parsing_model="gpt-4.1-mini",
    parsing_provider="openai",
    parsing_id="parsing",
    abstention=True,
    embedding_check=True,
)

print(config)
```

This prints a structured overview of models, execution settings, and enabled features — useful for verifying your configuration before running.

```python tags=["hide-cell"]
# Clean up the mock
_ = _patcher_validate.stop()
```

---

## Next Steps

- [Python API Verification](python-api.md) — run verification with your config
- [Using Presets](using-presets.md) — save and reuse configurations
- [PromptConfig](prompt-config.md) — inject custom instructions into pipeline stages
- [Response Quality Checks](response-quality-checks.md) — abstention and sufficiency detection
- [VerificationConfig Reference](../10-configuration-reference/verification-config.md) — exhaustive field table
