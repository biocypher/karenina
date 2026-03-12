---
jupyter:
  jupytext:
    formats: docs/core_concepts//md,docs/notebooks/core_concepts//ipynb
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

# Prompt Assembly

Prompt assembly is the system that constructs every LLM prompt in the verification pipeline. It combines three independent instruction sources (task, adapter, user) into a single `list[Message]` that is passed to the LLM adapter for invocation. You never call the assembler directly; the pipeline stages call it on your behalf. What you *can* control is the third layer: user instructions injected via `PromptConfig`.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
from unittest.mock import MagicMock, patch

mock_modules = {}
for mod in [
    "sqlalchemy", "sqlalchemy.orm", "sqlalchemy.ext",
    "sqlalchemy.ext.declarative", "sqlalchemy.engine",
    "sqlalchemy.sql", "sqlalchemy.event",
    "karenina.storage", "karenina.storage.base",
    "karenina.storage.engine", "karenina.storage.db_config",
    "karenina.storage.models", "karenina.storage.generated_models",
    "karenina.storage.auto_mapper", "karenina.storage.operations",
]:
    mock_modules[mod] = MagicMock()

_patcher = patch.dict("sys.modules", mock_modules)
_patcher.start()
```

## 1. What Is Prompt Assembly?

Every LLM call in the verification pipeline (parsing responses, detecting abstention, evaluating rubric traits, running deep judgment) needs a prompt. Prompt assembly is the mechanism that builds that prompt by layering three independent instruction sources in a fixed order:

1. **Task instructions**: what the LLM should do (built by the pipeline stage)
2. **Adapter instructions**: backend-specific adjustments (registered per adapter)
3. **User instructions**: your custom additions (from `PromptConfig` on `VerificationConfig`)

The assembler's job is narrow: take raw system/user text from a pipeline stage, append adapter-specific text, append user-provided text, and return messages ready for invocation. It does not decide *what* to evaluate, *which* model to call, or *how* to parse the result. Those responsibilities belong to the pipeline stages, adapter factory, and parser adapters respectively.

**Abstraction boundary**: the assembler sees raw text strings and adapter capabilities. It does not see the answer template, the rubric traits, or the question data directly. Those are already encoded into the task instructions by the time the assembler receives them.

## 2. Why the Abstraction Exists

Without prompt assembly, every pipeline stage that calls an LLM would need to know which adapter is in use and how to format prompts for it. This creates three problems:

1. **Coupling**: pipeline stages become tied to specific LLM backends
2. **Duplication**: every stage reimplements adapter-aware prompt formatting
3. **Rigidity**: users cannot customize prompts without modifying pipeline internals

The tri-section pattern solves all three. Pipeline stages produce format-agnostic task instructions. Adapters register their own prompt modifications declaratively. Users inject instructions via configuration. No layer needs to know about the others.

The result: the same pipeline code works across all backends (LangChain, Claude Tool, Claude Agent SDK, Manual), and you can customize any LLM call's prompt without touching a single line of pipeline or adapter code.

## 3. The Tri-Section Pattern

```
┌──────────────────────────────────────┐
│  1. Task Instructions                │  Built by the pipeline stage
│     (what the LLM should do)         │  (TemplatePromptBuilder, abstention
│                                      │   detection prompts, rubric builders)
├──────────────────────────────────────┤
│  2. Adapter Instructions             │  Registered per (interface, task) pair
│     (backend-specific adjustments)   │  (JSON schema formatting, extraction
│                                      │   directives, output format guidance)
├──────────────────────────────────────┤
│  3. User Instructions                │  Optional text from PromptConfig
│     (your custom additions)          │  (domain guidance, grading strictness,
│                                      │   normalization rules)
└──────────────────────────────────────┘
                   │
                   ▼
          list[Message] → LLM call
```

Each layer is **append-only**: adapter text is appended to the task text, then user text is appended after that. Nothing is removed or replaced. User instructions get the final position, giving them the last word.

### Layer 1: Task Instructions

The base system and user prompts for a specific pipeline operation. For example, when parsing a response into a template schema, `TemplatePromptBuilder` generates:

- A **system prompt** with extraction protocol, critical rules, and optionally tool-trace verification or ground-truth reference sections
- A **user prompt** with the original question and the response to parse

Each pipeline call type has its own prompt builder or prompt constants. These are internal to the pipeline; you do not modify them directly.

### Layer 2: Adapter Instructions

Backend-specific prompt modifications registered in the `AdapterInstructionRegistry`. Different adapters need different prompt content because they have different parsing mechanisms:

| Adapter | What its instructions add | Why |
|---------|---------------------------|-----|
| **LangChain** | JSON schema block, format instructions, parsing notes, response trailer | No native structured output; the model needs explicit schema and formatting guidance in the prompt |
| **Claude Tool** | Extraction rules, output format directive, JSON schema block | Uses native Anthropic tool use for structure, but still includes schema for compatible endpoints |
| **Claude Agent SDK** | Minimal best-interpretation directive | Relies on native structured output; minimal prompt additions needed |
| **Manual** | *(none)* | No LLM calls; human provides responses directly |

Adapter instructions are looked up by `(interface, task)` pair. Interfaces that route through another adapter share its instruction factories: `openrouter` and `openai_endpoint` both use the LangChain instructions.

### Layer 3: User Instructions

Optional custom text you inject via `PromptConfig` on `VerificationConfig`. This is the primary way to customize prompt behavior:

```python
from karenina.schemas.verification import PromptConfig

prompt_config = PromptConfig(
    parsing="Focus on gene symbols. Normalize all gene names to HGNC format.",
    generation="You are a genomics expert. Answer concisely.",
    rubric_evaluation="Grade strictly. Deduct for missing citations.",
)
print(f"PromptConfig fields: {list(PromptConfig.model_fields.keys())}")
print(f"Generation instruction: {prompt_config.generation}")
```

In a real workflow, you pass `PromptConfig` to `VerificationConfig`:

```python tags=["hide-input"]
# VerificationConfig requires parsing_models, so a realistic construction looks like:
#
# config = VerificationConfig(
#     parsing_models=[ModelConfig(id="judge", model_provider="anthropic", model_name="claude-haiku-4-5")],
#     prompt_config=PromptConfig(
#         parsing="Focus on gene symbols. Normalize all gene names to HGNC format.",
#         generation="You are a genomics expert. Answer concisely.",
#     ),
# )
```

User instructions are appended to the **system text** after adapter instructions, giving them the final position in the prompt.

## 4. How Assembly Works

The `PromptAssembler` is a dataclass that performs the assembly. Each pipeline evaluator creates one per LLM call:

```
Input: system_text, user_text (from pipeline stage)
  │
  ├─ 1. Look up adapter instruction factories by (interface, task)
  ├─ 2. Call each factory with instruction_context → AdapterInstruction
  ├─ 3. Append each instruction's system_addition and user_addition
  ├─ 4. Append user instructions (from PromptConfig) to system_text
  │
  └─ Output: list[Message] ready for LLM invocation
```

If the adapter does not support system prompts (reported via `PortCapabilities.supports_system_prompt == False`), the assembler prepends the system text to the user text as a single user message. Nothing is lost; the content is combined rather than discarded.

The assembler provides two methods:

| Method | Returns | Used by |
|--------|---------|---------|
| `assemble()` | `list[Message]` | Most pipeline stages (standard LLM calls) |
| `assemble_text()` | `tuple[str, str]` | Multi-stage flows like deep judgment that need intermediate text processing before building messages |

### Worked Example: Assembly Without Adapter Instructions

The simplest case: assembling a prompt when no adapter instructions are registered for the `(interface, task)` pair. This shows the core mechanics of the assembler in isolation.

```python
from karenina.benchmark.verification.prompts import PromptAssembler, PromptTask
from karenina.ports.capabilities import PortCapabilities

# Create an assembler for a parsing call with a fictional interface
assembler = PromptAssembler(
    task=PromptTask.PARSING,
    interface="demo_interface",  # no registered adapter instructions
    capabilities=PortCapabilities(supports_system_prompt=True),
)

# Assemble with task instructions and user instructions (no adapter layer)
messages = assembler.assemble(
    system_text="You are an evaluator that extracts structured information.",
    user_text="Parse this response: BCL2 is the primary target of venetoclax.",
    user_instructions="Normalize all gene names to HGNC format.",
)

for msg in messages:
    text = msg.content[0].text if hasattr(msg.content[0], "text") else str(msg.content)
    preview = text[:80].replace("\n", " ")
    print(f"[{msg.role.value}] {preview}...")
```

The `instruction_context` dict is passed through to adapter instruction factories. Each factory extracts what it needs: the LangChain parsing instruction uses `json_schema` and `format_instructions`; the Claude Tool instruction uses `json_schema`; others may ignore context entirely. Factories use `kwargs.get()` with defaults so they work even when keys are absent.

### System Prompt Fallback

When the adapter does not support system prompts, the assembler combines both texts into a single user message:

```python
# Adapter that does not support system prompts
no_sys_assembler = PromptAssembler(
    task=PromptTask.PARSING,
    interface="demo_interface",
    capabilities=PortCapabilities(supports_system_prompt=False),
)

messages = no_sys_assembler.assemble(
    system_text="You are an evaluator.",
    user_text="Parse this response.",
)

print(f"Number of messages: {len(messages)}")
print(f"Role: {messages[0].role.value}")
text = messages[0].content[0].text if hasattr(messages[0].content[0], "text") else str(messages[0].content)
print(f"Content:\n{text}")
```

### How the Pipeline Uses the Assembler

In practice, the pipeline stages build task-specific prompts, then hand them to the assembler. Here is the pattern used by the template parsing evaluator (you do not write this code; the pipeline runs it automatically):

```python
# --- Inside the template evaluator (you don't write this) ---
#
# 1. Task instructions (built by TemplatePromptBuilder)
# system_text = builder.build_system_prompt(has_tool_traces=False)
# user_text = builder.build_user_prompt(
#     question_text="Which is the putative target of venetoclax?",
#     response_to_parse="BCL2 is the primary pharmacological target...",
# )
#
# 2. Resolve user instructions from PromptConfig
# user_instructions = (
#     prompt_config.get_for_task(PromptTask.PARSING.value)
#     if prompt_config else None
# )
#
# 3. Assemble all three sections
# assembler = PromptAssembler(
#     task=PromptTask.PARSING,
#     interface=model_config.interface,        # e.g., "claude_tool"
#     capabilities=parser.capabilities,         # e.g., supports_system_prompt=True
# )
# messages = assembler.assemble(
#     system_text=system_text,
#     user_text=user_text,
#     user_instructions=user_instructions,
#     instruction_context={
#         "json_schema": answer_class.model_json_schema(),
#         "format_instructions": "",
#     },
# )
#
# 4. Parser adapter receives pre-assembled messages
# result = parser.parse_to_pydantic(messages, answer_class)
```

## 5. Pipeline LLM Calls

Every distinct LLM call in the pipeline is identified by a `PromptTask` enum value. This value determines which adapter instructions are looked up and which `PromptConfig` field provides user instructions.

```python
from karenina.benchmark.verification.prompts.task_types import PromptTask

# All 17 prompt task values, grouped by category
for task in PromptTask:
    print(f"  {task.value:<40} ({task.name})")
```

The full mapping from task to pipeline stage:

| Category | Task | Pipeline Stage | Description |
|----------|------|---------------|-------------|
| **Generation** | `generation` | GenerateAnswer | Answering model generates a response |
| **Parsing** | `parsing` | ParseTemplate | Judge LLM parses response into template schema |
| **Trace checks** | `abstention_detection` | AbstentionCheck | Detects model refusal or evasion |
| | `sufficiency_detection` | SufficiencyCheck | Checks response completeness for template |
| **Rubric evaluation** | `rubric_llm_trait_batch` | RubricEvaluation | All boolean/score LLM traits in one call |
| | `rubric_llm_trait_single` | RubricEvaluation | One boolean/score LLM trait per call |
| | `rubric_literal_trait_batch` | RubricEvaluation | All literal traits in one call |
| | `rubric_literal_trait_single` | RubricEvaluation | One literal trait per call |
| | `rubric_metric_trait` | RubricEvaluation | Metric trait confusion matrix extraction |
| **Deep judgment (template)** | `dj_template_excerpt_extraction` | DeepJudgmentAutoFail | Extract verbatim excerpts per template attribute |
| | `dj_template_hallucination` | DeepJudgmentAutoFail | Assess hallucination risk via search |
| | `dj_template_reasoning` | DeepJudgmentAutoFail | Map excerpts to template attributes |
| **Deep judgment (rubric)** | `dj_rubric_excerpt_extraction` | DeepJudgmentRubric | Extract excerpts for rubric traits |
| | `dj_rubric_hallucination` | DeepJudgmentRubric | Assess per-excerpt hallucination risk |
| | `dj_rubric_reasoning` | DeepJudgmentRubric | Generate trait evaluation reasoning |
| | `dj_rubric_score_extraction` | DeepJudgmentRubric | Extract final score from reasoning |

17 distinct LLM call types. Each one can have its own adapter instructions and user instructions.

## 6. Customizing Prompts via PromptConfig

`PromptConfig` is the user-facing customization point. Each field maps to one or more `PromptTask` values:

| Field | Applies to | Use when |
|-------|-----------|----------|
| `generation` | `generation` | Customize how the answering model responds (domain persona, response format, constraints) |
| `parsing` | `parsing` | Guide the judge LLM's extraction behavior (normalization rules, disambiguation, strictness) |
| `abstention_detection` | `abstention_detection` | Adjust abstention sensitivity (domain-specific refusal patterns) |
| `sufficiency_detection` | `sufficiency_detection` | Adjust sufficiency thresholds for your template |
| `rubric_evaluation` | All `rubric_*` tasks | Shared instructions for all rubric trait evaluations |
| `deep_judgment` | All `dj_*` tasks | Shared instructions for all deep judgment stages |

### Fallback Logic

`PromptConfig` uses a two-level resolution strategy:

1. **Direct match**: if a field name matches the task value exactly (e.g., `parsing` for `PromptTask.PARSING`), use it
2. **Category fallback**: if no direct match, `rubric_*` tasks fall back to `rubric_evaluation`; `dj_*` tasks fall back to `deep_judgment`

This means you can set `rubric_evaluation` once to affect all rubric LLM calls, or set individual task fields for fine-grained control. Direct matches take priority over fallbacks.

```python
from karenina.schemas.verification import PromptConfig

pc = PromptConfig(
    parsing="Normalize gene names to HGNC.",
    rubric_evaluation="Grade strictly.",
)

# Direct match
print(f"parsing:                  {pc.get_for_task('parsing')}")

# Category fallback: rubric_llm_trait_batch has no direct field,
# so it falls back to rubric_evaluation
print(f"rubric_llm_trait_batch:   {pc.get_for_task('rubric_llm_trait_batch')}")
print(f"rubric_metric_trait:      {pc.get_for_task('rubric_metric_trait')}")

# No match and no fallback
print(f"generation (not set):     {pc.get_for_task('generation')}")
```

### What User Instructions Can and Cannot Do

| Can do | Cannot do |
|--------|-----------|
| Append guidance to system prompts | Remove or replace task instructions |
| Influence LLM behavior via natural-language directives | Change which model is used or which adapter is selected |
| Add domain-specific context or normalization rules | Modify the template schema or rubric trait definitions |
| Adjust grading strictness or evaluation criteria | Skip pipeline stages or change execution order |

User instructions are a powerful, safe customization point. They modify *how* the LLM interprets its task without changing *what* the task is.

## 7. Debugging

Enable debug logging on `karenina.benchmark.verification.prompts` to see the assembled prompts for each LLM call. This shows the full text after all three layers have been applied:

```python
import logging
logging.getLogger("karenina.benchmark.verification.prompts").setLevel(logging.DEBUG)
print("Debug logging enabled for prompt assembly.")
```

When a pipeline result seems wrong, check the assembled prompt first. Common issues:

- **Adapter instructions not appearing**: the adapter's prompt module was not imported (check `registration.py` imports)
- **User instructions ignored**: `prompt_config` is `None` on `VerificationConfig`, or the field name does not match the task value
- **System text merged into user text**: the adapter's `PortCapabilities.supports_system_prompt` is `False`, so system text was prepended to user text (expected behavior, not a bug)

```python tags=["hide-cell"]
# Cleanup mock cell
_ = _patcher.stop()
```

## 8. Next Steps

- [Verification Pipeline](../verification-pipeline/): the 13 stages that use assembled prompts
- [Adapters](../../../core_concepts/adapters/): which LLM backends are available and how they differ
- [Prompt Assembly Internals](../../../advanced-pipeline/prompt-assembly/): technical deep dive into `PromptAssembler`, `AdapterInstructionRegistry`, `PromptTask`, and `PortCapabilities`
- [Prompt Config Reference](../../../reference/configuration/prompt-config/): full field documentation for `PromptConfig`
