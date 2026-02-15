# Prompt Assembly System

Every LLM call in the verification pipeline uses a **tri-section prompt pattern**. Three independent instruction sources are combined into the final prompt messages:

1. **Task instructions** --- the base system and user text for a specific pipeline call (e.g., parsing, abstention detection, rubric evaluation)
2. **Adapter instructions** --- text appended based on the LLM backend (e.g., LangChain appends JSON schema formatting; Claude Tool appends extraction directives)
3. **User instructions** --- optional per-task text from `PromptConfig`

This separation keeps each concern isolated: task prompt builders don't know about adapters, adapters don't know about user overrides, and user instructions don't depend on either.

## How It Works

```
Task Instructions          Adapter Instructions        User Instructions
(TemplatePromptBuilder,    (AdapterInstruction-        (PromptConfig)
 AbstentionPrompt, etc.)    Registry)

     system_text ──┐
                   ├──► + adapter.system_addition ──► + user_instructions ──► Final system text
     user_text ────┤
                   └──► + adapter.user_addition ──────────────────────────► Final user text
```

All instruction sources are **append-only**: adapter text is appended first, then user instructions. The final texts are either wrapped in `Message` objects or returned as raw strings depending on the call site.

## PromptAssembler

`PromptAssembler` is the single entry point that combines all three instruction sections. It is a dataclass with three fields:

| Field | Type | Description |
|-------|------|-------------|
| `task` | `PromptTask` | Identifies which pipeline LLM call this is for |
| `interface` | `str` | The adapter interface name (e.g., `"langchain"`, `"claude_tool"`) |
| `capabilities` | `PortCapabilities` | The adapter's declared capabilities |

### Methods

**`assemble(system_text, user_text, user_instructions, instruction_context)`** --- Returns `list[Message]`

The primary method. Applies adapter and user instructions, then builds `Message` objects. If the adapter does not support system prompts (`capabilities.supports_system_prompt == False`), the system text is prepended to the user text as a single user message.

**`assemble_text(system_text, user_text, user_instructions, instruction_context)`** --- Returns `tuple[str, str]`

Same tri-section logic but returns raw `(system_text, user_text)` strings instead of `Message` objects. Used by multi-stage flows (e.g., deep judgment) that need intermediate text processing before final message construction.

### Assembly Order

Both methods follow the same internal sequence:

1. **Look up adapter instructions** from the `AdapterInstructionRegistry` for the `(interface, task)` pair
2. **Append adapter additions** --- each registered factory produces an `AdapterInstruction` with `system_addition` and `user_addition` properties; non-empty additions are appended to their respective texts
3. **Append user instructions** --- if `user_instructions` is provided (from `PromptConfig`), it is appended to the system text
4. **Build messages** (for `assemble()` only) --- wrap the final texts in `Message.system()` and `Message.user()`, or combine into a single `Message.user()` if system prompts are not supported

### Usage Example

This is how the template parsing evaluator uses `PromptAssembler`:

```python
from karenina.benchmark.verification.prompts.assembler import PromptAssembler
from karenina.benchmark.verification.prompts.task_types import PromptTask

# 1. Build task-specific prompts
builder = TemplatePromptBuilder(answer_class=answer_class)
system_prompt = builder.build_system_prompt(has_tool_traces=has_tools)
user_prompt = builder.build_user_prompt(
    question_text=question_text,
    response_to_parse=raw_response,
)

# 2. Resolve user instructions from PromptConfig
user_instructions = (
    prompt_config.get_for_task(PromptTask.PARSING.value)
    if prompt_config else None
)

# 3. Assemble all three sections
assembler = PromptAssembler(
    task=PromptTask.PARSING,
    interface=model_config.interface,
    capabilities=parser.capabilities,
)
messages = assembler.assemble(
    system_text=system_prompt,
    user_text=user_prompt,
    user_instructions=user_instructions,
    instruction_context={"json_schema": schema, "format_instructions": fmt},
)
```

The `instruction_context` dict is passed to adapter instruction factories. Each factory extracts the parameters it needs (e.g., the LangChain parsing instruction uses `json_schema` and `format_instructions`; Claude Tool ignores them).

## AdapterInstructionRegistry

The registry is a class-level mapping from `(interface, task)` pairs to lists of instruction factories. It provides a global, shared mechanism for adapters to inject prompt modifications without coupling to specific pipeline stages.

### API

| Method | Description |
|--------|-------------|
| `register(interface, task, factory)` | Register a factory for an `(interface, task)` pair |
| `get_instructions(interface, task)` | Retrieve factories for a pair (empty list if none) |
| `clear()` | Clear all registrations (for testing) |

### AdapterInstruction Protocol

Each instruction factory must return an object implementing the `AdapterInstruction` protocol:

```python
class AdapterInstruction(Protocol):
    @property
    def system_addition(self) -> str:
        """Text to append to the system prompt (empty string for no addition)."""
        ...

    @property
    def user_addition(self) -> str:
        """Text to append to the user prompt (empty string for no addition)."""
        ...
```

### Instruction Registration

Adapters register their instructions in `adapters/<name>/prompts/*.py` files, which are imported at the bottom of each adapter's `registration.py`. This ensures instructions are registered when the adapter is loaded.

Example from the LangChain adapter (parsing):

```python
from karenina.ports.adapter_instruction import AdapterInstructionRegistry

def _langchain_format_instruction_factory(**kwargs):
    return _LangChainFormatInstruction(
        json_schema=kwargs.get("json_schema"),
        format_instructions=kwargs.get("format_instructions", ""),
    )

AdapterInstructionRegistry.register(
    "langchain", "parsing", _langchain_format_instruction_factory
)
# Also register for interfaces that route through LangChain
AdapterInstructionRegistry.register(
    "openrouter", "parsing", _langchain_format_instruction_factory
)
AdapterInstructionRegistry.register(
    "openai_endpoint", "parsing", _langchain_format_instruction_factory
)
```

### Registered Adapter Instructions

The following table shows all registered `(interface, task)` pairs across the codebase:

| Interface | Task Categories | What It Adds |
|-----------|----------------|--------------|
| `langchain` | parsing, rubric (`*_batch`, `*_single`, `metric`), deep judgment (`dj_*`) | JSON schema, format instructions, parsing notes |
| `openrouter` | parsing, rubric, deep judgment | Same as `langchain` (shared factories) |
| `openai_endpoint` | parsing, rubric, deep judgment | Same as `langchain` (shared factories) |
| `claude_tool` | parsing, rubric, deep judgment | Minimal extraction directives (native structured output) |
| `claude_agent_sdk` | parsing, rubric, deep judgment | Minimal best-interpretation directive (native structured output) |
| `manual` | *(none)* | No registered instructions |

The key difference: LangChain-based adapters need explicit JSON schema and format instructions because they lack native structured output. Claude-based adapters (Claude Tool and Claude Agent SDK) use native structured output, so their instructions are minimal --- just extraction or interpretation directives.

## PromptTask Values

Each `PromptTask` enum value identifies a distinct LLM call in the pipeline. The task value is used to look up both adapter instructions (via the registry) and user instructions (via `PromptConfig.get_for_task()`).

| Task | Pipeline Stage | Description |
|------|---------------|-------------|
| `generation` | GenerateAnswer | LLM generates a response to the question |
| `parsing` | ParseTemplate | Judge LLM parses response into template schema |
| `abstention_detection` | AbstentionCheck | Detects model refusal |
| `sufficiency_detection` | SufficiencyCheck | Checks response completeness |
| `rubric_llm_trait_batch` | RubricEvaluation | Batched boolean/score LLM traits |
| `rubric_llm_trait_single` | RubricEvaluation | Sequential single LLM trait |
| `rubric_literal_trait_batch` | RubricEvaluation | Batched literal (categorical) traits |
| `rubric_literal_trait_single` | RubricEvaluation | Sequential single literal trait |
| `rubric_metric_trait` | RubricEvaluation | Metric trait (confusion matrix) |
| `dj_template_excerpt_extraction` | DeepJudgmentAutoFail | Extract verbatim excerpts per attribute |
| `dj_template_hallucination` | DeepJudgmentAutoFail | Assess hallucination risk via search |
| `dj_template_reasoning` | DeepJudgmentAutoFail | Generate reasoning for excerpt-to-attribute mapping |
| `dj_rubric_excerpt_extraction` | DeepJudgmentRubricAutoFail | Extract excerpts for rubric traits |
| `dj_rubric_hallucination` | DeepJudgmentRubricAutoFail | Assess per-excerpt hallucination risk |
| `dj_rubric_reasoning` | DeepJudgmentRubricAutoFail | Generate trait evaluation reasoning |
| `dj_rubric_score_extraction` | DeepJudgmentRubricAutoFail | Extract final score from reasoning |

## PortCapabilities

`PortCapabilities` declares what prompt features an adapter supports. The assembler uses these to decide message formatting:

| Field | Type | Default | Effect |
|-------|------|---------|--------|
| `supports_system_prompt` | `bool` | `True` | If `False`, system text is prepended to user text as a single message |
| `supports_structured_output` | `bool` | `False` | Used by adapters to signal native structured output support |

## Customizing Prompts

### Via PromptConfig (User Instructions)

The most common customization point. Add instructions to `PromptConfig` fields to influence specific pipeline calls:

```python
from karenina.schemas.verification import VerificationConfig, PromptConfig

config = VerificationConfig(
    prompt_config=PromptConfig(
        parsing="Focus on gene symbols. Normalize all gene names to HGNC format.",
        rubric_evaluation="Grade strictly. Deduct points for missing citations.",
    ),
    # ...
)
```

User instructions are appended to the system text after adapter instructions. See [PromptConfig](../06-running-verification/prompt-config.md) for details on injection points and fallback logic.

### Via Adapter Instructions (For Adapter Authors)

To register custom instructions for a new adapter:

1. Create a dataclass implementing the `AdapterInstruction` protocol
2. Write a factory function that accepts `**kwargs` and returns the instruction instance
3. Register with `AdapterInstructionRegistry.register(interface, task, factory)`
4. Import the module from your adapter's `registration.py`

The factory receives the `instruction_context` dict passed to `PromptAssembler.assemble()`. Common context keys include:

| Key | Type | Provided By |
|-----|------|------------|
| `json_schema` | `dict` | Template parsing evaluator |
| `format_instructions` | `str` | Template parsing evaluator |

Factories should use `kwargs.get()` with defaults so they work even when keys are absent.

## Key Source Files

| File | Purpose |
|------|---------|
| `benchmark/verification/prompts/assembler.py` | `PromptAssembler` |
| `benchmark/verification/prompts/task_types.py` | `PromptTask` enum |
| `ports/adapter_instruction.py` | `AdapterInstructionRegistry`, `AdapterInstruction` protocol |
| `ports/capabilities.py` | `PortCapabilities` |
| `schemas/verification/prompt_config.py` | `PromptConfig` |
| `adapters/*/prompts/*.py` | Per-adapter instruction registrations |
| `benchmark/verification/prompts/parsing/parsing_instructions.py` | `TemplatePromptBuilder` |

## Next Steps

- [Prompt Config](../06-running-verification/prompt-config.md) --- configure user instructions per task
- [13 Stages in Detail](stages.md) --- which stages make LLM calls and use the assembler
- [Available Adapters](../12-advanced-adapters/available-adapters.md) --- adapter-specific prompt behavior
- [Verification Config Reference](../reference/configuration/verification-config.md) --- `prompt_config` field in `VerificationConfig`
- [Pipeline Overview](index.md) --- how stages execute and interact
