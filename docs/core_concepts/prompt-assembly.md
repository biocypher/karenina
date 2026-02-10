# Prompt Assembly

Every LLM call in the verification pipeline — parsing responses, evaluating rubric traits, checking sufficiency — uses prompts constructed by the **PromptAssembler**. Understanding how prompts are built helps you customize evaluation behavior and debug unexpected results.

## The Tri-Section Pattern

All pipeline prompts follow a three-layer composition pattern:

```
┌──────────────────────────────────────┐
│  1. Task Instructions                │  ← Built by the pipeline stage
│     (what the LLM should do)         │
├──────────────────────────────────────┤
│  2. Adapter Instructions             │  ← Registered per (interface, task)
│     (backend-specific adjustments)   │
├──────────────────────────────────────┤
│  3. User Instructions                │  ← Optional overrides from PromptConfig
│     (your custom additions)          │
└──────────────────────────────────────┘
                   │
                   ▼
          list[Message] → LLM call
```

### 1. Task Instructions

The **base prompt** for each pipeline operation. For example, when parsing a response into a template schema, the `TemplatePromptBuilder` generates instructions that include:

- The task description ("Parse the following response into the given schema")
- The template's JSON schema
- The question text and response to parse
- Formatting guidelines

Each pipeline stage that involves an LLM call has its own instruction builder. These are internal — you do not modify them directly.

### 2. Adapter Instructions

**Backend-specific prompt adjustments** registered via the `AdapterInstructionRegistry`. Different LLM backends have different capabilities, and the prompt needs to account for this.

For example:

- The **Claude Tool** adapter has native structured output, so the adapter instructions strip the JSON schema from the prompt (it is already provided via the tool definition)
- The **LangChain** adapter uses a manual JSON parsing fallback, so its instructions include explicit JSON formatting guidance

Adapter instructions are looked up by `(interface, task)` pair. Each adapter registers its own instruction factories at import time.

### 3. User Instructions

**Optional custom text** you can inject via `PromptConfig` on `VerificationConfig`. This is the primary way to customize prompt behavior without modifying pipeline code.

```python
from karenina.schemas.verification import VerificationConfig, PromptConfig

config = VerificationConfig(
    prompt_config=PromptConfig(
        parsing_system_prompt_addition="Always prefer the most specific interpretation.",
        answering_system_prompt="You are a genomics expert. Answer concisely.",
    ),
    ...
)
```

User instructions are appended after adapter instructions, giving them the final say.

## How Assembly Works

The `PromptAssembler` receives raw system and user text from the pipeline stage, then applies instructions in order:

```
Input: system_text, user_text (from pipeline stage)
  │
  ├─ 1. Look up adapter instructions by (interface, task)
  ├─ 2. Append adapter additions to system_text and user_text
  ├─ 3. Append user instructions to system_text
  │
  └─ Output: list[Message] ready for LLM invocation
```

If the adapter does not support system prompts (reported via `PortCapabilities`), the system text is prepended to the user text so nothing is lost.

## What This Means for Users

- **You can inject custom instructions** via `PromptConfig` without touching pipeline internals
- **Adapter differences are handled automatically** — the same pipeline code works across all backends
- **Parser adapters are pure executors** — they receive pre-assembled `list[Message]` and do not build prompts internally
- **Debugging prompts**: Enable debug logging on `karenina.benchmark.verification.prompts` to see the assembled prompts

## Next Steps

- [Verification Pipeline](verification-pipeline.md) — The 13 stages that use assembled prompts
- [Adapters](adapters.md) — Which LLM backends are available
- [Prompt Assembly Internals](../11-advanced-pipeline/prompt-assembly.md) — Technical deep dive into `PromptAssembler`, `AdapterInstructionRegistry`, and prompt tasks
