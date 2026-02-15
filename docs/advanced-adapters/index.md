# Adapter Architecture

Karenina uses a **hexagonal architecture** (ports and adapters) to decouple its verification pipeline from any specific LLM provider. This design means the pipeline never calls a provider's API directly — instead, it talks to abstract port protocols, and adapters translate those calls to the appropriate backend.

---

## Why Hexagonal Architecture

The verification pipeline needs to support multiple LLM backends — LangChain for broad provider coverage, Anthropic's Agent SDK for native Claude features, direct Anthropic API for tool use, and even pre-recorded traces for offline evaluation. Without a clean abstraction, every pipeline stage would need provider-specific code paths, making the codebase fragile and hard to extend.

The hexagonal pattern solves this with three benefits:

**Separation of concerns** — Pipeline stages call port methods (`ainvoke`, `aparse_to_pydantic`, `arun`) without knowing which backend fulfills the request. Adding a new provider requires only writing adapter classes, not modifying pipeline logic.

**Testability** — Ports are Python `Protocol` classes. Any object with the right method signatures satisfies the protocol via duck typing. Tests can substitute lightweight mock adapters without importing real SDK dependencies.

**Graceful degradation** — When an adapter's required dependency is missing (e.g., Claude CLI not installed for `claude_agent_sdk`), the system automatically falls back to an available alternative (e.g., `langchain`). This happens at the factory level, transparent to the pipeline.

---

## Architecture Overview

```
                          ┌─────────────────────────────┐
                          │     Verification Pipeline    │
                          │                             │
                          │  Stage 2: generate_answer   │
                          │  Stage 7: parse_template    │
                          │  Stage 11: rubric_evaluation│
                          └─────────┬───────────────────┘
                                    │
                         calls port protocols
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
        ┌──────────┐         ┌───────────┐         ┌──────────┐
        │ AgentPort │         │ ParserPort│         │ LLMPort  │
        │  arun()   │         │aparse_to_ │         │ainvoke() │
        │  run()    │         │ pydantic()│         │invoke()  │
        └─────┬────┘         └─────┬─────┘         └────┬─────┘
              │                     │                     │
    ┌─────────┴────────┐    ┌──────┴───────┐    ┌────────┴───────┐
    │    Adapter        │    │   Adapter     │    │    Adapter     │
    │  Implementations  │    │Implementations│    │ Implementations│
    └─────────┬────────┘    └──────┬───────┘    └────────┬───────┘
              │                     │                     │
   ┌──────┬──┴──┬──────┐          ...                   ...
   │      │     │      │
   ▼      ▼     ▼      ▼
 Lang  Claude Claude Manual
 Chain  SDK   Tool
```

The pipeline interacts **only** with the three port protocols. The adapter factory (`get_agent`, `get_parser`, `get_llm`) selects the correct implementation based on the `interface` field in `ModelConfig`.

---

## The Three Ports

Each port protocol defines a specific type of LLM interaction:

| Port | Purpose | Key Method | Used By |
|------|---------|------------|---------|
| **AgentPort** | Multi-turn agent execution with tools and MCP servers | `arun()` / `run()` | Stage 2 (generate answer) |
| **ParserPort** | LLM-based structured output parsing (judge LLM fills a schema) | `aparse_to_pydantic()` / `parse_to_pydantic()` | Stage 7 (parse template) |
| **LLMPort** | Simple stateless LLM invocation | `ainvoke()` / `invoke()` | Stages 5–6 (quality checks), 11 (rubric evaluation), deep judgment |

All three follow an **async-first** design — the primary methods are `async` (`ainvoke`, `aparse_to_pydantic`, `arun`), with synchronous wrappers for convenience.

For complete protocol signatures and method documentation, see [Port Types](ports.md).

---

## Adapter Registry and Factory

### The Registry

The `AdapterRegistry` maps interface strings to `AdapterSpec` objects. Each spec declares:

- Factory functions for creating `AgentPort`, `ParserPort`, and `LLMPort` instances
- An availability checker (e.g., verifying that `langchain_core` is importable)
- A fallback interface to use when the primary isn't available
- Capability flags (`supports_mcp`, `supports_tools`)

Registration happens lazily — adapter modules are imported on first registry access, keeping startup lightweight.

### The Factory

Three factory functions provide the public API:

| Function | Returns | Purpose |
|----------|---------|---------|
| `get_agent(model_config)` | `AgentPort` | Create an agent adapter for answer generation |
| `get_parser(model_config)` | `ParserPort` | Create a parser adapter for template evaluation |
| `get_llm(model_config)` | `LLMPort` | Create an LLM adapter for invocation (rubric eval, quality checks) |

All three functions:

1. Read `model_config.interface` to determine which adapter to use
2. Validate the configuration (model name, provider if required)
3. Check adapter availability
4. Fall back automatically if the preferred adapter is unavailable (`auto_fallback=True` by default)
5. Return a port implementation — never `None`

```python
from karenina.adapters.factory import get_agent, get_parser, get_llm
from karenina.schemas.config import ModelConfig

config = ModelConfig(
    id="answering",
    model_name="claude-sonnet-4-20250514",
    interface="claude_agent_sdk"
)

agent = get_agent(config)   # Returns ClaudeSDKAgentAdapter (or LangChain fallback)
parser = get_parser(config) # Returns ClaudeSDKParserAdapter (or LangChain fallback)
llm = get_llm(config)       # Returns ClaudeSDKLLMAdapter (or LangChain fallback)
```

### Auto-Fallback

When an adapter's dependencies are missing, the factory transparently falls back:

```
claude_agent_sdk (Claude CLI not found) → falls back to langchain
claude_tool (anthropic package missing) → falls back to langchain
langchain (langchain_core not installed) → no fallback (raises error)
manual — always available, no fallback needed
```

---

## Supported Interfaces

| Interface | Backend | MCP | Tools | Structured Output | Fallback |
|-----------|---------|:---:|:-----:|:-----------------:|----------|
| `langchain` | LangChain (multi-provider) | Yes | Yes | No (JSON fallback) | None |
| `openrouter` | OpenRouter API (routes to langchain) | Yes | Yes | No (JSON fallback) | None |
| `openai_endpoint` | OpenAI-compatible (routes to langchain) | Yes | Yes | No (JSON fallback) | None |
| `claude_agent_sdk` | Anthropic Agent SDK | Yes | Yes | Yes (native) | `langchain` |
| `claude_tool` | Anthropic Python SDK | Yes | Yes | Yes (native) | `langchain` |
| `manual` | Pre-recorded traces | No | No | No | None |

`openrouter` and `openai_endpoint` are **routing interfaces** — they resolve to the `langchain` adapter at creation time, sharing the same implementation with interface-specific configuration (API keys, base URLs).

For per-adapter feature details and configuration, see [Available Adapters](available-adapters.md).

---

## Adapter Instructions

Adapters can customize the prompts sent to LLMs by registering **adapter instructions** via the `AdapterInstructionRegistry`. Each registration maps an `(interface, task)` pair to an instruction factory that provides text to append to the system and/or user prompt.

For example, the `claude_tool` adapter strips JSON schema instructions from parsing prompts because Claude's native structured output handles schema enforcement automatically. The `langchain` adapter adds JSON formatting instructions since it relies on manual parsing.

This mechanism keeps adapters as **pure executors** — they receive pre-assembled messages from the `PromptAssembler` and don't build prompts internally. Adapter-specific tuning is declarative, registered at import time.

For the complete prompt assembly system, see [Prompt Assembly](../11-advanced-pipeline/prompt-assembly.md).

---

## Duck Typing for Port Compliance

Adapters implement ports via **duck typing**, not class inheritance. The port protocols are defined as `typing.Protocol` classes — any object with matching method signatures satisfies the contract:

```python
# Port protocol (in karenina/ports/llm.py)
class LLMPort(Protocol):
    async def ainvoke(self, messages: list[Message]) -> LLMResponse: ...
    def invoke(self, messages: list[Message]) -> LLMResponse: ...

# Adapter (in karenina/adapters/langchain/llm.py)
class LangChainLLMAdapter:  # No inheritance from LLMPort
    async def ainvoke(self, messages: list[Message]) -> LLMResponse:
        # LangChain-specific implementation
        ...
```

This means you can create a new adapter without importing the port classes — just implement the right methods. Type checkers verify compliance statically.

---

## Unified Message Format

All ports use a shared `Message` class from `karenina.ports.messages`:

```python
from karenina.ports.messages import Message

# Create messages using factory methods
system = Message.system("You are a helpful evaluator.")
user = Message.user("Evaluate this response: ...")
```

Messages support structured content blocks — text, tool use, tool results, and thinking — providing a consistent format across all adapters regardless of the underlying provider's message schema.

---

## Section Contents

| Page | Description |
|------|-------------|
| [Port Types](ports.md) | Complete protocol signatures for LLMPort, ParserPort, and AgentPort |
| [Available Adapters](available-adapters.md) | Per-adapter features, configuration, and capabilities |
| [MCP Integration](mcp-integration.md) | How adapters handle MCP servers, tool schemas, and trace capture |
| [Writing Custom Adapters](writing-adapters.md) | Guide to implementing new adapters and registering them |

## Related

- [Adapters Overview](../core_concepts/adapters.md) — Conceptual introduction to the adapter system
- [Verification Pipeline](../11-advanced-pipeline/index.md) — How the pipeline uses ports during execution
- [Prompt Assembly](../11-advanced-pipeline/prompt-assembly.md) — The tri-section prompt system and adapter instructions
- [Verification Config Reference](../reference/configuration/verification-config.md) — ModelConfig fields that configure adapters
