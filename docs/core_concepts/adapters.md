# Adapters

Adapters are LLM backend interfaces that handle communication with language models during verification. Karenina uses a **hexagonal architecture** (ports and adapters) where each adapter implements three port protocols — LLMPort, AgentPort, and ParserPort — providing a consistent API regardless of the underlying LLM provider.

## Available Interfaces

Karenina ships with six adapter interfaces:

| Interface | Backend | MCP Support | Tools | Structured Output (Parser) | Fallback |
|-----------|---------|:-----------:|:-----:|:--------------------------:|----------|
| `langchain` | LangChain (multi-provider) | Yes | Yes | No (manual JSON fallback) | None |
| `openrouter` | OpenRouter API | Yes | Yes | No (manual JSON fallback) | None |
| `openai_endpoint` | OpenAI-compatible endpoints | Yes | Yes | No (manual JSON fallback) | None |
| `claude_agent_sdk` | Anthropic Agent SDK | Yes | Yes | Yes (native) | `langchain` |
| `claude_tool` | Anthropic Python SDK | Yes | Yes | Yes (native) | `langchain` |
| `manual` | Pre-recorded traces | No | No | No | None |

---

## Interface Details

### `langchain` — Multi-Provider Default

The default adapter, supporting all LLM providers available through LangChain: Anthropic Claude, OpenAI GPT, Google Gemini, and many more.

**When to use**: You need to evaluate across multiple providers, or you want the broadest model support.

**Requirements**: `langchain_core` package (installed by default).

**Key features**:

- Structured output via `with_structured_output()` on the LLM side
- Parser uses manual JSON parsing fallback (no native structured output in parser)
- Retry logic with exponential backoff
- Usage metadata tracking

### `openrouter` — OpenRouter API

A **routing interface** that delegates to the LangChain adapter. Provides access to 200+ models from various providers through a single API.

**When to use**: You want access to many models through one API key, or need models not directly available via other adapters.

**Requirements**: `langchain_core` package, `OPENROUTER_API_KEY` environment variable.

**How it works**: Shares the same adapter implementations as `langchain` — the factory resolves `openrouter` to `langchain` at creation time.

### `openai_endpoint` — OpenAI-Compatible Endpoints

A **routing interface** that delegates to the LangChain adapter. Connects to any server implementing the OpenAI API specification.

**When to use**: You run a local LLM server (LM Studio, Ollama, vLLM, etc.) or use a cloud provider with an OpenAI-compatible API.

**Requirements**: `langchain_core` package, plus `endpoint_base_url` and `endpoint_api_key` on `ModelConfig`.

**How it works**: Same adapter implementations as `langchain`, with endpoint URL and API key configured per model.

### `claude_agent_sdk` — Native Anthropic Agent SDK

Direct integration with Anthropic's Agent SDK, providing native access to all Claude-specific features.

**When to use**: You work exclusively with Claude models and want native structured output, thinking support, and direct SDK access.

**Requirements**: Claude Code CLI (`claude` binary) installed and available in PATH. Falls back to `langchain` if unavailable.

**Key features**:

- Native structured output in both LLM and parser (no JSON parsing fallback needed)
- System prompt support in parser
- Access to Claude-specific features (extended thinking, vision)
- Direct Anthropic SDK usage for better type safety

### `claude_tool` — Anthropic SDK with Tool Runner

A lighter-weight Anthropic integration that uses the `anthropic` Python SDK with a `tool_runner` for agentic workflows.

**When to use**: You want native Claude integration without the full Agent SDK, or prefer a simpler tool execution model.

**Requirements**: `anthropic` Python package. Falls back to `langchain` if unavailable.

**Key features**:

- Native structured output in both LLM and parser
- System prompt support in parser
- Automatic tool execution loop via tool_runner
- Simpler agent implementation compared to `claude_agent_sdk`

### `manual` — Pre-Recorded Traces

A special-purpose interface that replays pre-recorded LLM traces instead of making live API calls.

**When to use**: Testing, CI/CD pipelines, reproducibility, or evaluating responses from external systems.

**Requirements**: Always available (no external dependencies).

**Key features**:

- No live LLM calls — returns stored traces by question hash
- Thread-safe session storage via `ManualTraceManager`
- Only the agent adapter is functional; LLM and parser adapters raise errors if invoked directly

See [Manual Interface](manual-interface.md) for details on trace format and usage.

---

## Choosing an Adapter

| Scenario | Recommended Interface |
|----------|----------------------|
| Evaluate across multiple LLM providers | `langchain` |
| Access 200+ models via single API | `openrouter` |
| Use a local LLM server (Ollama, vLLM, LM Studio) | `openai_endpoint` |
| Claude-only evaluation with full SDK features | `claude_agent_sdk` |
| Claude-only with lightweight tool support | `claude_tool` |
| Replay pre-recorded traces (testing, CI) | `manual` |
| Not sure / getting started | `langchain` (default) |

---

## Routing and Fallback

Two interfaces are **routing interfaces** that delegate to `langchain`:

- `openrouter` routes to `langchain` — same adapter code, different API endpoint
- `openai_endpoint` routes to `langchain` — same adapter code, custom base URL

Two interfaces have **automatic fallback** to `langchain`:

- `claude_agent_sdk` falls back if the Claude CLI is not installed
- `claude_tool` falls back if the `anthropic` package is not installed

Fallback is automatic when `auto_fallback=True` (the default) in the factory functions.

---

## Ports Architecture

Each adapter implements three port protocols defined in `karenina.ports`:

| Port | Purpose | Key Methods |
|------|---------|-------------|
| **LLMPort** | Chat/completion calls | `invoke()`, `ainvoke()`, `with_structured_output()` |
| **ParserPort** | Structured output parsing | `parse_to_pydantic()`, `aparse_to_pydantic()` |
| **AgentPort** | Tool-using agents | `run()`, `arun()` |

Adapters use **duck typing** for protocol compliance — they implement the required methods without explicitly inheriting from the protocol classes.

Each port exposes a `capabilities` property reporting what the adapter supports:

| Adapter | Parser: Structured Output | Parser: System Prompt |
|---------|:-------------------------:|:---------------------:|
| `langchain` | No | Yes |
| `claude_agent_sdk` | Yes | Yes |
| `claude_tool` | Yes | Yes |
| `manual` | No | No |

---

## Configuration

Adapters are configured through `ModelConfig`:

```python
from karenina.schemas.config import ModelConfig

# LangChain with OpenAI
config = ModelConfig(
    id="gpt4",
    model_name="gpt-4o",
    model_provider="openai",
    interface="langchain",
)

# Claude via Agent SDK
config = ModelConfig(
    id="claude",
    model_name="claude-sonnet-4-20250514",
    model_provider="anthropic",
    interface="claude_agent_sdk",
)

# OpenAI-compatible endpoint (local server)
config = ModelConfig(
    id="local",
    model_name="llama-3-70b",
    model_provider="openai",
    interface="openai_endpoint",
    endpoint_base_url="http://localhost:8000/v1",
    endpoint_api_key="not-needed",
)
```

The adapter factory functions create the appropriate adapter:

```python
from karenina.adapters.factory import get_llm, get_agent, get_parser

llm = get_llm(config)       # Returns LLMPort
agent = get_agent(config)    # Returns AgentPort
parser = get_parser(config)  # Returns ParserPort
```

---

## Next Steps

- [MCP Overview](mcp-overview.md) — Tool-augmented evaluation with MCP servers
- [Manual Interface](manual-interface.md) — Using pre-recorded traces
- [Evaluation Modes](evaluation-modes.md) — Controlling which evaluation units run
- [Running Verification](../06-running-verification/index.md) — End-to-end verification workflow
- [Advanced Adapters](../12-advanced-adapters/index.md) — Ports and adapters architecture deep dive, writing custom adapters
