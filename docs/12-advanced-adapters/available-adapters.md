# Available Adapters

Karenina ships with four adapter implementations and two routing interfaces, totaling six `interface` values. This page documents each adapter's implementation details, capabilities, configuration requirements, and adapter-specific behavior.

For the conceptual introduction to adapters, see [Adapters Overview](../04-core-concepts/adapters.md). For port protocol signatures, see [Port Types](ports.md).

---

## Feature Comparison

| Feature | `langchain` | `claude_agent_sdk` | `claude_tool` | `manual` |
|---------|:-----------:|:------------------:|:-------------:|:--------:|
| **Multi-provider support** | Yes (all LangChain providers) | No (Anthropic only) | No (Anthropic only) | N/A |
| **MCP server support** | Yes | Yes | Yes | No |
| **Tool use** | Yes | Yes | Yes | No |
| **Parser: structured output** | No (JSON fallback) | Yes (native) | Yes (native) | No |
| **Parser: system prompt** | Yes | Yes | Yes | No |
| **Prompt caching** | Via middleware config | Via SDK | Native API support | No |
| **Auto-fallback** | None (base adapter) | `langchain` | `langchain` | None |
| **Availability check** | `langchain_core` importable | `claude` CLI in PATH | `anthropic` importable | Always available |

---

## `langchain` — Multi-Provider Default

The default adapter, supporting all LLM providers available through LangChain: Anthropic, OpenAI, Google, Cohere, and many more.

### Implementation Classes

| Port | Class | Module |
|------|-------|--------|
| AgentPort | `LangChainAgentAdapter` | `adapters.langchain.agent` |
| ParserPort | `LangChainParserAdapter` | `adapters.langchain.parser` |
| LLMPort | `LangChainLLMAdapter` | `adapters.langchain.llm` |

### Configuration

```python
from karenina.schemas.config import ModelConfig

config = ModelConfig(
    id="gpt4",
    model_name="gpt-4o",
    model_provider="openai",  # Required for langchain
    interface="langchain",
)
```

**Required fields**: `id`, `model_name`, `model_provider`

**Environment variables**: Provider-specific API key (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`)

### Parser Behavior

The LangChain parser does **not** use native structured output (`supports_structured_output = False`). Instead, it uses a two-stage approach:

1. **Primary**: Calls the LLM with JSON formatting instructions in the prompt, then parses the response as JSON
2. **Fallback**: If JSON parsing fails, applies `json-repair` to fix common formatting issues

The parser implements retry logic with feedback:

- **Null-value feedback**: Detects required fields with `None` values and sends a correction prompt asking the LLM to fill them
- **Format feedback**: Detects JSON format errors and asks for clean JSON output

### Agent Behavior

The LangChain agent uses LangGraph for multi-turn agent execution. When MCP servers are configured, the agent supports:

- Configurable middleware via `AgentMiddlewareConfig` (summarization, prompt caching, retry)
- Tool execution with configurable limits (`model_call_limit`, `tool_call_limit`)
- Context summarization at a configurable threshold (default: 80% of context window)

### Adapter Instructions

The LangChain adapter registers prompt instructions for three task categories:

- **Parsing**: Adds JSON formatting instructions (since it lacks native structured output)
- **Rubric evaluation**: Adapter-specific rubric prompt tuning
- **Deep judgment**: Adapter-specific deep judgment prompt tuning

Instructions are registered for three interfaces: `langchain`, `openrouter`, and `openai_endpoint` (all share the same adapter).

---

## `openrouter` — OpenRouter API

A **routing interface** that delegates to the LangChain adapter. Provides access to 200+ models from various providers through a single API.

### How It Works

The `openrouter` interface uses the same adapter classes as `langchain`. At registration time, `routes_to = "langchain"` tells the registry that this interface resolves to LangChain.

### Configuration

```python
config = ModelConfig(
    id="openrouter-claude",
    model_name="anthropic/claude-sonnet-4-20250514",
    interface="openrouter",
    # model_provider is optional for openrouter
)
```

**Required fields**: `id`, `model_name`

**Environment variables**: `OPENROUTER_API_KEY`

### When to Use

- Access models from multiple providers through one API key
- Use models not directly available via other adapters
- Compare models across providers with consistent billing

---

## `openai_endpoint` — OpenAI-Compatible Endpoints

A **routing interface** that delegates to the LangChain adapter. Connects to any server implementing the OpenAI API specification.

### How It Works

Like `openrouter`, this interface uses the same LangChain adapter classes with `routes_to = "langchain"`. The custom base URL and API key are passed through to the underlying adapter.

### Configuration

```python
config = ModelConfig(
    id="local-llama",
    model_name="llama-3.1-70b",
    interface="openai_endpoint",
    endpoint_base_url="http://localhost:8000/v1",  # Required
    endpoint_api_key="not-needed",                  # Required
    # model_provider is optional for openai_endpoint
)
```

**Required fields**: `id`, `model_name`, `endpoint_base_url`, `endpoint_api_key`

Both `endpoint_base_url` and `endpoint_api_key` are validated at factory creation time — missing either raises `AdapterUnavailableError`.

### When to Use

- Local LLM servers: Ollama, LM Studio, vLLM, text-generation-inference
- Cloud providers with OpenAI-compatible APIs
- Self-hosted model deployments

---

## `claude_agent_sdk` — Native Anthropic Agent SDK

Direct integration with Anthropic's Agent SDK, providing native access to all Claude-specific features.

### Implementation Classes

| Port | Class | Module |
|------|-------|--------|
| AgentPort | `ClaudeSDKAgentAdapter` | `adapters.claude_agent_sdk.agent` |
| ParserPort | `ClaudeSDKParserAdapter` | `adapters.claude_agent_sdk.parser` |
| LLMPort | `ClaudeSDKLLMAdapter` | `adapters.claude_agent_sdk.llm` |

### Configuration

```python
config = ModelConfig(
    id="claude-sonnet",
    model_name="claude-sonnet-4-20250514",
    interface="claude_agent_sdk",
    # model_provider is optional (not needed for Claude SDK)
)
```

**Required fields**: `id`, `model_name`

**Availability check**: Looks for the `claude` CLI binary in PATH via `shutil.which("claude")`. If not found, falls back to `langchain`.

**Install**: `npm install -g @anthropic-ai/claude-code`

### Parser Behavior

The Claude SDK parser uses **native structured output** (`supports_structured_output = True`):

- Uses `query()` with `output_format={'type': 'json_schema', ...}` to get structured responses
- Returns a Python dict directly (not a JSON string), validated with `schema.model_validate(dict)`
- SDK handles retries autonomously via `max_turns` (minimum 2 for structured output)

### Agent Behavior

The agent adapter uses `ClaudeSDKClient` for full agent functionality:

- Native session management with multi-turn conversation support
- Built-in tool execution via the SDK
- MCP server integration (native support)
- Trace capture via `sdk_messages_to_raw_trace()` conversion

The LLM adapter uses a lighter-weight `query()` call for simple single-turn invocations.

### Adapter Instructions

Registers prompt instructions for `claude_agent_sdk` interface covering parsing, rubric, and deep judgment tasks.

---

## `claude_tool` — Anthropic Python SDK with Tool Runner

A lighter-weight Anthropic integration that uses the `anthropic` Python SDK directly, with `tool_runner` for agentic workflows.

### Implementation Classes

| Port | Class | Module |
|------|-------|--------|
| AgentPort | `ClaudeToolAgentAdapter` | `adapters.claude_tool.agent` |
| ParserPort | `ClaudeToolParserAdapter` | `adapters.claude_tool.parser` |
| LLMPort | `ClaudeToolLLMAdapter` | `adapters.claude_tool.llm` |

### Configuration

```python
config = ModelConfig(
    id="claude-tool",
    model_name="claude-sonnet-4-20250514",
    interface="claude_tool",
    # model_provider is optional
    # anthropic_base_url and anthropic_api_key are optional
    # (defaults to ANTHROPIC_API_KEY env var)
)
```

**Required fields**: `id`, `model_name`

**Availability check**: Checks that the `anthropic` Python package is importable. Also checks for `mcp` package (optional — needed only for MCP server support). Falls back to `langchain` if `anthropic` is missing.

**Install**: `pip install anthropic`

### Parser Behavior

The Claude Tool parser uses **native structured output** (`supports_structured_output = True`):

- Uses `client.beta.messages.parse()` with a Pydantic schema for type-safe parsing
- Returns a parsed Pydantic instance directly — no manual JSON extraction needed
- Anthropic SDK handles transient error retries natively

### Agent Behavior

The agent adapter uses `tool_runner` for automatic agent loops:

- Executes tool calls automatically until the model produces a final response
- MCP server support via HTTP/SSE transport (session-per-run semantics)
- Trace capture via `claude_tool_messages_to_raw_trace()` conversion

### Prompt Caching

The Claude Tool adapter has built-in support for Anthropic's prompt caching:

- Uses `cache_control: "ephemeral"` on API messages
- Default cache TTL: 5 minutes (1 hour available)
- Reduces costs and latency for repeated prompts with shared prefixes

### Adapter Instructions

Registers prompt instructions for `claude_tool` interface. Notably, the parsing instructions **strip JSON schema formatting** from prompts because Claude's native structured output handles schema enforcement automatically.

---

## `manual` — Pre-Recorded Traces

A special-purpose interface that replays pre-recorded LLM traces instead of making live API calls.

### Implementation Classes

| Port | Class | Module |
|------|-------|--------|
| AgentPort | `ManualAgentAdapter` | `adapters.manual` |
| ParserPort | `ManualParserAdapter` | `adapters.manual` |
| LLMPort | `ManualLLMAdapter` | `adapters.manual` |

### Configuration

```python
config = ModelConfig(
    id="manual",
    model_name="manual",
    interface="manual",
    manual_traces=manual_traces,  # ManualTraces object with pre-recorded data
)
```

**Required fields**: `manual_traces` (validated by ModelConfig model validator)

**Availability**: Always available — no external dependencies.

### How It Works

Only the **agent adapter** is functional:

- `ManualAgentAdapter.run()` looks up the pre-recorded trace for the given question hash and returns it as an `AgentResult`
- Returns zero token usage (`input_tokens=0, output_tokens=0`)
- Can include optional `agent_metrics` dict (iterations, limit_reached)

The LLM and parser adapters are intentional **no-ops**:

- `ManualLLMAdapter.invoke()` raises `ManualInterfaceError` with the message `"llm.invoke()"`
- `ManualParserAdapter.aparse_to_pydantic()` raises `ManualInterfaceError` with the message `"parser.aparse_to_pydantic()"`

This design is deliberate — in the manual workflow, the answering trace is pre-recorded (bypassing the agent), but parsing and verification still run live with a different model configuration.

### Capabilities

No system prompt support, no structured output, no MCP, no tools. The manual adapter's `capabilities` property returns `PortCapabilities(supports_system_prompt=False, supports_structured_output=False)`.

### When to Use

- **Testing and CI**: Run verification without API calls or costs
- **Reproducibility**: Evaluate the same traces repeatedly with different configurations
- **External traces**: Evaluate responses from models or systems outside karenina
- **Development**: Iterate on templates and rubrics without waiting for LLM responses

See [Manual Interface](../04-core-concepts/manual-interface.md) for trace format and workflow details.

---

## Choosing an Adapter

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Getting started | `langchain` | Default, broadest provider support |
| Multi-provider evaluation | `langchain` | Supports OpenAI, Anthropic, Google, and more |
| Single API key for many models | `openrouter` | 200+ models, one API key |
| Local LLM server | `openai_endpoint` | Ollama, vLLM, LM Studio, etc. |
| Claude-only with full features | `claude_agent_sdk` | Native structured output, session management |
| Claude-only, lightweight | `claude_tool` | Native structured output, prompt caching, simpler setup |
| Offline / CI / reproducibility | `manual` | No API calls, pre-recorded traces |
| Need native structured output | `claude_agent_sdk` or `claude_tool` | Both have `supports_structured_output = True` |

---

## Provider Requirements Summary

| Interface | `model_provider` Required | API Key Environment Variable |
|-----------|:-------------------------:|------------------------------|
| `langchain` | Yes | Provider-specific: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` |
| `openrouter` | No | `OPENROUTER_API_KEY` |
| `openai_endpoint` | No | Set via `endpoint_api_key` on ModelConfig |
| `claude_agent_sdk` | No | `ANTHROPIC_API_KEY` (used by Claude CLI) |
| `claude_tool` | No | `ANTHROPIC_API_KEY` (or `anthropic_api_key` on ModelConfig) |
| `manual` | No | None |

---

## Related

- [Adapter Architecture](index.md) — Hexagonal architecture, registry, and factory system
- [Port Types](ports.md) — Complete protocol signatures for LLMPort, ParserPort, AgentPort
- [MCP Integration](mcp-integration.md) — How adapters handle MCP servers and tool schemas
- [Writing Custom Adapters](writing-adapters.md) — Guide to implementing and registering new adapters
- [Adapters Overview](../04-core-concepts/adapters.md) — Conceptual introduction to the adapter system
