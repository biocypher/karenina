# Available Adapters

Karenina ships with six adapter packages (`langchain`, `claude_agent_sdk`, `claude_tool`, `langchain_deep_agents`, `manual`, `taskeval`) and two routing interfaces (`openrouter`, `openai_endpoint`) that delegate to LangChain, totaling **eight registered `interface` values**. This page documents each adapter's implementation details, capabilities, configuration requirements, and adapter-specific behavior.

For the conceptual introduction to adapters, see [Adapters Overview](../core_concepts/adapters.md). For port protocol signatures, see [Port Types](ports.md).

---

## Feature Comparison

| Feature | `langchain` | `langchain_deep_agents` | `claude_agent_sdk` | `claude_tool` | `manual` | `taskeval` |
|---------|:-----------:|:-----------------------:|:------------------:|:-------------:|:--------:|:----------:|
| **Multi-provider support** | Yes (all LangChain providers) | Yes (all LangChain providers) | No (Anthropic only) | No (Anthropic only) | N/A | N/A |
| **MCP server support** | Yes | Yes | Yes | Yes | No | No |
| **Tool use** | Yes | Yes | Yes | Yes | No | No |
| **Agent tier** | `tool_loop` | `deep_agent` | `deep_agent` | `tool_loop` | `tool_loop` | N/A (no-op) |
| **Parser: structured output** | No (JSON fallback) | Yes (native via LangChain) | Yes (native) | Yes (native) | No | No (no-op) |
| **Parser: system prompt** | Yes | Yes | Yes | Yes | No | N/A |
| **Prompt caching** | Via middleware config | Via provider config | Via SDK | Native API support | No | No |
| **Auto-fallback** | None (base adapter) | None (explicit install) | `langchain` | `langchain` | None | None |
| **Availability check** | `langchain_core` importable | `deepagents` importable | `claude` CLI in PATH | `anthropic` importable | Always available | Always available |

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
    id="haiku",
    model_name="claude-haiku-4-5",
    model_provider="anthropic",  # Required for langchain
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

## `langchain_deep_agents` — LangChain Deep Agents

A natively agentic adapter using LangChain Deep Agents (`create_deep_agent`). Unlike the standard `langchain` adapter, which orchestrates each tool call turn explicitly (`tool_loop`), this adapter wraps a full agent runtime with built-in planning, context management, and subagent orchestration (`deep_agent` tier). It supports all LangChain-compatible providers.

### Implementation Classes

| Port | Class | Module |
|------|-------|--------|
| AgentPort | `DeepAgentsAgentAdapter` | `adapters.langchain_deep_agents.agent` |
| ParserPort | `DeepAgentsParserAdapter` | `adapters.langchain_deep_agents.parser` |
| LLMPort | `DeepAgentsLLMAdapter` | `adapters.langchain_deep_agents.llm` |

### Configuration

```python
from karenina.schemas.config import ModelConfig

config = ModelConfig(
    id="deep-agent",
    model_name="claude-sonnet-4-20250514",
    model_provider="anthropic",  # Required for langchain_deep_agents
    interface="langchain_deep_agents",
)
```

**Required fields**: `id`, `model_name`, `model_provider`

**Environment variables**: Provider-specific API key (e.g., `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`)

**Install**: `pip install deepagents langchain-mcp-adapters`

**Availability check**: Checks that the `deepagents` package is importable. No fallback interface is provided because Deep Agents' natively agentic behavior cannot be meaningfully approximated by the scaffolded LangChain adapter.

### Agent Behavior

The agent adapter uses `create_deep_agent()` from the `deepagents` package, which returns a compiled LangGraph graph. The adapter handles:

- System prompt extraction and forwarding to `create_deep_agent(system_prompt=...)`
- Backend selection: uses `FilesystemBackend` with an optional `workspace_path` from `AgentConfig`, falling back to the current working directory
- Recursion limit control via LangGraph config (derived from `AgentConfig.max_turns`)
- Dual trace output: both a raw trace string and structured `trace_messages` list
- Usage metadata extraction from `AIMessage.response_metadata`
- Recursion limit detection from LangGraph state (`is_last_step`)
- MCP server support via `langchain-mcp-adapters` (converts `MCPServerConfig` to MCP tools using `AsyncExitStack` for session lifetime management)
- Static tool support (karenina `Tool` objects converted to LangChain tools and passed to `create_deep_agent`)

Extra configuration can be passed through `AgentConfig.extra`, which forwards arbitrary keyword arguments to `create_deep_agent()`.

### LLM Behavior

For single-turn calls, the LLM adapter uses LangChain's `init_chat_model` directly (not `create_deep_agent`), since no agent loop or tool calling is needed. This keeps simple invocations lightweight.

### Parser Behavior

The Deep Agents parser uses **native structured output** (`supports_structured_output = True`) via LangChain's `with_structured_output()`. It uses `include_raw=True` to preserve the `AIMessage` alongside the parsed output, ensuring usage metadata is not lost. If structured output fails, the parser falls back to JSON text extraction.

### Adapter Instructions

Registers prompt instructions for `langchain_deep_agents` interface covering parsing, rubric, and deep judgment tasks.

### When to Use

- You need a full agent runtime with built-in planning and subagent orchestration
- You want provider-agnostic agentic evaluation (not limited to Anthropic)
- You need the `deep_agent` tier for capturing complete tool call traces through `AgentPort`
- Your evaluation tasks benefit from filesystem access, context management, or multi-step planning

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

See [Manual Interface](../notebooks/core_concepts/manual-interface.ipynb) for trace format and workflow details.

---

## `taskeval` — TaskEval No-Op Interface

A registry sentinel used by TaskEval (evaluation of pre-collected LLM outputs). The interface registers an `AdapterSpec` with `llm_factory=None`, `parser_factory=None`, and `agent_factory=None`: no LLM call ever happens through this adapter. It exists so that `ModelConfig(interface="taskeval")` validates and so `AdapterRegistry.get_spec("taskeval")` returns a real spec when the TaskEval pipeline routes around answer generation.

### Implementation

| Aspect | Value |
|--------|-------|
| Module | `adapters.taskeval` (registration in `adapters.taskeval.registration`) |
| Spec | `AdapterSpec(interface="taskeval", llm_factory=None, parser_factory=None, agent_factory=None, supports_mcp=False, supports_tools=False, requires_provider=False)` |
| Availability | Always available |

### Configuration

```python
from karenina.schemas.config import ModelConfig

config = ModelConfig(
    id="taskeval-answerer",
    model_name="recorded-output",   # any sentinel name
    interface="taskeval",
)
```

**Required fields**: `id`, `model_name` (the values are arbitrary sentinels; they identify the recorded output set, not a live model).

### When to Use

- Running TaskEval to score pre-collected LLM outputs, chatbot logs, or agent traces against rubrics/templates
- Producing `VerificationResult` records for outputs that were never generated by karenina

This adapter is wired by TaskEval; users do not normally instantiate it directly. See the [TaskEval skill / docs](../notebooks/core_concepts/task-eval.ipynb) for the supported workflow.

<div class="admonition note">
<p class="admonition-title">Not an adapter: <code>karenina/src/karenina/adapters/deep_agents/</code></p>
<p>A directory named <code>deep_agents/</code> exists in the source tree but is <strong>not</strong> a registered adapter. Only <code>__pycache__</code> and an empty <code>prompts/</code> subdirectory remain; there is no <code>registration.py</code> and no entry in <code>AdapterRegistry._load_builtins()</code>. The actual deep-agents adapter is <code>langchain_deep_agents/</code> (interface <code>"langchain_deep_agents"</code>), documented above. Treat the bare <code>deep_agents/</code> path as historical residue, not a usable interface.</p>
</div>

---

## Choosing an Adapter

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Getting started | `langchain` | Default, broadest provider support |
| Multi-provider evaluation | `langchain` | Supports OpenAI, Anthropic, Google, and more |
| Single API key for many models | `openrouter` | 200+ models, one API key |
| Local LLM server | `openai_endpoint` | Ollama, vLLM, LM Studio, etc. |
| Natively agentic evaluation (multi-provider) | `langchain_deep_agents` | Deep agent runtime with planning and subagents |
| Claude-only with full features | `claude_agent_sdk` | Native structured output, session management |
| Claude-only, lightweight | `claude_tool` | Native structured output, prompt caching, simpler setup |
| Offline / CI / reproducibility | `manual` | No API calls, pre-recorded traces |
| Need native structured output | `claude_agent_sdk`, `claude_tool`, or `langchain_deep_agents` | All have `supports_structured_output = True` |

---

## Provider Requirements Summary

| Interface | `model_provider` Required | API Key Environment Variable |
|-----------|:-------------------------:|------------------------------|
| `langchain` | Yes | Provider-specific: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` |
| `langchain_deep_agents` | Yes | Provider-specific: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` |
| `openrouter` | No | `OPENROUTER_API_KEY` |
| `openai_endpoint` | No | Set via `endpoint_api_key` on ModelConfig |
| `claude_agent_sdk` | No | `ANTHROPIC_API_KEY` (used by Claude CLI) |
| `claude_tool` | No | `ANTHROPIC_API_KEY` (or `anthropic_api_key` on ModelConfig) |
| `manual` | No | None |
| `taskeval` | No | None |

---

## Related

- [Adapter Architecture](index.md) — Hexagonal architecture, registry, and factory system
- [Port Types](ports.md) — Complete protocol signatures for LLMPort, ParserPort, AgentPort
- [MCP Integration](mcp-integration.md) — How adapters handle MCP servers and tool schemas
- [Writing Custom Adapters](writing-adapters.md) — Guide to implementing and registering new adapters
- [Adapters Overview](../core_concepts/adapters.md) — Conceptual introduction to the adapter system
