# ModelConfig Reference

This is the exhaustive reference for all `ModelConfig` fields. For a tutorial introduction with examples, see [VerificationConfig Tutorial](../../06-running-verification/verification-config.md) and [Adapters Overview](../../core_concepts/adapters.md).

`ModelConfig` is a Pydantic model with **19 fields** organized into 7 categories below. Import: `from karenina.schemas import ModelConfig`.

---

## Model Identity

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str \| None` | `None` | Unique identifier for this model configuration. **Required** for all non-manual interfaces. Defaults to `"manual"` for manual interface. Used in results to identify which model produced each result. |
| `model_name` | `str \| None` | `None` | Model name passed to the underlying provider (e.g., `"gpt-4o"`, `"claude-sonnet-4-20250514"`, `"gemini-2.0-flash"`). **Required** for all non-manual interfaces. Defaults to `"manual"` for manual interface. |
| `model_provider` | `str \| None` | `None` | LLM provider name (e.g., `"openai"`, `"anthropic"`, `"google_genai"`). **Required** only for the `langchain` interface (passed to `init_chat_model()`). Not required for other interfaces. |
| `interface` | `Literal["langchain", "openrouter", "openai_endpoint", "claude_agent_sdk", "claude_tool", "manual"]` | `"langchain"` | Which adapter backend to use. See [Adapters Overview](../../core_concepts/adapters.md) for capabilities and trade-offs. |

**Validation rules:**

- Non-manual interfaces require both `id` and `model_name`
- Manual interface auto-sets `id="manual"` and `model_name="manual"` if not provided
- Only `langchain` interface requires `model_provider`

---

## Model Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | `float` | `0.1` | Sampling temperature. Lower values (0.0–0.3) produce more deterministic output; higher values (0.7–1.0) increase creativity. |
| `max_tokens` | `int` | `8192` | Maximum tokens for model response. |
| `system_prompt` | `str \| None` | `None` | Custom system prompt. When `None`, a default is applied based on context: answering models get an expert assistant prompt, parsing models get a validation assistant prompt. |
| `max_retries` | `int` | `2` | Maximum retry attempts for model calls during template generation. |

---

## Interface-Specific: OpenAI Endpoint

These fields apply only when `interface="openai_endpoint"`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `endpoint_base_url` | `str \| None` | `None` | Custom endpoint base URL for OpenAI-compatible APIs (e.g., `"https://my-server.com/v1"`). Required for `openai_endpoint` interface. |
| `endpoint_api_key` | `SecretStr \| None` | `None` | API key for the custom endpoint. Stored as `SecretStr` to prevent accidental logging. |

---

## Interface-Specific: Anthropic

These fields apply when `interface="claude_agent_sdk"` or `interface="claude_tool"`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `anthropic_base_url` | `str \| None` | `None` | Custom Anthropic API endpoint URL. Use for proxies or self-hosted deployments. |
| `anthropic_api_key` | `SecretStr \| None` | `None` | Override the `ANTHROPIC_API_KEY` environment variable. Stored as `SecretStr` to prevent accidental logging. |

---

## MCP Configuration

These fields configure MCP (Model Context Protocol) tool access. See [MCP Integration Overview](../../core_concepts/mcp-overview.md) for architecture and usage patterns.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mcp_urls_dict` | `dict[str, str] \| None` | `None` | Mapping of MCP server names to URLs. When provided, the model runs as an agent with access to these MCP tool servers. Example: `{"filesystem": "http://localhost:3001/sse", "database": "http://localhost:3002/sse"}`. |
| `mcp_tool_filter` | `list[str] \| None` | `None` | Restrict which MCP tools are available. When `None`, all discovered tools from all servers are available. Example: `["read_file", "query_db"]`. |
| `mcp_tool_description_overrides` | `dict[str, str] \| None` | `None` | Override tool descriptions sent to the model. Useful for improving tool selection accuracy. Keys are tool names, values are replacement descriptions. |
| `max_context_tokens` | `int \| None` | `None` | Token threshold for triggering summarization middleware. When not set: `langchain` uses fraction-based triggering (auto-detected from model), `openai_endpoint` auto-detects from `/v1/models` API if available, `openrouter` defaults to `100000 × trigger_fraction`. |

**Validation rules:**

- MCP is not supported with `interface="manual"` (raises `ValueError`)

---

## Agent Middleware

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_middleware` | `AgentMiddlewareConfig \| None` | `None` | Middleware configuration for MCP-enabled agents. Only applies when `mcp_urls_dict` is provided. Controls retry behavior, execution limits, summarization, and prompt caching. When `None`, defaults are used. |

`AgentMiddlewareConfig` contains 5 sub-configurations:

| Sub-Config | Type | Description |
|------------|------|-------------|
| `limits` | `AgentLimitConfig` | Agent execution limits (model/tool call caps) |
| `model_retry` | `ModelRetryConfig` | Model call retry configuration |
| `tool_retry` | `ToolRetryConfig` | Tool call retry configuration |
| `summarization` | `SummarizationConfig` | Conversation summarization configuration |
| `prompt_caching` | `PromptCachingConfig` | Anthropic prompt caching (only Anthropic models) |

### AgentLimitConfig

Controls maximum model and tool calls to prevent infinite loops or excessive costs.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_call_limit` | `int` | `25` | Maximum LLM calls per agent invocation. |
| `tool_call_limit` | `int` | `50` | Maximum tool calls per agent invocation. |
| `exit_behavior` | `Literal["end", "continue"]` | `"end"` | `"end"`: returns partial response gracefully. `"continue"`: blocks exceeded calls but continues. |

### ModelRetryConfig

Controls automatic retry behavior for failed model calls with exponential backoff.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | `int` | `2` | Maximum retry attempts (total calls = max_retries + 1 initial). |
| `backoff_factor` | `float` | `2.0` | Multiplier for exponential backoff between retries. |
| `initial_delay` | `float` | `2.0` | Initial delay in seconds before first retry. |
| `max_delay` | `float` | `10.0` | Maximum delay in seconds between retries. |
| `jitter` | `bool` | `True` | Add random jitter (±25%) to retry delays. |
| `on_failure` | `Literal["continue", "raise"]` | `"continue"` | `"continue"`: returns partial response. `"raise"`: raises exception. |

### ToolRetryConfig

Controls automatic retry behavior for failed tool calls with exponential backoff.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | `int` | `3` | Maximum retry attempts for tool calls. |
| `backoff_factor` | `float` | `2.0` | Multiplier for exponential backoff between retries. |
| `initial_delay` | `float` | `1.0` | Initial delay in seconds before first retry. |
| `on_failure` | `Literal["return_message", "raise"]` | `"return_message"` | `"return_message"`: returns error as message. `"raise"`: raises exception. |

### SummarizationConfig

Automatically summarizes conversation history when approaching token limits.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable automatic summarization (default: `True` for MCP agents). |
| `model` | `str \| None` | `None` | Model for summarization (defaults to a lightweight model like gpt-4o-mini). |
| `trigger_fraction` | `float` | `0.8` | Fraction of context window that triggers summarization (0.0–1.0). |
| `trigger_tokens` | `int \| None` | `None` | Number of tokens that triggers summarization (overrides `trigger_fraction`). |
| `keep_messages` | `int` | `20` | Number of recent messages to preserve after summarization. |

### PromptCachingConfig

Reduces costs and latency by caching static prompt content on Anthropic's servers. Only applies to Anthropic models with the `langchain` interface.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable prompt caching (default: `True` for Anthropic models with MCP tools). |
| `ttl` | `Literal["5m", "1h"]` | `"5m"` | Time to live for cached content: `"5m"` (5 minutes) or `"1h"` (1 hour). |
| `min_messages_to_cache` | `int` | `0` | Minimum messages before caching starts. |
| `unsupported_model_behavior` | `Literal["ignore", "warn", "raise"]` | `"warn"` | Behavior when using non-Anthropic models. |

---

## Manual Interface

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `manual_traces` | `ManualTraces \| None` | `None` | Pre-recorded trace data for the manual interface. **Required** when `interface="manual"`. Excluded from serialization. See [Manual Interface](../../core_concepts/manual-interface.md) for format details. |

**Validation rules:**

- `interface="manual"` requires `manual_traces` to be set
- `interface="manual"` does not support MCP (`mcp_urls_dict` must be `None`)

---

## Advanced

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `extra_kwargs` | `dict[str, Any] \| None` | `None` | Extra keyword arguments passed to the underlying model interface. Useful for vendor-specific API keys, custom parameters, or provider-specific settings not covered by other fields. |

---

## Common Configuration Patterns

**LangChain with OpenAI:**

```python
ModelConfig(
    id="gpt-4o",
    model_name="gpt-4o",
    model_provider="openai",
    interface="langchain",
)
```

**Claude Agent SDK:**

```python
ModelConfig(
    id="claude-sonnet",
    model_name="claude-sonnet-4-20250514",
    interface="claude_agent_sdk",
)
```

**OpenAI-compatible endpoint:**

```python
ModelConfig(
    id="local-model",
    model_name="my-model",
    interface="openai_endpoint",
    endpoint_base_url="https://my-server.com/v1",
    endpoint_api_key="sk-...",
)
```

**MCP-enabled agent with middleware:**

```python
from karenina.schemas import ModelConfig
from karenina.schemas.config.models import (
    AgentMiddlewareConfig,
    AgentLimitConfig,
    SummarizationConfig,
)

ModelConfig(
    id="claude-agent",
    model_name="claude-sonnet-4-20250514",
    interface="claude_agent_sdk",
    mcp_urls_dict={
        "filesystem": "http://localhost:3001/sse",
    },
    agent_middleware=AgentMiddlewareConfig(
        limits=AgentLimitConfig(model_call_limit=10),
        summarization=SummarizationConfig(keep_messages=10),
    ),
)
```

---

## Related

- [VerificationConfig Reference](verification-config.md) — uses `ModelConfig` for `answering_models` and `parsing_models`
- [VerificationConfig Tutorial](../../06-running-verification/verification-config.md) — step-by-step configuration guide
- [Adapters Overview](../../core_concepts/adapters.md) — interface comparison and selection guide
- [MCP Integration Overview](../../core_concepts/mcp-overview.md) — MCP architecture and adapter capabilities
- [Environment Variables](../../03-configuration/environment-variables.md) — API keys and path configuration
