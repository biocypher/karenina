# Available Adapters

Karenina ships with eight adapter packages (`langchain`, `claude_agent_sdk`, `claude_tool`, `langchain_deep_agents`, `google_genai`, `omp`, `manual`, `taskeval`) and two routing interfaces (`openrouter`, `openai_endpoint`) that delegate to LangChain, totaling **ten registered `interface` values**. This page documents each adapter's implementation details, capabilities, configuration requirements, and adapter-specific behavior.

For the conceptual introduction to adapters, see [Adapters Overview](../core_concepts/adapters.md). For port protocol signatures, see [Port Types](ports.md).

---

## Feature Comparison

| Feature | `langchain` | `langchain_deep_agents` | `claude_agent_sdk` | `claude_tool` | `google_genai` | `omp` | `manual` | `taskeval` |
|---------|:-----------:|:-----------------------:|:------------------:|:-------------:|:--------------:|:-----:|:--------:|:----------:|
| **Multi-provider support** | Yes (all LangChain providers) | Yes (all LangChain providers) | No (Anthropic only) | No (Anthropic only) | No (Gemini only) | Yes (OMP catalog) | N/A | N/A |
| **MCP server support** | Yes | Yes | Yes | Yes | Yes | Yes (ACP) | No | No |
| **Tool use** | Yes | Yes | Yes | Yes | Yes | Yes (built in) | No | No |
| **Agent tier** | `tool_loop` | `deep_agent` | `deep_agent` | `tool_loop` | `tool_loop` | `deep_agent` | `tool_loop` | N/A (no-op) |
| **Parser: structured output** | No (JSON fallback) | Yes (native via LangChain) | Yes (native) | Yes (native) | Yes (native JSON schema) | N/A (agent-only) | No | No (no-op) |
| **Parser: system prompt** | Yes | Yes | Yes | Yes | Yes | N/A | No | N/A |
| **Prompt caching** | Via middleware config | Via provider config | Via SDK | Native API support | Not configured (cache reads reported in usage) | Provider/OMP cache | No | No |
| **Auto-fallback** | None (base adapter) | None (explicit install) | `langchain` | `langchain` | None | `langchain` when unavailable | None | None |
| **Availability check** | `langchain_core` importable | `deepagents` importable | `claude` CLI in PATH | `anthropic` importable | `google.genai` importable | `omp` CLI + `acp` package | Always available | Always available |

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

### Custom Anthropic-Compatible Endpoint

For an Anthropic-compatible gateway or proxy, configure its URL and key on
`ModelConfig` instead of relying on `ANTHROPIC_API_KEY` from the environment:

```python
config = ModelConfig(
    id="gateway-model",
    model_name="gateway-model",
    model_provider="anthropic",
    interface="langchain",
    anthropic_base_url="https://gateway.example/anthropic",
    anthropic_api_key="gateway-key",
)
```

Both fields are forwarded to LangChain's Anthropic model as `base_url` and
`api_key`. They require `model_provider="anthropic"`; configuring either on a
different provider or unsupported interface raises a validation error. Do not
also set `extra_kwargs["base_url"]` or `extra_kwargs["api_key"]`, as ambiguous
connection settings are rejected.

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

For structured parsing (the [parse_template stage](../core_concepts/verification-pipeline.md)), the adapter treats `openai_endpoint` specially: it builds the structured model with `with_structured_output(schema, method="json_schema")` so the server enforces the schema through guided decoding rather than the LangChain default function-calling path, which small self-hosted models often fail (silently returning `None` or raising). If the backend rejects the `json_schema` method, the adapter retries `with_structured_output` without the override, and if that also fails it falls back to manual JSON parsing at invoke time. For the most reliable parsing, run a server that supports guided decoding / JSON-schema-constrained output (vLLM and Ollama both do).

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

### Endpoints Not Served at `/v1`

By default the interface appends `/v1` to `endpoint_base_url` (the common OpenAI-compatible convention), so `http://localhost:8000` and `http://localhost:8000/v1` behave identically. Some hosted APIs expose their OpenAI-compatible path under a different version segment (for example a z.ai coding endpoint served at `.../v4/chat/completions`). For those, pass `endpoint_base_url_mode` through [`extra_kwargs`](../reference/configuration/model-config.md#advanced) so the base URL is used exactly as given:

```python
config = ModelConfig(
    id="zai-coding",
    model_name="glm-4.6",
    interface="openai_endpoint",
    endpoint_base_url="https://api.z.ai/api/coding/paas/v4",
    endpoint_api_key="sk-...",
    extra_kwargs={"endpoint_base_url_mode": "raw"},
)
```

`endpoint_base_url_mode` accepts `"auto_v1"` (the default: appends `/v1` unless the URL already ends in `/v1`) or `"raw"` (uses `endpoint_base_url` verbatim, only stripping a trailing slash). Any other value raises. The same key works for the [`langchain_deep_agents`](#langchain_deep_agents-langchain-deep-agents) interface when it targets an OpenAI-compatible endpoint.

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

For Anthropic-compatible gateways, use the same `anthropic_base_url` and
`anthropic_api_key` fields shown in the `langchain` configuration above. They
are forwarded to the underlying LangChain Anthropic model and require
`model_provider="anthropic"`.

**Install**: `pip install deepagents langchain-mcp-adapters`

**Availability check**: Checks that the `deepagents` package is importable. No fallback interface is provided because Deep Agents' natively agentic behavior cannot be meaningfully approximated by the scaffolded LangChain adapter.

### Extended Thinking (`effort`)

For `model_provider="anthropic"`, pass an `effort` tier through `extra_kwargs` to enable Anthropic extended thinking:

```python
config = ModelConfig(
    id="deep-agent",
    model_name="claude-sonnet-4-20250514",
    model_provider="anthropic",
    interface="langchain_deep_agents",
    extra_kwargs={"effort": "high"},
)
```

The tier maps to an extended-thinking `budget_tokens` value:

| `effort` | `budget_tokens` |
|----------|-----------------|
| `"low"` | 2048 |
| `"medium"` | 8192 |
| `"high"` | 16384 |
| `"xhigh"` | 16384 |
| `"max"` | 32000 |

When `effort` is set, the adapter turns on extended thinking with that budget, forces `temperature=1` (a requirement of Anthropic extended thinking, so any configured temperature is overridden with a warning), and raises `max_tokens` to sit above the budget (at least `budget + 4096`, since Anthropic requires `max_tokens` greater than `thinking.budget_tokens`). An unrecognized tier raises `AdapterUnavailableError`. `effort` is ignored for non-Anthropic providers. The tiers mirror the `claude --effort` reasoning levels used by the native [`claude_agent_sdk`](#claude_agent_sdk-native-anthropic-agent-sdk) path.

### Agent Behavior

The agent adapter uses `create_deep_agent()` from the `deepagents` package, which returns a compiled LangGraph graph. The adapter handles:

- System prompt extraction and forwarding to `create_deep_agent(system_prompt=...)`
- Backend selection (four-way, keyed off `ModelConfig.agent_runtime.backend`): `FilesystemBackend` (`filesystem`, the default) exposes file tools rooted at the `AgentConfig.workspace_path` (or the current working directory when unset), `ContainerSandboxBackend` (`container`) runs file and shell tools inside a Docker/Singularity/Apptainer image, and `LocalShellBackend` (`local_shell`) runs them directly on the host (unsafe, logged as such). When `access_mode='read_only'`, the chosen backend is wrapped in a `ReadOnlyBackend` that blocks writes and command execution. See [Sandboxed Runtime Backends](../core_concepts/agentic-evaluation.md).
- Recursion limit control via LangGraph config (derived from `AgentConfig.max_turns`)
- Dual trace output: both a raw trace string and structured `trace_messages` list
- Usage metadata extraction from `AIMessage.response_metadata`
- Recursion limit detection from LangGraph state (`is_last_step`)
- MCP server support via `langchain-mcp-adapters` (converts `MCPServerConfig` to MCP tools using `AsyncExitStack` for session lifetime management)
- Static tool support (karenina `Tool` objects converted to LangChain tools and passed to `create_deep_agent`)

Extra configuration can be passed through `AgentConfig.extra`, which forwards arbitrary keyword arguments to `create_deep_agent()`.

When the model targets an OpenAI-compatible hosted endpoint via `endpoint_base_url`, `extra_kwargs['endpoint_base_url_mode']` controls how that base URL is finalized: `'auto_v1'` (the default) appends `/v1` when it is missing, while `'raw'` passes the URL through unchanged. Use `'raw'` for hosted APIs that already carry a provider-specific version segment as the final OpenAI SDK base URL (for example Z.ai's coding endpoint).

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

The Claude SDK parser uses **native structured output** (`supports_structured_output = True`), but it makes **direct API calls** and bypasses the Agent SDK subprocess transport used by the agent and LLM adapters:

- For the Anthropic API (no custom base URL), calls `anthropic.AsyncAnthropic().messages.create()` with `output_config={'format': {'type': 'json_schema', 'schema': ...}}` for constrained decoding
- For custom OpenAI-compatible endpoints (vLLM, sglang), calls `openai.AsyncOpenAI().chat.completions.create()` with `response_format={'type': 'json_schema', ...}`. The OpenAI-compatible base URL is derived from the model's Anthropic base URL by default (same host and port, with `/v1` appended). Set `extra_kwargs['claude_sdk_parser_openai_base_url']` to point these structured-output calls at an explicit OpenAI-compatible schema endpoint instead
- Reads the text response and parses it with `json.loads()` (falling back to `extract_json_from_response()` for fenced or mixed output), then validates via `schema.model_validate(data)`
- Retries are handled by the Anthropic/OpenAI SDK client's native `max_retries`, derived from `RetryPolicy.derive_sdk_max_retries()`, not by `RetryExecutor`

### Agent Behavior

The agent adapter uses `ClaudeSDKClient` for full agent functionality:

- Native session management with multi-turn conversation support
- Built-in tool execution via the SDK
- MCP server integration (native support)
- Trace capture via `sdk_messages_to_raw_trace()` conversion
- Runtime backend selection via `ModelConfig.agent_runtime.backend`: `native` (the default) runs on the host with Claude Code's Bash sandbox (toggled by `sandbox_enabled`), while `container` wraps every Bash command so it runs inside a Docker/Singularity/Apptainer image mounted at `/workspace`. See [Sandboxed Runtime Backends](../core_concepts/agentic-evaluation.md)

By default the adapter disallows Claude Code's bundled `Skill` tool (it applies `disallowed_tools=["Skill"]` unless the caller sets that key). Newer Claude Code CLIs (2.1.170 and later) advertise their bundled skills by appending a system-role message to every request's `messages` array. That message contaminates benchmark conversations with skill listings, and OpenAI-compatible Anthropic endpoints such as vLLM's `/v1/messages` reject it with HTTP 400 because only `user` and `assistant` roles are accepted there. Disallowing the `Skill` tool suppresses the system-role message entirely. To re-enable skills, pass `AgentConfig(extra={"disallowed_tools": []})`.

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

## `google_genai`: Gemini Through the Google GenAI SDK

Uses the `google-genai` Python SDK directly for Gemini models, on Vertex AI or on the Gemini API (including gateways that expose it). All three ports are implemented. Compared with reaching Gemini through `langchain`, this adapter keeps Gemini's grounding metadata (search queries and sources) in the trace, reports thinking tokens and search counts in usage, and can cap Google Search per agent run.

### Implementation Classes

| Port | Class | Module |
|------|-------|--------|
| AgentPort | `GoogleGenAIAgentAdapter` | `adapters.google_genai.agent` |
| ParserPort | `GoogleGenAIParserAdapter` | `adapters.google_genai.parser` |
| LLMPort | `GoogleGenAILLMAdapter` | `adapters.google_genai.llm` |

### Configuration

```python
from karenina.schemas.config import ModelConfig

config = ModelConfig(
    id="gemini-judge",
    model_name="gemini-3.8-flash",
    interface="google_genai",
    extra_kwargs={"project": "my-gcp-project"},  # or set GOOGLE_CLOUD_PROJECT
)
```

**Required fields**: `id`, `model_name`. `model_provider` is not used.

**Install**: `pip install 'karenina[google-genai]'`. The `google-genai` package also comes with karenina's core LangChain Google dependency, so the adapter is usually available without the extra.

**Availability check**: Checks that `google.genai` is importable. There is no fallback interface: a silent fallback to LangChain would drop the grounding trail and the search budget.

Settings are validated when the adapter is built, so a configuration mistake fails before the first model call.

#### Backends and Credentials

| Backend | Selected when | Credentials | Notes |
|---------|---------------|-------------|-------|
| `vertex` (default) | `endpoint_base_url` is not set | Google application default credentials (`gcloud auth application-default login`) | Project from `extra_kwargs["project"]` or `GOOGLE_CLOUD_PROJECT`. Location from `extra_kwargs["location"]`, then `GOOGLE_CLOUD_LOCATION`, then `"global"` |
| `gemini_api` | `endpoint_base_url` is set, or `extra_kwargs["backend"]="gemini_api"` | `endpoint_api_key`, then `GEMINI_API_KEY`, then `GOOGLE_API_KEY` | `endpoint_base_url` points the client at a gateway. `project` and `location` are ignored with a warning |

Setting `endpoint_api_key` with the `vertex` backend raises `ValueError`. Missing or expired Google credentials fail as a permanent error whose message names the `gcloud auth application-default login` fix.

#### ModelConfig Fields

| Field | Effect |
|-------|--------|
| `temperature` | Sampling temperature (some Gemini models ignore it) |
| `max_tokens` | Sent as `max_output_tokens`. Thinking and answer tokens share this budget |
| `system_prompt` | System instruction used when the messages carry none |
| `request_timeout` | Wall-clock limit for each model call. Defaults to 300 seconds when unset, so one stuck call cannot hold a whole run |
| `retry_policy` | Retries through `RetryExecutor`. The SDK's own HTTP retries are switched off, so the [retry policy](../advanced-pipeline/error-handling.md) is the only retry layer and only the failed call is repeated. One exception cannot be switched off: on its aiohttp transport the SDK resends a request once, after a 1 to 10 second sleep, when the connection fails or drops, inside the same attempt and its time limit. A retried call that Gemini answered without a usable result (for example an empty `STOP`) still counts toward usage and the search budget |
| `mcp_urls_dict`, `mcp_tool_filter`, `mcp_tool_description_overrides`, `mcp_http_timeout`, `mcp_sse_read_timeout` | MCP servers and tool filtering, with the same meaning as for `claude_tool` |
| `extra_kwargs` | The keys in the next table. Any other key raises `ValueError` |

#### `extra_kwargs` Keys

| Key | Default | Meaning |
|-----|---------|---------|
| `backend` | `"gemini_api"` when `endpoint_base_url` is set, else `"vertex"` | Backend to call |
| `project` | `GOOGLE_CLOUD_PROJECT` | Vertex project |
| `location` | `GOOGLE_CLOUD_LOCATION`, else `"global"` | Vertex location |
| `builtin_tools` | `[]` | Gemini server-side tools to enable: any of `"google_search"`, `"url_context"` |
| `max_search_queries` | `None` (no cap) | Google Search budget per agent run (see below) |
| `thinking_level` | `None` (model default) | Passed to `ThinkingConfig.thinking_level`, checked against the levels the installed SDK knows |
| `thinking_budget` | `None` | Passed to `ThinkingConfig.thinking_budget`. Cannot be combined with `thinking_level` |
| `include_thoughts` | `True` on `gemini_api`, `False` on `vertex` | Request thought summaries. They appear as thinking blocks in agent traces |
| `stream` | `True` on `gemini_api`, `False` on `vertex` | Use streamed calls, so a gateway that cuts silent connections does not drop long calls |
| `resolve_grounding_urls` | `False` | Follow Google's grounding redirect links to the real source URL (one `HEAD` request per source) |
| `generate_content_config` | `{}` | Extra `GenerateContentConfig` fields passed through, for example `seed` or `safety_settings`. Fields the adapter sets itself (`tools`, `temperature`, `max_output_tokens`, `system_instruction`, `thinking_config`, the response schema fields, `http_options`, `automatic_function_calling`) are rejected |

### Built-In Tools and the Search Budget

`google_search` and `url_context` are off by default. Gemini runs them on Google's side inside a single model call, so they need no MCP server and no tool loop:

- They apply to every agent turn and to every LLM-port call made with this `ModelConfig`, for example [rubric evaluation](../core_concepts/rubrics/index.md) with a Gemini parsing model. The parser port never enables them, so template parsing and score extraction never search.
- Google bills each search query, and page text read through `url_context` counts as input tokens.
- `url_context` opens only URLs written out in the request. Given an identifier such as a DOI or PMID, the model searches instead. To let a model open pages it finds, give it a fetch tool through an [MCP server](mcp-integration.md).

`max_search_queries` is a soft cap per agent run. After each call the agent adds the number of queries Gemini reported, and once the count reaches the cap it removes Google Search from the remaining turns (a note in `raw_trace` records when). Gemini can run several searches inside one call, so a run can end above the cap. LLM-port calls are single calls and have no budget.

Callers that build their own `AgentConfig` can override `builtin_tools`, `max_search_queries` and `max_tool_result_chars` for one run through `AgentConfig.extra`. `max_tool_result_chars` (default 50,000) cuts longer function and MCP tool results, with a marker, before they reach the model.

### Agent Behavior

The agent runs a karenina-owned tool loop. The SDK's automatic function calling is switched off:

- Each turn is one `generate_content` call, and `AgentConfig.max_turns` counts these calls.
- Static tools and MCP tools are declared as Gemini function declarations. Names are rewritten to fit Gemini's naming rules and mapped back when the model calls them. Two tools whose rewritten names collide fail the run before any model call.
- The adapter runs each function call itself. A tool error goes back to the model as an error result and is marked `is_error` in the trace. It never ends the run.
- The model's turns are sent back unchanged, so Gemini's thought signatures survive across turns.
- `AgentConfig.timeout` bounds the whole run, MCP connection setup included. On a timeout or the turn limit the agent returns the partial trace with `timeout_reached` or `limit_reached` set. A timeout before any message was produced raises `AgentTimeoutError`.
- Every model call, agent turns included, holds one permit of the global LLM limiter, so [`VerificationConfig.max_concurrent_requests`](../reference/configuration/verification-config.md) also caps agent turns.
- There are no file or shell tools (`supports_file_tools=False`, `supports_code_execution=False`). Files reach the agent only through an MCP server.

Gemini's finish reasons are handled as follows:

| Finish reason | Agent | LLM port |
|---------------|-------|----------|
| `STOP` with text or tool calls | Normal | Normal |
| `STOP` with no text and no tool call, or a stream that ends without a finish reason | Retried as a server error | Retried as a server error |
| `MAX_TOKENS` | Text kept, note added. A final turn with no text raises `AgentResponseError` | Text kept. Structured output then usually fails to validate |
| `TOO_MANY_TOOL_CALLS` (Gemini's own server-side tool loop ran too long) | Run ends with `limit_reached=True` and a note, or raises `AgentResponseError` when no text was written | Permanent error |
| `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL` | A corrective user message is added and the loop continues (counts as a turn) | Permanent error |
| `SAFETY`, `RECITATION`, `PROHIBITED_CONTENT`, `BLOCKLIST`, `SPII`, `OTHER`, unknown reasons | Permanent error naming the reason | Permanent error naming the reason |

### Trace Capture

`trace_messages` is built turn by turn: assistant messages (text, tool calls, and thinking blocks when `include_thoughts` is on), tool results, and Gemini's server-side activity as synthetic tool pairs:

| Synthetic tool | `ToolUseContent.input` | `ToolResultContent` content (JSON) |
|----------------|------------------------|-------------------------------------|
| `google_search` | `{"queries": [...]}` | `{"server_side": true, "sources": [{"title", "uri"}], "supported_segments": n}` |
| `url_context` | `{"urls": [...]}` | `{"server_side": true, "urls": [{"url", "status"}]}` |

The pairs come before the model's text for the same turn, the order in which they happened, with ids `google_search_<turn>` and `url_context_<turn>`. The trace viewer, regex highlighting and trace parsers therefore see searches and page reads like any other tool call. Each assistant message also carries that call's usage in `usage_metadata`.

`raw_trace` comes from the shared formatter, plus one `[Note: ...]` line for each event that ended or shaped the run (turn limit, timeout, finish reasons, search budget).

A single LLM-port call returns text only. For example, `GenerateAnswer` runs a `google_genai` answering model without MCP servers through `LLMPort`, so its searches show up only in usage, not in the trace.

### Usage

| `UsageMetadata` field | Source in Gemini's `usage_metadata` |
|-----------------------|-------------------------------------|
| `input_tokens` | `prompt_token_count` + `tool_use_prompt_token_count` (cached tokens included) |
| `output_tokens` | `candidates_token_count` + `thoughts_token_count` |
| `cache_read_tokens` | `cached_content_token_count` |
| `reasoning_tokens` | `thoughts_token_count` (already counted in `output_tokens`) |
| `web_search_queries` | Number of Google Search queries. Set when Google Search was available to the call or a query was reported |
| `cost_usd` | Always `None` (karenina has no price table) |

An agent run's usage is the sum over all its calls. See [UsageMetadata](ports.md#usagemetadata).

### LLM and Parser Behavior

The LLM adapter makes single calls through the same call path as the agent. Structured output (`with_structured_output`) sends the Pydantic schema as `response_json_schema`, validates the reply with the schema, and retries with the validation error as feedback (3 retries by default). If Gemini rejects the JSON Schema form, the adapter tries once more with the SDK's own schema conversion. `astream` and `stream_invoke` are supported and skip thought parts.

The parser uses the LLM adapter's **native structured output** (`supports_structured_output = True`) with 2 validation retries, and always with built-in tools off. A reply that never validates raises `ParseError`.

### Errors and Retries

| Condition | Error category |
|-----------|----------------|
| HTTP 5xx (including a gateway's HTML error page), empty or truncated responses, non-JSON response bodies | `SERVER_ERROR` (retried) |
| HTTP 429 or `RESOURCE_EXHAUSTED` | `RATE_LIMIT` (retried) |
| Network failures, including while refreshing Google credentials | `CONNECTION` (retried) |
| Other HTTP 4xx (400, 403, 404), missing credentials, blocking finish reasons | `PERMANENT` |

### Adapter Instructions

None. The default parsing, rubric, abstention and sufficiency instructions apply.

### When to Use

- Gemini models where you need the search queries and sources in the trace, thinking-token counts, or a cap on searches
- Gemini on Vertex AI with Google application default credentials, or through a Gemini API gateway
- A Gemini judge for [agentic rubric traits](../core_concepts/rubrics/agentic-traits.md) that checks claims with MCP tools and Google Search (`context_mode="trace_only"`, see [Agentic Rubric Evaluation](../advanced-pipeline/agentic-rubric-evaluation.md))

---

## `omp` — Oh My Pi over Agent Client Protocol

An AgentPort-only deep-agent adapter for the multi-provider Oh My Pi coding agent. Karenina launches `omp acp` and uses ACP v1 for version/capability negotiation, sessions, streamed messages and thoughts, tool lifecycle events, MCP configuration, usage, and cancellation.

```python
config = ModelConfig(
    id="omp-glm",
    model_provider="zhipu-coding-plan",
    model_name="glm-5.3",
    interface="omp",
    extra_kwargs={"thinking": "high"},
)
```

**Required fields**: `id`, `model_provider`, `model_name`

**Install**: install OMP from <https://omp.sh/>, then `pip install 'karenina[omp]'`.

The adapter uses the selector `model_provider/model_name`, so the example selects `zhipu-coding-plan/glm-5.3`. OMP resolves credentials from its auth store or provider environment variables.

The adapter implements AgentPort only. Configure a separate LLM/Parser model for parsing, rubric evaluation, and other single-turn stages. Its `langchain` fallback applies when OMP/ACP is unavailable; it does not supply the unsupported OMP LLM/Parser factories.

OMP has built-in file and shell tools. Karenina `Tool` objects are not accepted because ACP v1 has no client-supplied callable-tool definition, but MCP stdio/HTTP/SSE servers are forwarded through `session/new`. By default skills, rules, and extensions are disabled for benchmark reproducibility; opt in with `extra_kwargs` keys `enable_skills`, `enable_rules`, and `enable_extensions`.

Native OMP runs on the host. `workspace_path` selects its working directory and read-only mode uses ACP permission rejection, but neither is an OS sandbox. Run Karenina/OMP in external isolation when hard containment is required.

ACP reports cache usage separately. Karenina records it in `cache_read_tokens`/`cache_creation_tokens` and keeps `total_tokens = input_tokens + output_tokens`.

`AgentConfig.timeout` covers ACP initialization, session/model setup, and the prompt. On timeout the adapter sends cancellation, drains updates for `extra_kwargs["timeout_cancel_grace"]` seconds (default 10), and returns the partial trace with `timeout_reached=True`. A timeout before any trace raises `AgentTimeoutError`. Cancellation, connection cleanup, and process shutdown have bounded waits; only the last 4,000 bytes of stderr are retained for diagnostics.

HTTP and SSE MCP servers require the matching capability in OMP's ACP initialization response. Unsupported transports fail before session creation. Stdio MCP servers do not require an optional capability.

ACP v1 provides no enforceable turn ceiling, so `AgentConfig.max_turns` is best effort. Use `AgentConfig.timeout` when the run needs a wall-clock limit.

Some OMP provider failures arrive over ACP as assistant text with `end_turn`, rather than a protocol error; OMP omits turn usage when no tokens were consumed. A successful completed turn therefore requires a non-zero ACP usage summary. Missing or all-zero usage raises `AgentResponseError`, including for providers or OMP versions that do not expose turn usage, so authentication or routing errors do not become successful benchmark answers. Partial timeout/limit results may lack usage. ACP `refusal` responses also raise `AgentResponseError`.

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

All three manual adapters return the default `PortCapabilities()` with no arguments, so `supports_system_prompt` is `True` (the dataclass default) and `supports_structured_output` is `False`. These capability flags are inert in practice: the LLM and parser adapters are no-ops that raise `ManualInterfaceError`, and the agent adapter only replays a pre-recorded trace, so no system prompt, structured output, MCP, or tools ever reach a live backend.

### When to Use

- **Testing and CI**: Run verification without API calls or costs
- **Reproducibility**: Evaluate the same traces repeatedly with different configurations
- **External traces**: Evaluate responses from models or systems outside karenina
- **Development**: Iterate on templates and rubrics without waiting for LLM responses

See [Manual Interface](../core_concepts/manual-interface.md) for trace format and workflow details.

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

This adapter is wired by TaskEval; users do not normally instantiate it directly. See the [TaskEval skill / docs](../core_concepts/task-eval.md) for the supported workflow.

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
| Gemini with Google Search grounding or Vertex AI | `google_genai` | Search queries and sources in the trace, search budget, thinking-token usage |
| Multi-provider coding-agent subscriptions | `omp` | Standard ACP bridge to OMP providers and built-in tools |
| Offline / CI / reproducibility | `manual` | No API calls, pre-recorded traces |
| Need native structured output | `claude_agent_sdk`, `claude_tool`, `google_genai`, or `langchain_deep_agents` | All have `supports_structured_output = True` |

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
| `google_genai` | No | `vertex` backend: Google application default credentials (no key). `gemini_api` backend: `GEMINI_API_KEY` or `GOOGLE_API_KEY` (or `endpoint_api_key` on ModelConfig) |
| `omp` | Yes | Provider-specific OMP auth or environment variable |
| `manual` | No | None |
| `taskeval` | No | None |

---

## Related

- [Adapter Architecture](index.md) — Hexagonal architecture, registry, and factory system
- [Port Types](ports.md) — Complete protocol signatures for LLMPort, ParserPort, AgentPort
- [MCP Integration](mcp-integration.md) — How adapters handle MCP servers and tool schemas
- [Writing Custom Adapters](writing-adapters.md) — Guide to implementing and registering new adapters
- [Adapters Overview](../core_concepts/adapters.md) — Conceptual introduction to the adapter system
