# Adapter Selection Reference

Choose an adapter by setting `interface` on ModelConfig. The default `"langchain"` works for the majority of use cases.

## Decision Table

| Need | `interface=` | Provider Required? | MCP Support | Tool Support | Fallback |
|------|-------------|-------------------|-------------|-------------|----------|
| General-purpose (Anthropic, OpenAI, Google) | `"langchain"` (default) | Yes | Yes | Yes | None (this IS the fallback) |
| OpenRouter models | `"openrouter"` | Yes | Yes | Yes | None (routes through LangChain) |
| OpenAI-compatible endpoint (vLLM, Ollama, etc.) | `"openai_endpoint"` | Yes | Yes | Yes | None (routes through LangChain) |
| Native Anthropic SDK integration | `"claude_agent_sdk"` | No | Yes | Yes | `"langchain"` |
| Anthropic tool-use with native schemas | `"claude_tool"` | No | Yes | Yes | `"langchain"` |
| Autonomous deep agent execution | `"langchain_deep_agents"` | Yes | Yes | Yes | None (explicit install required) |
| Gemini on Vertex AI or the Gemini API, with Google Search grounding | `"google_genai"` | No | Yes | Yes | None (intentional: a fallback would drop grounding and the search budget) |
| Pre-recorded/offline traces | `"manual"` | No | No | No | None (intentional) |
| Pre-collected output evaluation (TaskEval sentinel) | `"taskeval"` | No | No | No | None (no-op; never calls the LLM) |

**Agent tier**: `"claude_agent_sdk"` and `"langchain_deep_agents"` are registered with `agent_tier="deep_agent"`: full agent runtimes with built-in tools (the runtime handles tool loops internally, and `GenerateAnswer` prefers the `AgentPort` path to capture the full trace). Every other interface defaults to `agent_tier="tool_loop"` (the adapter orchestrates each tool-call turn explicitly). Use a `deep_agent`-tier interface for the parsing model when enabling `agentic_parsing=True`. For `AgenticRubricTrait`, a `deep_agent` judge handles any trait, and a `tool_loop` judge with MCP support (for example `"google_genai"`) handles `context_mode="trace_only"` traits without `materialize_trace`.

## When to Switch from LangChain

**Stay on `"langchain"` (default)** when:
- Using any major provider (Anthropic, OpenAI, Google) for standard benchmarks
- Running MCP tool-use evaluations through LangChain's agent framework
- You have no specific reason to change

**Switch to `"claude_tool"`** when:
- Evaluating tool-use with Anthropic models and you need native tool schema handling
- The LangChain tool integration does not support a specific tool pattern you need

**Switch to `"claude_agent_sdk"`** when:
- You need direct Anthropic API access without the LangChain abstraction layer
- Using Anthropic-specific features not exposed through LangChain
- Using agentic parsing (`agentic_parsing=True`) or agentic rubric traits (`AgenticRubricTrait`)

**Switch to `"langchain_deep_agents"`** when:
- Using agentic parsing (`agentic_parsing=True`) or agentic rubric traits
- You need the investigation agent to use tools autonomously to verify artifacts
- Preferred over `"claude_agent_sdk"` when you want LangChain-based tool ecosystem

**Agentic judges**: `agentic_parsing=True` is rejected at config validation unless every parsing model has `agent_tier="deep_agent"` (`"claude_agent_sdk"` or `"langchain_deep_agents"`). `AgenticRubricTrait` judges follow `AgenticRubricTrait.accepts_judge_interface()`: `deep_agent` interfaces judge any trait (they need file tools for workspaces and materialized traces), while `tool_loop` interfaces with MCP support (`"langchain"`, `"openrouter"`, `"openai_endpoint"`, `"claude_tool"`, `"google_genai"`) judge only `context_mode="trace_only"` traits with `materialize_trace=False`. Their tools are the MCP servers in the judge's `mcp_urls_dict` (plus server-side tools such as Gemini's Google Search). A judge that cannot serve a trait raises `ValueError` at the start of the agentic rubric stage.

**Switch to `"google_genai"`** when:
- Evaluating or judging with Gemini on Vertex AI (Google application default credentials, the default backend) or through a Gemini API gateway (`endpoint_base_url` selects the `gemini_api` backend)
- You need Google Search or URL reading (`extra_kwargs={"builtin_tools": ["google_search", "url_context"]}`, off by default) with the queries and sources recorded as `google_search` / `url_context` tool calls in agent traces (a single LLM call reports only the query count in usage)
- You want a per-run search cap (`max_search_queries`, a soft cap: Search is removed between model calls, so one call can exceed it) or thinking-token and search counts in usage (`reasoning_tokens`, `web_search_queries`)
- You need a Gemini judge for agentic rubric traits that calls MCP tools (karenina runs the tool loop, so MCP servers work)

Note that `url_context` only opens URLs written out in the request. To let a judge open pages it finds, give it a fetch tool through an MCP server.

**Switch to `"openai_endpoint"`** when:
- Connecting to self-hosted models (vLLM, Ollama, TGI)
- Using any OpenAI-compatible API endpoint
- Requires `endpoint_base_url` and `endpoint_api_key` on ModelConfig

**Switch to `"manual"`** when:
- Evaluating pre-recorded LLM outputs (TaskEval workflow)
- Running offline analysis without live LLM calls
- Requires `manual_traces` on ModelConfig

## Extra Configuration by Interface

### `"openai_endpoint"`

```python
ModelConfig(
    id="local-model",
    model_provider="openai",
    model_name="meta-llama/Llama-3-70b",
    interface="openai_endpoint",
    endpoint_base_url="http://localhost:8000/v1",
    endpoint_api_key=SecretStr("your-api-key"),
)
```

### `"claude_tool"` / `"claude_agent_sdk"`

```python
ModelConfig(
    id="claude-native",
    model_name="claude-sonnet-4-20250514",
    interface="claude_tool",  # or "claude_agent_sdk"
    # model_provider NOT required for these interfaces
    # Optional: override API endpoint/key
    anthropic_base_url="https://custom-proxy.example.com",
    anthropic_api_key=SecretStr("sk-ant-..."),
)
```

### `"google_genai"`

```python
ModelConfig(
    id="gemini-judge",
    model_name="gemini-3.8-flash",
    interface="google_genai",
    # model_provider NOT used. Default backend is Vertex AI with Google ADC.
    mcp_urls_dict={"literature": "http://localhost:8765/mcp"},  # optional MCP servers
    extra_kwargs={
        "project": "my-gcp-project",          # or GOOGLE_CLOUD_PROJECT
        "builtin_tools": ["google_search"],   # off by default
        "max_search_queries": 5,              # soft cap per agent run
    },
)
```

Unknown `extra_kwargs` keys raise `ValueError`. Accepted keys: `backend`, `project`, `location`, `builtin_tools`, `max_search_queries`, `thinking_level`, `thinking_budget`, `include_thoughts`, `stream`, `resolve_grounding_urls`, `generate_content_config`. Install with `pip install 'karenina[google-genai]'`. For the `gemini_api` backend set `endpoint_base_url` (and `endpoint_api_key`, or `GEMINI_API_KEY` / `GOOGLE_API_KEY`).

### `"manual"`

```python
from karenina.adapters.manual import ManualTraces

traces = ManualTraces(benchmark)  # benchmark is required (builds the question-text index)
traces.register_traces({"<question-id>": [...pre-recorded messages...]})
ModelConfig(
    interface="manual",
    manual_traces=traces,
    # id and model_name default to "manual"
)
```

### `"openrouter"`

```python
ModelConfig(
    id="openrouter-model",
    model_provider="openrouter",
    model_name="anthropic/claude-sonnet-4-20250514",
    interface="openrouter",
)
```

## External Adapters

Adapters can be registered via Python entry points in the `karenina.adapters` group. External adapters that conflict with built-in interface names are skipped with a warning.
