# MCP Integration

**Model Context Protocol (MCP)** is a standardized protocol that enables LLMs to access external tools and data sources during verification. When an answering model has MCP access, it can invoke tools — web search, database queries, API calls, file operations — instead of relying solely on its training data.

## Why Use MCP in Verification?

Standard verification sends a question to an LLM and evaluates the response. MCP-enabled verification gives the LLM **tool access**, turning it into an agent that can gather information before answering.

**Use cases**:

- **Current information** — Search the web for recent drug approvals or regulatory changes
- **Database access** — Query genomics databases for gene annotations
- **API integration** — Call external services for real-time data
- **File operations** — Read data files or configuration
- **Custom tools** — Domain-specific tools for specialized benchmarks

**Example**: A benchmark question asks "What is the current FDA approval status of drug X?" With MCP, the LLM can search for the latest information rather than relying on its training data cutoff.

## When to Use MCP vs Simple LLM

| Scenario | Recommendation |
|----------|---------------|
| Questions have definitive answers from training data | Simple LLM (no MCP) |
| Questions require current or real-time information | MCP with web search |
| Questions need data from specific databases | MCP with database tools |
| Benchmark tests tool usage ability itself | MCP required |
| Reproducibility is the top priority | Simple LLM or [manual interface](manual-interface.md) |
| Cost and latency must be minimized | Simple LLM (MCP adds overhead) |

MCP verification takes longer and costs more (multiple LLM calls per question due to tool use loops), so use it only when tool access is needed.

## Architecture

MCP integration involves three components:

```
                          ┌─────────────────┐
                          │   MCP Servers    │
                          │  (external tools)│
                          └────────┬─────────┘
                                   │ tool calls
                                   ▼
┌──────────┐    question    ┌─────────────┐    parsed answer    ┌──────────┐
│ Karenina │ ──────────────►│  Agent Port │ ──────────────────► │ Pipeline │
│          │                │  (adapter)  │                     │ (verify) │
└──────────┘                └─────────────┘                     └──────────┘
                                   │
                              multi-turn
                              agent loop
```

1. **MCP Servers** — External processes that expose tools via the MCP protocol (HTTP or stdio transport)
2. **Agent Port** — The adapter's `AgentPort` implementation connects to servers, discovers tools, and runs the agent loop
3. **Agent Middleware** — Configurable retry logic, execution limits, conversation summarization, and prompt caching

The agent loop runs multiple LLM calls: the model generates a response, optionally invokes tools, receives tool results, and continues until it produces a final answer or hits a limit.

## Configuration

MCP is configured on `ModelConfig`, not `VerificationConfig`. Each answering model can have its own MCP server configuration:

```python
from karenina.schemas import ModelConfig

model_config = ModelConfig(
    id="agent-claude",
    model_name="claude-sonnet-4-5-20250929",
    model_provider="anthropic",
    interface="langchain",
    mcp_urls_dict={
        "biocontext": "https://mcp.biocontext.ai/mcp/",
        "web_search": "https://search-server.example.com/mcp/",
    },
)
```

### Key MCP Fields on ModelConfig

| Field | Type | Description |
|-------|------|-------------|
| `mcp_urls_dict` | `dict[str, str] \| None` | Map of server names to URLs. Setting this enables MCP agent mode. |
| `mcp_tool_filter` | `list[str] \| None` | Restrict which tools the agent can use (by tool name). If `None`, all discovered tools are available. |
| `mcp_tool_description_overrides` | `dict[str, str] \| None` | Override tool descriptions sent to the LLM (useful for optimizing tool selection). |
| `max_context_tokens` | `int \| None` | Token threshold for triggering conversation summarization. |
| `agent_middleware` | `AgentMiddlewareConfig \| None` | Middleware stack controlling retry, limits, summarization, and caching. |

### Agent Middleware

The `AgentMiddlewareConfig` controls agent execution behavior. It only applies when `mcp_urls_dict` is set.

| Component | Config Class | Key Settings | Defaults |
|-----------|-------------|-------------|----------|
| **Execution limits** | `AgentLimitConfig` | `model_call_limit`, `tool_call_limit`, `exit_behavior` | 25 model calls, 50 tool calls, `"end"` |
| **Model retry** | `ModelRetryConfig` | `max_retries`, `backoff_factor`, `on_failure` | 2 retries, 2.0x backoff, `"continue"` |
| **Tool retry** | `ToolRetryConfig` | `max_retries`, `backoff_factor`, `on_failure` | 3 retries, 2.0x backoff, `"return_message"` |
| **Summarization** | `SummarizationConfig` | `enabled`, `trigger_fraction`, `keep_messages` | Enabled, 80% threshold, keep 20 messages |
| **Prompt caching** | `PromptCachingConfig` | `enabled`, `ttl`, `unsupported_model_behavior` | Enabled, 5 min TTL, `"warn"` for non-Anthropic |

**Summarization** automatically condenses conversation history when it approaches the context window limit, preventing failures on long multi-turn interactions.

**Prompt caching** reduces costs and latency for Anthropic models by caching static content (system prompts, tool definitions) on Anthropic's servers. Only applies to Anthropic models with the `langchain` interface.

## Adapter-Specific MCP Handling

Each adapter connects to MCP servers differently:

| Feature | `langchain` | `claude_agent_sdk` | `claude_tool` |
|---------|:-----------:|:------------------:|:-------------:|
| **Transport** | HTTP/SSE | HTTP/SSE + Stdio | HTTP/SSE |
| **MCP library** | `langchain-mcp-adapters` | Anthropic Agent SDK | `mcp` Python SDK |
| **Tool filtering** | Yes | Yes | Yes |
| **Description overrides** | Yes | Yes | Yes |
| **Prompt caching middleware** | Yes (Anthropic only) | No | No |
| **Summarization middleware** | Yes (configurable) | Built-in | Built-in |
| **Retry middleware** | Yes (configurable) | Yes | Yes |
| **Stdio servers** | No | Yes | No |
| **Requires CLI** | No | Yes (Claude Code) | No |

### `langchain` — Middleware-Based

The default adapter uses `langchain-mcp-adapters` to connect to MCP servers over HTTP. Tools are wrapped as LangChain `Tool` objects and passed to a LangGraph agent. The full middleware stack (retry, limits, summarization, prompt caching) is configurable via `AgentMiddlewareConfig`.

### `claude_agent_sdk` — Native Anthropic

Uses Anthropic's Agent SDK with native MCP session support. The only adapter that supports **stdio transport** (local subprocess MCP servers). Requires Claude Code CLI installed.

### `claude_tool` — Direct SDK

Uses the `mcp` Python SDK to connect to HTTP/SSE servers directly. Session lifecycle is managed automatically via `AsyncExitStack` with cleanup on completion.

### `openrouter` / `openai_endpoint` — Via LangChain

These routing interfaces delegate to the `langchain` adapter, so they inherit the same MCP behavior.

### `manual` — No MCP

The manual interface uses pre-recorded traces and **does not support MCP**. Configuring `mcp_urls_dict` on a manual model raises a `ValueError`.

## MCP Execution Flow

When verification runs with MCP enabled:

1. **Connect** — Adapter connects to MCP servers listed in `mcp_urls_dict`
2. **Discover tools** — Available tools are fetched from each server
3. **Filter** — If `mcp_tool_filter` is set, only matching tools are exposed to the LLM
4. **Run agent loop** — The LLM receives the question plus available tools and runs a multi-turn loop (generate → call tools → receive results → continue)
5. **Apply middleware** — Retry logic handles transient failures; summarization condenses long conversations; limits cap execution
6. **Capture trace** — All messages (assistant responses + tool calls + tool results) are captured as `trace_messages`
7. **Return result** — Final response, full trace, and usage metadata are returned for pipeline evaluation

## Trace Handling

After agent execution, the trace is available in two formats:

- **`trace_messages`** — Structured list of `Message` objects (assistant and tool messages only, excluding user messages)
- **`raw_trace`** — Legacy string serialization of all messages

The verification pipeline can evaluate either the **final response** or the **full trace**. This is controlled by two `VerificationConfig` fields:

- `use_full_trace_for_template` — Use the complete trace (not just the final answer) for template parsing
- `use_full_trace_for_rubric` — Use the complete trace for rubric evaluation

## Next Steps

- [Running MCP-Enabled Verification](../06-running-verification/mcp-verification.md) — Step-by-step workflow with configuration examples
- [Adapters](adapters.md) — Full adapter comparison and configuration
- [Evaluation Modes](evaluation-modes.md) — How MCP interacts with template and rubric evaluation
- [Manual Interface](manual-interface.md) — Alternative for reproducible testing without live tools
- [VerificationConfig Reference](../10-configuration-reference/verification-config.md) — Trace handling and MCP-related config fields
