# MCP Integration Deep Dive

This page documents **how** adapters handle MCP servers internally — the connection lifecycle, tool schema transformation, agent execution loops, trace capture, and message conversion. For MCP **configuration and usage**, see the [MCP overview](../04-core-concepts/mcp-overview.md) and [MCP-enabled verification](../06-running-verification/mcp-verification.md).

---

## MCP Execution Flow

When `mcp_urls_dict` is set on a `ModelConfig`, the verification pipeline switches from simple LLM invocation to agent-based execution. The full MCP execution flow follows these steps:

```
1. Connect        Connect to each MCP server (HTTP/SSE or stdio)
       │
2. Discover       List available tools from each server
       │
3. Filter         Apply mcp_tool_filter to restrict tool set
       │
4. Transform      Convert MCP tools to adapter-native format
       │
5. Execute        Run agent loop (LLM call → tool use → tool result → repeat)
       │
6. Capture        Record trace (messages + tool results + usage)
       │
7. Convert        Transform adapter messages to unified Message format
       │
8. Return         AgentResult with final_response, raw_trace, trace_messages
```

Each adapter implements this flow differently based on its backend infrastructure.

---

## Core MCP Utilities

The `karenina.utils.mcp` module provides adapter-agnostic utilities for MCP server interaction:

### Session Management (`utils/mcp/client.py`)

| Function | Purpose |
|----------|---------|
| `connect_mcp_session(exit_stack, config)` | Connect to a single MCP server via HTTP/SSE. Registers cleanup with `AsyncExitStack`. |
| `connect_all_mcp_servers(exit_stack, mcp_servers)` | Connect to all configured servers in parallel. Returns `dict[str, ClientSession]`. |
| `get_all_mcp_tools(sessions)` | Call `list_tools()` on each session. Returns `list[tuple[server_name, session, mcp_tool]]`. |

These utilities use the core `mcp` Python package (`mcp.client.streamable_http.streamablehttp_client`) for HTTP transport. The Claude Tool adapter reuses these directly; other adapters have their own connection mechanisms.

### Tool Description Management (`utils/mcp/tools.py`)

| Function | Purpose |
|----------|---------|
| `afetch_tool_descriptions(mcp_urls_dict, tool_filter)` | Async: fetch tool names and descriptions from MCP servers. 30-second timeout per server. |
| `fetch_tool_descriptions(mcp_urls_dict, tool_filter)` | Sync wrapper with smart event loop detection — uses shared `BlockingPortal`, falls back to `ThreadPoolExecutor`, or `asyncio.run()`. 45-second overall timeout. |
| `apply_tool_description_overrides(tools, overrides)` | Modify tool descriptions in-place. Works with both LangChain Tool and MCP Tool objects. |

---

## Tool Schema Transformation

MCP servers expose tools with a standard schema. Each adapter converts these into its native tool format:

### MCP Tool Structure

An MCP tool has three fields:

- **`name`** — Unique tool identifier (e.g., `"web_search"`)
- **`description`** — Human-readable purpose text
- **`inputSchema`** — JSON Schema dict defining the tool's parameters

### Adapter-Specific Transformations

| Adapter | Transformation | Details |
|---------|---------------|---------|
| **LangChain** | MCP tool → LangChain Tool | Via `langchain-mcp-adapters` `MultiServerMCPClient.get_tools()`. Returns LangChain Tool objects for direct use in LangGraph agents. |
| **Claude Tool** | MCP tool → `@beta_async_tool` function | `wrap_mcp_tool()` creates an async function decorated with Anthropic's `@beta_async_tool`. The function calls `session.call_tool(tool.name, kwargs)` and records results in a `ToolResultCollector`. |
| **Claude Agent SDK** | MCP tool → SDK-native tools | The SDK discovers and manages tools natively via `mcp_servers` in `ClaudeAgentOptions`. Tools use `mcp__{server_name}__{tool_name}` naming. |
| **Manual** | N/A | No MCP support. Raises `ValueError` if `mcp_urls_dict` is set. |

### Tool Filtering and Description Overrides

Tool filtering and description overrides apply at steps 3-4 of the execution flow:

1. **Filtering** — Only tools whose names appear in `mcp_tool_filter` are passed to the agent. Applied by each adapter before creating the agent.
2. **Overrides** — Tool descriptions are replaced using `apply_tool_description_overrides()` before the agent sees them. Useful for improving tool selection accuracy.

---

## Adapter-Specific MCP Strategies

### LangChain Adapter

**Transport**: HTTP/SSE only (URL-based, via `langchain-mcp-adapters`)

**Key classes**:

- `LangChainAgentAdapter` — The AgentPort implementation
- `MultiServerMCPClient` — From langchain-mcp-adapters, manages MCP connections

**Connection flow**:

1. `_convert_mcp_servers_to_urls()` converts `MCPServerConfig` dicts to URL strings. Raises `AgentExecutionError` for stdio transport (not supported by LangChain's URL-based infrastructure).
2. `acreate_mcp_client_and_tools()` creates a `MultiServerMCPClient` with streamable HTTP config, fetches tools with 30-second timeout, applies filter and description overrides.
3. Returns `(client, tools)` where tools are LangChain Tool objects.

**Agent execution**:

1. `_create_agent()` initializes the LLM via `init_chat_model_unified()`, loads tools, builds middleware stack via `build_agent_middleware()`, and creates a LangGraph agent with `InMemorySaver` checkpointer.
2. `arun()` converts messages to LangChain format, invokes the agent via `agent.ainvoke()`, and tracks token usage via `get_usage_metadata_callback()`.
3. On `GraphRecursionError`, extracts partial state from the checkpointer via `extract_partial_agent_state()` — the `InMemorySaver` enables recovering the conversation so far even when the limit is hit.

**Trace capture**: Messages are collected from the agent response dict (`{"messages": [...]}`), converted back to unified format via `LangChainMessageConverter.from_provider()`, and formatted into `raw_trace` via `harmonize_agent_response()`.

**Cleanup**: `cleanup_mcp_client()` attempts multiple strategies — `client.close()`, `client.aclose()`, `client.__exit__()` — with warning if no method found.

**Unique features**:

- Middleware stack: summarization, prompt caching, retry logic via `AgentMiddlewareConfig`
- Partial state recovery on recursion limit via `InMemorySaver` checkpointer
- Multi-provider support (any LangChain-compatible LLM)

### Claude Agent SDK Adapter

**Transport**: HTTP/SSE and stdio (native SDK support)

**Key classes**:

- `ClaudeSDKAgentAdapter` — The AgentPort implementation
- `ClaudeSDKClient` — Wrapper around the Claude Agent SDK

**MCP config conversion** (`mcp.py`):

`convert_mcp_config()` translates karenina's simplified format to SDK-native format:

| Input Format | Detection | Output Type |
|-------------|-----------|-------------|
| URL (`http://` or `https://`) | Starts with `http` | `{"type": "http", "url": ...}` |
| Command with args (whitespace-separated) | Contains whitespace | `{"command": cmd, "args": [...]}` |
| Single path | Default | `{"command": path}` |

`validate_mcp_config()` checks that HTTP configs have `type="http"` and `url`, stdio configs have `command`. `convert_and_validate_mcp_config()` combines both, raising `McpConfigValidationError` on failure.

!!! note
    HTTP URLs map to `type="http"` (streamable HTTP), not `"sse"`. The Claude Agent SDK handles the transport protocol internally.

**Agent execution**:

1. `_convert_mcp_servers()` handles both karenina format and pre-converted SDK format (detects by checking for `"type"` or `"command"` keys).
2. `_build_options()` creates `ClaudeAgentOptions` with `permission_mode="bypassPermissions"`, configured max_turns, and MCP servers. When MCP servers are present, it restricts available tools to only MCP tools via `allowed_tools` list, using the naming convention `mcp__{server_name}__{tool_name}`.
3. `arun()` extracts a prompt string (user messages joined) and system prompt, then uses `ClaudeSDKClient` as an async context manager:

```
async with ClaudeSDKClient(options) as client:
    await client.query(prompt_string)
    async for msg in client.receive_response():
        collected_messages.append(msg)
```

4. Limit detection via `ResultMessage.subtype` field check.

**Trace capture**: All messages from `client.receive_response()` (UserMessage, AssistantMessage, ResultMessage) are collected, then converted via `ClaudeSDKMessageConverter.from_provider()` and formatted into `raw_trace` via `sdk_messages_to_raw_trace()`.

**Unique features**:

- **Stdio transport** — The only adapter supporting stdio-based MCP servers (local commands like `npx @mcp/server-github`)
- Native SDK MCP management — the SDK handles connection lifecycle, tool discovery, and execution internally
- Session ID from `ResultMessage` — enables tracing back to specific SDK sessions

### Claude Tool Adapter

**Transport**: HTTP/SSE only (via core `mcp` Python package)

**Key classes**:

- `ClaudeToolAgentAdapter` — The AgentPort implementation
- `ToolResultCollector` — Records tool execution results for trace reconstruction
- Uses Anthropic SDK's `client.beta.messages.tool_runner()` for the agent loop

**Connection flow**:

1. Creates `AsyncExitStack` for resource management.
2. Calls `connect_all_mcp_servers(exit_stack, mcp_servers)` from `utils/mcp/client.py` (or the adapter's own `mcp.py` which reimplements the same functions).
3. Calls `get_all_mcp_tools(sessions)` to discover tools.
4. Each tool is wrapped with `wrap_mcp_tool()`:
   - Creates an async function decorated with `@beta_async_tool`
   - The function calls `session.call_tool(mcp_tool.name, kwargs)` and extracts text content from the result
   - Results are recorded in a `ToolResultCollector` for trace reconstruction

**Agent execution**:

1. Static tools (karenina `Tool` objects) are wrapped via `wrap_static_tool()` with the same collector pattern.
2. `apply_cache_control_to_tool()` adds Anthropic `cache_control` to the last tool's schema, enabling prompt caching.
3. `_execute_agent_loop()` converts messages to Anthropic format, creates a `tool_runner` via `client.beta.messages.tool_runner()`, and iterates over responses:
   - For each response: increment turn counter, drain `ToolResultCollector` for recorded results, match with previous `tool_use` blocks, inject as `tool_result` messages, convert to unified Message format, aggregate usage, check turn limit.

**Trace capture**: The `ToolResultCollector` is central to trace reconstruction:

```python
class ToolResultCollector:
    def record(tool_name, content, is_error=False)  # Capture result
    def drain() -> list[ToolResultRecord]            # Get and clear
```

Each `ToolResultRecord` holds `tool_name`, `content`, and `is_error`. After each LLM response that includes `tool_use` blocks, the collector is drained and results are matched to the corresponding tool calls.

**Unique features**:

- **Native prompt caching** — `cache_control` with `ephemeral` type added to tool schemas, enabling Anthropic's prompt caching for efficiency
- **Direct API** — No framework overhead; uses Anthropic Python SDK directly
- `tool_runner` handles the multi-turn loop automatically

### Manual Adapter

MCP is **not supported** with the manual interface. Setting `mcp_urls_dict` on a `ModelConfig` with `interface="manual"` raises a `ValueError`. The manual adapter replays pre-recorded traces, so live tool access is not applicable.

---

## Trace Capture and Message Conversion

Each adapter captures traces in two formats:

1. **`raw_trace`** (string) — Legacy format with delimiter-separated messages for backward compatibility
2. **`trace_messages`** (list[Message]) — Structured format using karenina's unified Message type

### Unified Message Format

```python
@dataclass
class Message:
    role: Role           # SYSTEM, USER, ASSISTANT, TOOL
    content: list[Content]  # TextContent, ToolUseContent, ToolResultContent, ThinkingContent
```

Content block types:

| Type | Fields | Used For |
|------|--------|----------|
| `TextContent` | `text` | Plain text responses |
| `ToolUseContent` | `id`, `name`, `input` | Tool invocation requests |
| `ToolResultContent` | `tool_use_id`, `content`, `is_error` | Tool execution results |
| `ThinkingContent` | `thinking` | Extended thinking blocks (Anthropic) |

### Adapter Message Conversion

Each adapter has a message converter that translates between its native format and unified Messages:

**LangChain** (`LangChainMessageConverter`):

| LangChain Type | Unified Role | Notes |
|----------------|-------------|-------|
| `SystemMessage` | `SYSTEM` | System prompt |
| `HumanMessage` | `USER` | User input |
| `AIMessage` | `ASSISTANT` | LLM response, with optional `ToolUseContent` from `tool_calls` |
| `ToolMessage` | `TOOL` | Tool results via `ToolResultContent` |

Bidirectional: `to_provider()` (unified → LangChain) and `from_provider()` (LangChain → unified).

**Claude Tool** (Anthropic SDK format):

| Anthropic Format | Unified Role | Notes |
|-----------------|-------------|-------|
| `{"role": "assistant", "content": [...]}` | `ASSISTANT` | Content blocks: text, tool_use, thinking |
| `{"role": "user", "content": [...]}` | `USER` | Anthropic convention: tool results sent in user messages |

Functions: `convert_to_anthropic()` (unified → Anthropic dict) and `convert_from_anthropic_message()` (Anthropic response → unified Message).

**Claude Agent SDK** (SDK message types):

| SDK Type | Unified Role | Notes |
|----------|-------------|-------|
| `UserMessage` | `USER` | Input messages |
| `AssistantMessage` | `ASSISTANT` | With TextBlock, ToolUseBlock, ThinkingBlock content |
| `ResultMessage` | `ASSISTANT` | Final response with subtype for limit detection |

Input conversion uses `to_prompt_string()` (extracts user messages as joined text) and `extract_system_prompt()` (extracts system messages). Output conversion handles SDK-specific block types.

### Raw Trace Format

The `raw_trace` string is a legacy format that concatenates messages with delimiters. Each adapter has its own formatter:

- **LangChain**: `harmonize_agent_response()` — formats from LangChain message list
- **Claude Tool**: `claude_tool_messages_to_raw_trace()` — formats from unified messages
- **Claude Agent SDK**: `sdk_messages_to_raw_trace()` — formats from SDK message list

The verification pipeline uses `trace_messages` (structured) by default. The `raw_trace` is available for debugging and backward compatibility.

---

## Transport Protocol Comparison

| Transport | LangChain | Claude Agent SDK | Claude Tool |
|-----------|-----------|-----------------|-------------|
| **HTTP/SSE** | Via langchain-mcp-adapters `MultiServerMCPClient` | Native SDK support (`type="http"`) | Via core `mcp` package `streamablehttp_client` |
| **Stdio** | Not supported (URL-based only) | Native SDK support (`command` + `args`) | Not supported (HTTP/SSE only) |
| **Connection library** | langchain-mcp-adapters | Claude Agent SDK | mcp (Python package) |
| **Cleanup** | `cleanup_mcp_client()` with fallback strategies | SDK-managed (context manager) | `AsyncExitStack` with registered cleanup |

---

## Recursion Limit Handling

Each adapter handles the agent exceeding its turn limit differently:

| Adapter | Mechanism | Behavior |
|---------|-----------|----------|
| **LangChain** | Catches `GraphRecursionError` | Extracts partial state from `InMemorySaver` checkpointer. Returns what the agent produced so far. |
| **Claude Tool** | Turn counter in `_execute_agent_loop()` | Compares `turn` count against `config.max_turns`. Breaks the loop and returns partial results. |
| **Claude Agent SDK** | `ResultMessage.subtype` field | The SDK signals limit reached via the result message. Sets `limit_reached=True` on `AgentResult`. |

In all cases, the pipeline's **RecursionLimitAutoFail** stage (Stage 3) checks `AgentResult.limit_reached` and marks the verification as auto-failed if the limit was hit, while preserving the captured trace for inspection.

---

## Key Source Files

| File | Purpose |
|------|---------|
| `utils/mcp/client.py` | Adapter-agnostic MCP session management |
| `utils/mcp/tools.py` | Tool description fetching and override |
| `ports/agent.py` | AgentPort protocol, AgentConfig, AgentResult, MCPServerConfig types |
| `ports/messages.py` | Unified Message type and content blocks |
| `adapters/langchain/mcp.py` | LangChain MCP client and tool loading |
| `adapters/langchain/agent.py` | LangChain agent adapter with LangGraph |
| `adapters/claude_agent_sdk/mcp.py` | Claude SDK MCP config conversion |
| `adapters/claude_agent_sdk/agent.py` | Claude SDK agent adapter |
| `adapters/claude_tool/mcp.py` | Claude Tool MCP session management |
| `adapters/claude_tool/agent.py` | Claude Tool agent adapter with tool_runner |
| `adapters/claude_tool/tools.py` | Tool wrapping and ToolResultCollector |
| `adapters/factory.py` | Agent factory routing |

---

## Related

- [MCP overview](../04-core-concepts/mcp-overview.md) — Configuration and when to use MCP
- [MCP-enabled verification](../06-running-verification/mcp-verification.md) — Running MCP verification workflows
- [Adapter architecture](index.md) — Hexagonal architecture and port/adapter pattern
- [Port types](ports.md) — Complete AgentPort protocol reference
- [Available adapters](available-adapters.md) — Per-adapter features and configuration
