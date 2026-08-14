---
name: karenina-adapter-implement
description: Implement a new karenina adapter following the design spec. Guided file-by-file code generation with conventions enforcement. Use as Phase 3 of adapter creation.
---

# Phase 3: Implement Karenina Adapter

Implement the adapter package following the design spec from Phase 2.

## Prerequisites

- Design spec from `/karenina-adapter-design`
- Optional dependencies added to `pyproject.toml`

## Before you begin

1. Invoke the `using-karenina` skill and read its `references/advanced-adapters/` directory for adapter domain context
2. Read the design spec for this adapter
3. Read `CLAUDE.md` in the karenina submodule for coding conventions
4. **Read the port protocol files** (these define the exact interfaces you must implement):
   - `karenina/src/karenina/ports/agent.py` (AgentPort, AgentConfig, AgentResult including `timeout_reached`, Tool, MCPServerConfig)
   - `karenina/src/karenina/ports/llm.py` (LLMPort, LLMResponse with `is_partial` and `usage_unavailable`, StreamingLLMResponse)
   - `karenina/src/karenina/ports/capabilities.py` (PortCapabilities including `supports_streaming`)
   - `karenina/src/karenina/ports/parser.py` (ParserPort, ParsePortResult)
   - `karenina/src/karenina/ports/messages.py` (Message, Role, TextContent, ToolUseContent, ToolResultContent)
   - `karenina/src/karenina/ports/usage.py` (UsageMetadata)
5. Read the reference adapter that most closely matches your design:
   - Natively agentic: `karenina/src/karenina/adapters/claude_agent_sdk/`
   - Scaffolded: `karenina/src/karenina/adapters/langchain/`

## Implementation order

Implement files in this order (dependencies flow downward):

1. **`__init__.py`**: Lazy exports with `__getattr__`, full `__all__`. See [adapter-file-template.md](adapter-file-template.md).
2. **`availability.py`**: Import check for the SDK package. Return `AdapterAvailability`.
3. **`errors.py`**: Map SDK exceptions to `AgentExecutionError`, `AgentTimeoutError`, `AgentResponseError`.
4. **`initialization.py`** (if needed): Model creation via `init_chat_model` or SDK-specific setup.
5. **`messages.py`**: Bidirectional conversion (karenina `Message` <-> SDK message types).
   **Common pitfall**: karenina's field names differ from most SDKs. Read the actual dataclass fields in `ports/messages.py` before writing conversion code. In particular:
   - `ToolUseContent` uses `id` (not `tool_call_id`)
   - `ToolResultContent` uses `tool_use_id` (not `tool_call_id`)
   These names are easy to get wrong when translating from SDKs that use `tool_call_id` for both.
6. **`mcp.py`** (if applicable): Convert `MCPServerConfig` to SDK-compatible tool definitions.
7. **`trace.py`**: Extract `raw_trace` (delimited string) and `trace_messages` (structured list).
8. **`usage.py`**: Extract `UsageMetadata` (input_tokens, output_tokens, total_tokens, model).
9. **`agent.py`**: Core `arun()` implementation integrating all above modules.
10. **`llm.py`**: Single-turn LLM calls (simpler than agent).
11. **`parser.py`**: Structured output parsing via `aparse_to_pydantic()`.
12. **`registration.py`**: `AdapterSpec` + `AdapterRegistry.register()`.
13. **`prompts/`**: Adapter instruction registration (parsing, rubric, deep_judgment).

Adapter files are tightly interdependent (agent.py imports messages.py, trace.py, usage.py, errors.py). Write implementation files first, then write cold tests for the complete adapter. Run tests after each logical group of files.

## Required protocol methods

Every adapter must implement `aclose()` on all three port classes (AgentPort, LLMPort, ParserPort). This is a required protocol method, not optional. If the adapter holds no resources, implement it as a no-op:

```python
async def aclose(self) -> None:
    """Release adapter resources."""
```

For adapters that hold MCP sessions, SDK clients, or other resources, `aclose()` must clean them up. Use `AsyncExitStack` for MCP session management (see the MCP session pattern below).

Every adapter must also implement a `capabilities` property on all three port classes:

```python
@property
def capabilities(self) -> PortCapabilities:
    return PortCapabilities(
        supports_system_prompt=True,
        supports_structured_output=False,
        supports_streaming=False,
    )
```

`supports_streaming` defaults to `False`. Set it to `True` only if the LLM adapter implements `astream()` and `stream_invoke()` with real chunked output. When declared, the pipeline may route calls through the streaming path to capture partial content on request timeouts.

## Streaming on LLMPort

The LLMPort protocol exposes two streaming methods that every LLM adapter must define, even if only to raise `NotImplementedError`:

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from karenina.ports.llm import StreamingLLMResponse

@asynccontextmanager
async def astream(self, messages: list[Message]) -> AsyncIterator[StreamingLLMResponse]:
    """Open a streaming LLM connection.

    Yields a StreamingLLMResponse whose async iteration produces text chunks
    and whose ``accumulated_content`` holds whatever arrived so far.
    """
    sdk_messages = self._converter.to_provider(messages)
    response = StreamingLLMResponse()

    async def _chunk_generator() -> AsyncIterator[str]:
        async for chunk in sdk_stream(sdk_messages):
            text = extract_text(chunk)
            if text:
                yield text
            # Capture usage from whichever chunk carries it (often the final one).
            if has_usage(chunk):
                response.usage = extract_usage_from_chunk(chunk, model_name=self._config.model_name)

    response._set_chunk_source(_chunk_generator())
    yield response
    response.is_complete = True

def stream_invoke(self, messages: list[Message], timeout: float | None = None) -> LLMResponse:
    """Sync wrapper: stream with optional wall-clock timeout, returning an LLMResponse."""
    # Delegate to an async helper that wraps astream() in asyncio.timeout(timeout),
    # then route through get_async_portal() / ThreadPoolExecutor / asyncio.run()
    # using the same pattern as invoke(). See LangChainLLMAdapter.stream_invoke for
    # a reference implementation.
```

The async timeout helper that `stream_invoke()` delegates to must capture partial content:

```python
async def _astream_with_timeout(self, messages: list[Message], timeout: float | None) -> LLMResponse:
    is_partial = False
    async with self.astream(messages) as sr:
        try:
            async with asyncio.timeout(timeout):
                async for _chunk in sr:
                    pass
        except TimeoutError:
            is_partial = True
            logger.warning(
                "Streaming timeout after %ss: captured %d chars of partial response",
                timeout,
                len(sr.accumulated_content),
            )

    return LLMResponse(
        content=sr.accumulated_content,
        usage=sr.usage,
        raw=None,
        is_partial=is_partial,
        usage_unavailable=is_partial,
    )
```

When the stream is interrupted, both `is_partial` and `usage_unavailable` must be set. `usage_unavailable=True` is a signal to downstream stages that the token counts on this response do not reflect real consumption (the final chunk that carried them never arrived), so cost reporting must not treat the zeros as authoritative. If the SDK has no streaming API, implement both methods as:

```python
async def astream(self, messages: list[Message]) -> AsyncIterator[StreamingLLMResponse]:
    raise NotImplementedError(f"{type(self).__name__} does not support streaming")

def stream_invoke(self, messages: list[Message], timeout: float | None = None) -> LLMResponse:
    raise NotImplementedError(f"{type(self).__name__} does not support streaming")
```

Then leave `capabilities.supports_streaming=False`.

## MCP session management with AsyncExitStack

Adapters that connect to MCP servers must use `AsyncExitStack` to manage session lifetimes. Never create MCP sessions inside an `async with` block that closes before the tools are used.

```python
# Correct: AsyncExitStack keeps sessions alive for the agent's lifetime
self._exit_stack = AsyncExitStack()
sessions = await connect_all_mcp_servers(self._exit_stack, mcp_servers)
tools = await get_all_mcp_tools(sessions)
# ... use tools in agent loop ...
# sessions stay alive until aclose() calls self._exit_stack.aclose()

# Wrong: sessions close before the agent can use them
async with AsyncExitStack() as stack:
    sessions = await connect_all_mcp_servers(stack, mcp_servers)
    tools = await get_all_mcp_tools(sessions)
# sessions are closed here, but tools still reference them
```

Clean up the exit stack in `aclose()`:

```python
async def aclose(self) -> None:
    if self._exit_stack:
        await self._exit_stack.aclose()
```

## max_retries warning

Not all adapters support the `max_retries` parameter on `with_structured_output()`. If your adapter does not support it, emit a warning when it is passed:

```python
def with_structured_output(
    self, schema: type[BaseModel], *, max_retries: int | None = None
) -> "MyProviderLLMAdapter":
    if max_retries is not None:
        logger.warning(
            "%s does not support max_retries (got %d), ignoring",
            type(self).__name__, max_retries,
        )
    # ... rest of implementation ...
```

## Critical implementation pitfalls

These are bugs discovered during hot testing across all four existing adapters. Each was silent (adapter appeared to work) until specific test scenarios revealed it. Three of these were found in 3 out of 4 adapters, so they are not edge cases.

**1. Structured output MUST serialize to JSON (llm.py, parser.py)**
When `with_structured_output()` is configured and the framework returns a Pydantic model or dict, the adapter must serialize it to valid JSON. Using `str()` on a Pydantic model produces Python repr (e.g., `YesNo(answer=True, confidence=0.9)`), not valid JSON. Callers do `json.loads(response.content)` and crash. This bug was found in 3 out of 4 adapters.

```python
# CORRECT
if isinstance(response, BaseModel):
    content = response.model_dump_json()
elif isinstance(response, dict):
    content = json.dumps(response)

# WRONG (produces "MyModel(field='value')", not parseable JSON)
content = str(response)
```

**2. Turn limits MUST be wired to the framework (agent.py)**
Accepting `AgentConfig.max_turns` is not enough; the adapter must map it to the SDK's actual limit mechanism. Without this, the agent runs indefinitely regardless of the configured limit. Common mappings:
- LangGraph: `recursion_limit = max_turns * 2` (each tool call + response = 2 steps)
- Anthropic SDK tool_runner: check `turns >= config.max_turns` after each response
- Claude CLI: pass `max_turns` via agent options
- Custom frameworks: verify which parameter controls iteration count

**3. No-tools fallback path (agent.py)**
Every agent adapter must handle the case where `tools=None` and `mcp_servers=None`. Without this, adapters crash when called without tools (found in 2 out of 4 adapters). Options:
- Fall back to single-turn `messages.create()` (Claude Tool approach)
- Use built-in tools (Deep Agents approach with FilesystemBackend)
- Fall back to simple chat completion (no tool loop)

The fallback path must still:
- Honor `config.timeout` (wrap in `asyncio.wait_for()`)
- Return a valid `AgentResult` with traces, usage, turns
- Set `limit_reached=False` (single turn, no limit to hit)

**4. Timeout must apply to ALL execution paths (agent.py)**
If the adapter has multiple execution paths (e.g., tool_runner loop vs single-turn create), ALL paths must wrap their API call in `asyncio.wait_for()` when `config.timeout` is set. A common bug: the no-tools fallback bypasses the timeout wrapper entirely.

```python
# CORRECT: both paths honor timeout
if tools:
    coro = self._run_with_tools(messages, tools, config)
else:
    coro = self._run_single_turn(messages, config)

if config.timeout:
    result = await asyncio.wait_for(coro, timeout=config.timeout)
else:
    result = await coro
```

**5. HTTP request timeout from ModelConfig (llm.py, agent.py)**
`ModelConfig.request_timeout` (float or None) controls the per-request HTTP timeout for LLM API calls. The pipeline stamps this from `VerificationConfig.request_timeout` (default 120s). Every adapter that creates an HTTP client must pass it through:
- LangChain adapters: pass `request_timeout=self._config.request_timeout` as a kwarg to `init_chat_model` or `ChatOpenAI`
- Anthropic SDK adapters: pass `timeout=self._config.request_timeout` to `Anthropic()` / `AsyncAnthropic()`
- Other SDKs: find the equivalent client-level timeout parameter

If `request_timeout` is None, omit it (use the SDK default). This is separate from `config.timeout` (agent execution timeout) which limits the overall agent loop.

**6. Virtual filesystem backend (agent.py)**
If the SDK defaults to a virtual/in-memory filesystem, the agent's built-in tools (`ls`, `read_file`, etc.) return empty results for real paths. The agent compensates by generating synthetic data, producing plausible-looking but incorrect results. Fix: always configure a real filesystem backend, especially when `workspace_path` is set.

**7. Structured output discards usage metadata (parser.py)**
Many SDKs' `with_structured_output()` returns only the parsed dict/model, discarding the raw LLM response where token counts live. This causes `ParserPort` to return `usage.total_tokens=0`. Fix: use `include_raw=True` (or the SDK's equivalent) to get both the parsed output and the raw response with usage metadata.

**8. Recursion limit with empty state (agent.py)**
When the agent hits its turn/recursion limit, some SDKs raise an exception before populating the result state. If the adapter only handles exceptions by re-raising, it produces an unexpected error instead of a partial `AgentResult(limit_reached=True)`. Fix: catch limit exceptions, attempt to recover partial state, and return a partial result. **If no messages were collected at all** (the SDK raised before producing any output), return a minimal `AgentResult(limit_reached=True, final_response="[Recursion limit reached]", trace_messages=[], turns=0)` rather than raising `AgentResponseError`. This case is common when `max_turns` is very low (e.g., 1).

**8b. Agent wall-clock timeout partial recovery (agent.py)**
Wall-clock timeouts follow the same principle as recursion-limit recovery but use a different `AgentResult` field. When `config.timeout` fires mid-run, prefer returning a partial `AgentResult(timeout_reached=True, ...)` over raising `AgentTimeoutError`. Concretely:

- Accumulate messages, tool calls, and usage into local variables **as the loop runs**, not only at the end. This way, if cancellation interrupts the SDK call, the partial state is still in adapter memory.
- Wrap the loop in `try: await asyncio.wait_for(...)` / `except TimeoutError:` inside `arun()`. On `TimeoutError`, set a local `timeout_reached = True` flag and fall through to the normal result-construction code path.
- If at least one assistant message (or tool call) was collected, build and return `AgentResult(..., timeout_reached=True)`, appending a marker like `"[Note: Agent timed out - partial response shown]"` to `raw_trace` so downstream display can distinguish it.
- If **no** messages were collected at all (the timeout fired before the first turn), raise `AgentTimeoutError` with a message like `"timed out ... with no messages"`. There is nothing to salvage and the pipeline needs a clear error.
- Aggregate `usage` across whatever turns completed before the cutoff so cost reporting for the partial run is accurate.

The claude_tool and langchain adapters follow this pattern (see `adapters/claude_tool/agent.py` and `adapters/langchain/agent.py`). The claude_agent_sdk and langchain_deep_agents adapters currently raise on timeout because their SDKs do not expose incremental state; adapter authors choosing to raise must still set `usage_unavailable=True` on any partial `LLMResponse` created by the LLM path.

The pipeline propagates `AgentResult.timeout_reached` into `VerificationResult` as the `response_timeout_partial` metadata flag, so downstream stages can treat a partial trace differently from a clean completion.

**9. Single-turn usage only (usage.py)**
Some SDKs provide token counts per-turn, not aggregated. If usage extraction reads only the last message, a 10-turn run that cost $0.50 reports as $0.02. Fix: sum `input_tokens` and `output_tokens` across all AI messages in the conversation.

**10. Deferred error pattern for MCP cleanup (agent.py)**
When an exception propagates through an `AsyncExitStack` during MCP session cleanup, it can get wrapped in `ExceptionGroup`. Use a deferred error pattern to prevent this:

```python
deferred_error: Exception | None = None
async with AsyncExitStack() as exit_stack:
    # ... MCP setup and agent execution ...
    try:
        result = await execute_agent()
    except Exception as e:
        deferred_error = mapped_error
        deferred_error.__cause__ = e

# exit_stack closes cleanly (no exception propagating through it)
if deferred_error is not None:
    raise deferred_error
```

## Key conventions (from CLAUDE.md)

- `from __future__ import annotations` at top of every file
- `logger = logging.getLogger(__name__)` after all imports
- Google-style docstrings on all public methods
- Lazy `%`-style log formatting (not f-strings)
- Duck typing for Protocol compliance (no explicit `class Foo(LLMPort):`)
- Exception chaining: `raise ... from e`
- PEP 604 unions: `X | None`, not `Optional[X]`
- Files under 800 lines
- No cross-adapter imports

## Registry integration

After all adapter files are complete, choose the path that matches your distribution model. No Literal or InterfaceType changes are needed for any path; `ModelConfig.interface` accepts any string and the registry validates it dynamically.

### Manual registration

Call `AdapterRegistry.register(AdapterSpec(...))` in your own code before creating a `ModelConfig` that uses the interface. This is the simplest path for experiments or one-off adapters.

```python
from karenina.adapters.registry import AdapterRegistry, AdapterSpec

AdapterRegistry.register(AdapterSpec(
    interface="my_adapter",
    # ... factories, flags ...
))
```

### Built-in adapter (shipped with karenina)

1. Write `registration.py` in your adapter package (it calls `AdapterRegistry.register()` at import time)
2. Add an import in `karenina/src/karenina/adapters/registry.py`, inside `_load_builtins()`:
   ```python
   try:
       from karenina.adapters.<name> import registration as _xx  # noqa: F401
   except ImportError:
       logger.debug("<Name> registration module not available")
   ```
3. Verify `AdapterRegistry.get_spec("<interface>")` returns the correct spec
4. Run existing registry tests: `uv run pytest tests/unit/adapters/test_registry.py -x -v`

### Installable plugin (separate package)

1. Write `registration.py` in your plugin package (it calls `AdapterRegistry.register()` at import time)
2. Declare an entry point in your plugin's `pyproject.toml`:
   ```toml
   [project.entry-points."karenina.adapters"]
   <interface> = "<your_package>.registration"
   ```
3. Install the plugin (`pip install .` or `uv pip install .`); the registry discovers it automatically
4. Verify `AdapterRegistry.get_spec("<interface>")` returns the correct spec

## Commit strategy

Commit after each logical group (e.g., "availability + errors", "messages", "trace + usage", "agent", "llm + parser", "registration + prompts"). Each commit should leave tests passing.

## Next Step

Run `/karenina-adapter-test` to begin Phase 4.
