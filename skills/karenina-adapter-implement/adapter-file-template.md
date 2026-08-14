# Adapter File Template

Reference for the canonical files in a karenina adapter package.

## `__init__.py`

```python
"""<SDK Name> adapter for <description>.

Adapter classes:
    - <Name>AgentAdapter: Agent loops via <SDK method>
    - <Name>LLMAdapter: Simple LLM invocation
    - <Name>ParserAdapter: Structured output parsing

Utilities:
    - <Name>MessageConverter: Convert between unified Message and SDK types
    - check_<name>_available: Check if SDK is installed
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    "<Name>AgentAdapter",
    "<Name>LLMAdapter",
    "<Name>ParserAdapter",
    "<Name>MessageConverter",
    "check_<name>_available",
]

if TYPE_CHECKING:
    from .<module> import <Class>

def __getattr__(name: str) -> Any:
    """Lazy import adapter classes to avoid circular imports."""
    _imports = {
        "<Name>AgentAdapter": "<package>.agent",
        # ... one entry per exported name
    }
    if name in _imports:
        import importlib
        module = importlib.import_module(_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module '<package>' has no attribute '{name}'")
```

## `availability.py`

Check if the SDK is installed. Return `AdapterAvailability` from `karenina.adapters.registry`.

```python
from karenina.adapters.registry import AdapterAvailability

def check_<name>_available() -> AdapterAvailability:
    try:
        import <sdk_package>
        return AdapterAvailability(available=True, reason="...")
    except ImportError:
        return AdapterAvailability(available=False, reason="...", fallback_interface=None)
```

## `errors.py`

Map SDK exceptions to karenina's hierarchy:
- `AgentExecutionError`: general failures, turn limits
- `AgentTimeoutError`: timeouts
- `AgentResponseError`: malformed responses

Return tuple `(exception, limit_reached: bool)`.

## `messages.py`

Bidirectional converter class:
- `to_prompt_string(messages)`: karenina Message -> SDK input format
- `extract_system_prompt(messages)`: Pull system messages out separately
- `from_provider(sdk_messages)`: SDK response messages -> karenina Message list

**IMPORTANT: Read `karenina/src/karenina/ports/messages.py` first.** The field names are not what you'd guess:

```python
# Correct imports
from karenina.ports import Content, Message, Role, TextContent, ToolResultContent, ToolUseContent

# Role enum values: Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.TOOL
# Message factory methods: Message.user("text"), Message.system("text"), Message.assistant("text")
# Message text access: msg.text (NOT msg.text_content or msg.content)
# Message role access: msg.role (returns Role enum), msg.role.value (returns string)
# Building messages with content blocks:
#   Message(role=Role.ASSISTANT, content=[TextContent(text="hello")])
#   Message(role=Role.TOOL, content=[ToolResultContent(tool_use_id="id", content="result")])
# ToolUseContent fields: id, name, input (NOT tool_use_id, tool_name, tool_input)
```

Handle all content types: TextContent, ToolUseContent, ToolResultContent.

## `trace.py`

Convert SDK messages to karenina's dual trace format:
- `raw_trace`: Delimited string with `--- AI Message ---`, `--- Tool Call ---`, `--- Tool Result ---`
- `trace_messages`: Direct conversion via message converter

## `usage.py`

Extract `UsageMetadata(input_tokens, output_tokens, total_tokens, model)` from SDK responses.
Also provide `extract_actual_model()` to get the model name from response metadata.

## `agent.py`

Core adapter (~250-350 lines). **Read `karenina/src/karenina/ports/agent.py` first** to understand AgentResult fields.

Implementation flow sketch for `arun()`:

```python
async def arun(self, messages, tools=None, mcp_servers=None, config=None):
    config = config or AgentConfig()

    # 1. Convert messages
    prompt = self._converter.to_prompt_string(messages)
    system_prompt = self._converter.extract_system_prompt(messages)

    # 2. Initialize model (from initialization.py)
    model = create_chat_model(self._config)

    # 3. Configure backend for real filesystem (NOT virtual state!)
    #    See pitfall #1 in SKILL.md
    backend = create_real_filesystem_backend(config.workspace_path)

    # 4. Create agent from SDK
    agent = sdk_create_agent(model=model, system_prompt=system_prompt, backend=backend)

    # 5. Invoke with timeout (prefer partial recovery over raising)
    timeout_reached = False
    try:
        if config.timeout:
            result = await asyncio.wait_for(agent.ainvoke(...), timeout=config.timeout)
        else:
            result = await agent.ainvoke(...)
    except TimeoutError:
        # Try to recover partial state from the SDK. If the SDK keeps its
        # conversation state in a checkpointer or stream buffer, read it here.
        result = recover_partial_state(agent) or {"messages": []}
        timeout_reached = True
    except Exception as e:
        mapped, limit_reached = wrap_sdk_error(e)
        if limit_reached:
            return AgentResult(..., limit_reached=True)  # partial result
        raise mapped from e

    # 6. Extract results
    sdk_messages = result.get("messages", [])

    # If timeout hit and we truly have nothing, raise. Otherwise fall through
    # to return a partial AgentResult with timeout_reached=True.
    if timeout_reached and not sdk_messages:
        raise AgentTimeoutError(f"Agent execution timed out after {config.timeout}s with no messages")

    raw_trace = sdk_messages_to_raw_trace(sdk_messages)
    if timeout_reached:
        raw_trace += "\n\n[Note: Agent timed out - partial response shown]"

    return AgentResult(
        final_response=extract_final_response(sdk_messages),
        raw_trace=raw_trace,                                        # from trace.py
        trace_messages=self._converter.from_provider(sdk_messages), # from messages.py
        usage=extract_usage(sdk_messages, model=self._config.model_name),  # from usage.py
        turns=count_ai_messages(sdk_messages),
        limit_reached=result.get("is_last_step", False),
        actual_model=extract_actual_model(sdk_messages),
        timeout_reached=timeout_reached,
    )
```

Key methods:
- `arun(messages, tools, mcp_servers, config) -> AgentResult`
- `run(...)`: sync wrapper (see pattern below)
- `aclose()`: cleanup (often a no-op)

**Sync wrapper pattern** (copy this for `run()` in agent.py, `invoke()` in llm.py, `parse_to_pydantic()` in parser.py):

```python
def run(self, messages, tools=None, mcp_servers=None, config=None):
    from karenina.utils.async_runtime import get_async_portal

    portal = get_async_portal()
    if portal is not None:
        return portal.call(self.arun, messages, tools, mcp_servers, config)

    try:
        asyncio.get_running_loop()
        # In async context: run in a thread
        def run_in_thread():
            return asyncio.run(self.arun(messages, tools, mcp_servers, config))
        timeout = config.timeout if config and config.timeout else 600
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result(timeout=timeout)
    except RuntimeError:
        # No event loop: safe to use asyncio.run
        return asyncio.run(self.arun(messages, tools, mcp_servers, config))
```

**Lazy SDK import pattern** (for SDKs that may not be installed):

```python
# Module level: sentinel for lazy import
_create_agent_fn = None

async def arun(self, ...):
    global _create_agent_fn
    if _create_agent_fn is None:
        from sdk_package import create_agent as _fn
        _create_agent_fn = _fn
    # Now use _create_agent_fn(...)
```

This allows the module to load without the SDK installed (availability check gates actual use), and enables monkeypatching in tests.

## `llm.py`

Single-turn LLM calls (no agent loop). Key methods:
- `ainvoke(messages) -> LLMResponse`
- `invoke(...)`: sync wrapper
- `with_structured_output(schema) -> Self` (returns NEW instance, does not mutate)
- `astream(messages)`: async context manager yielding `StreamingLLMResponse` (raise `NotImplementedError` if the SDK has no streaming API)
- `stream_invoke(messages, timeout)`: sync wrapper that streams, returning an `LLMResponse` with `is_partial=True` and `usage_unavailable=True` when the timeout fires (raise `NotImplementedError` if streaming is unsupported)
- `capabilities` property returning `PortCapabilities` (set `supports_streaming=True` only if both streaming methods are implemented)

**HTTP request timeout:** When creating the SDK client, pass `self._config.request_timeout` as the client-level HTTP timeout. This is set by the pipeline from `VerificationConfig.request_timeout` (default 120s). Only pass it when not None:

```python
kwargs = {}
if self._config.request_timeout is not None:
    kwargs["timeout"] = self._config.request_timeout  # or "request_timeout" for LangChain models
client = SdkClient(**kwargs)
```

**LLMResponse construction:**

```python
from karenina.ports import LLMResponse
from karenina.ports.usage import UsageMetadata
from karenina.ports.capabilities import PortCapabilities

# ainvoke must return LLMResponse (NOT Message, NOT AgentResult)
return LLMResponse(
    content="the text response",
    usage=UsageMetadata(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        model="model-name",
    ),
    raw=sdk_response_object,  # optional, for debugging
    # is_partial defaults to False. Set to True only on streaming timeout.
    # usage_unavailable defaults to False. Set to True when token counts
    # could not be captured (e.g., streaming timeout dropped the final chunk).
)

# capabilities property
@property
def capabilities(self) -> PortCapabilities:
    return PortCapabilities(
        supports_system_prompt=True,
        supports_structured_output=True,
        supports_streaming=True,  # only if astream()/stream_invoke() work
    )
```

## `parser.py`

Structured output parsing. Key methods:
- `aparse_to_pydantic(messages, schema) -> ParsePortResult[T]`
- `parse_to_pydantic(...)`: sync wrapper
- `capabilities` property

## `registration.py`

Register with the AdapterRegistry:
```python
_spec = AdapterSpec(
    interface="<interface_name>",
    description="...",
    agent_factory=_create_agent,
    llm_factory=_create_llm,
    parser_factory=_create_parser,
    availability_checker=_check_availability,
    fallback_interface=None,
    supports_mcp=True,
    supports_tools=True,
    agent_tier="deep_agent",  # or "tool_loop"
    requires_provider=False,  # True if adapter needs model_provider to be set
)
AdapterRegistry.register(_spec)
```

Then import prompt modules at the bottom to trigger instruction registration.

## `prompts/` (parsing.py, rubric.py, deep_judgment.py)

Register adapter-specific prompt instructions via `AdapterInstructionRegistry.register()`.
Each provides `system_addition` and `user_addition` properties.

## Testing: How to mock the SDK

Use `monkeypatch` to replace the SDK's agent creation and model initialization so tests don't make real API calls:

```python
from unittest.mock import AsyncMock, MagicMock
import pytest

@pytest.mark.asyncio
async def test_arun_basic(model_config, monkeypatch):
    # 1. Build a fake SDK response
    from langchain_core.messages import AIMessage  # or your SDK's message type

    mock_result = {
        "messages": [AIMessage(content="The answer is 42.")],
        "is_last_step": False,
    }

    # 2. Mock the agent to return it
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(return_value=mock_result)

    # 3. Patch the SDK's creation function (use the lazy import sentinel)
    monkeypatch.setattr(
        "karenina.adapters.<name>.agent._create_agent_fn",
        lambda **_kw: mock_agent,
    )
    monkeypatch.setattr(
        "karenina.adapters.<name>.agent.create_chat_model",
        lambda _cfg, **_kw: MagicMock(),
    )

    # 4. Run and verify
    adapter = YourAgentAdapter(model_config)
    result = await adapter.arun(
        messages=[Message.user("test")],
        config=AgentConfig(max_turns=5),
    )

    assert "42" in result.final_response
    assert result.usage.input_tokens >= 0  # ALWAYS check usage is not broken
    assert result.limit_reached is False
```

**CRITICAL**: Always assert `usage.total_tokens >= 0` and `usage.model is not None`. Zero tokens indicates broken usage extraction that will silently corrupt cost reporting.

## Common Pitfalls

| File | Pitfall |
|------|---------|
| `__init__.py` | Forgetting lazy imports causes circular import errors |
| `messages.py` | Using wrong field names (check actual `Message` dataclass) |
| `agent.py` | Not handling sync-from-async correctly (use `get_async_portal`) |
| `agent.py` | Using a virtual/in-memory backend instead of real filesystem access. If the SDK defaults to virtual state (e.g., Deep Agents' StateBackend), the agent cannot see workspace files on disk. Always configure a real filesystem backend when workspace_path is involved. |
| `trace.py` | Missing tool call delimiters breaks regex highlighting |
| `registration.py` | Importing prompt modules before `AdapterRegistry.register()` |
| `errors.py` | Not detecting recursion limit errors (check error string) |
