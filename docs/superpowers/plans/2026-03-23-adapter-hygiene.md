# Adapter & Port Protocol Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 15 issues across 4 adapters, 3 port protocols, and cross-adapter behavior: fragile heuristics, missing protocol methods, operator precedence bugs, import-time side effects, and dead MCP integration code.

**Architecture:** Sequential dependency-ordered fixes. First wire up Deep Agents MCP/tools (095 -> 091/092), then formalize port protocols (138 -> 139), then fix independent issues in any order. Each task produces a commit.

**Tech Stack:** Python 3.13, Pydantic v2, asyncio (AsyncExitStack), pytest, LangChain, Claude Agent SDK

**Spec:** `docs/superpowers/specs/2026-03-23-adapter-hygiene-design.md`

---

## File Structure

No new source files created. All changes are modifications to existing files.

**Source files modified:**

| File | Issues | Changes |
|------|--------|---------|
| `src/karenina/adapters/langchain_deep_agents/llm.py` | 095, 142 | Add `aclose()`, add `max_retries` warning in `with_structured_output()` |
| `src/karenina/adapters/langchain_deep_agents/agent.py` | 091 | Wire tools/mcp_servers into `arun()` via AsyncExitStack |
| `src/karenina/adapters/langchain_deep_agents/mcp.py` | 092 | Accept `AsyncExitStack`, fix session lifetime |
| `src/karenina/adapters/langchain_deep_agents/errors.py` | 093 | Fix operator precedence on line 40 |
| `src/karenina/adapters/langchain_deep_agents/prompts/rubric.py` | 094 | Add `rubric_dynamic_presence_check` to `_RUBRIC_TASKS` |
| `src/karenina/adapters/langchain/trace.py` | 067 | Remove `_detect_tool_error` heuristic |
| `src/karenina/adapters/claude_agent_sdk/agent.py` | 068, 069 | Route errors through `wrap_sdk_error()`, fix MCP config detection |
| `src/karenina/adapters/claude_agent_sdk/errors.py` | 068 | Add turn-limit case to `wrap_sdk_error()` |
| `src/karenina/adapters/claude_agent_sdk/llm.py` | 142 | Add `max_retries` warning in `with_structured_output()` |
| `src/karenina/adapters/claude_tool/llm.py` | 090 | Remove module-level `load_dotenv()` |
| `src/karenina/adapters/claude_tool/agent.py` | 090 | Remove module-level `load_dotenv()` |
| `src/karenina/benchmark/verification/evaluators/template/deep_judgment.py` | 086 | Replace `PydanticOutputParser` with native Pydantic v2 |
| `src/karenina/benchmark/verification/evaluators/template/evaluator.py` | 086 | Remove unused `PydanticOutputParser` import |
| `src/karenina/ports/errors.py` | 068 | Add `limit_reached` attribute to `AgentExecutionError` |
| `src/karenina/ports/llm.py` | 138, 142 | Add `aclose()` to protocol, update `with_structured_output` docstring |
| `src/karenina/ports/agent.py` | 138, 139 | Add `aclose()` and `capabilities` to protocol |
| `src/karenina/ports/parser.py` | 138 | Add `aclose()` to protocol |

**Test files modified or created:**

| File | Tests for |
|------|-----------|
| `tests/unit/adapters/langchain_deep_agents/test_llm.py` | 095 (aclose), 142 (max_retries warning) |
| `tests/unit/adapters/langchain_deep_agents/test_agent.py` | 091 (MCP/tools wiring) |
| `tests/unit/adapters/langchain_deep_agents/test_mcp.py` | 092 (session lifetime) |
| `tests/unit/adapters/langchain_deep_agents/test_errors.py` (CREATE) | 093 (operator precedence) |
| `tests/unit/adapters/langchain_deep_agents/test_rubric_registration.py` (CREATE) | 094 (task registration) |
| `tests/unit/adapters/langchain/test_trace.py` (CREATE or find existing) | 067 (heuristic removal) |
| `tests/unit/adapters/claude_sdk/test_errors.py` | 068 (turn-limit wrapping) |
| `tests/unit/adapters/claude_sdk/test_mcp_config.py` | 069 (format detection) |
| `tests/unit/adapters/claude_tool/test_llm_adapter.py` | 090 (no load_dotenv side effect) |
| `tests/unit/adapters/conformance/test_llm_port.py` | 138 (aclose on protocol) |
| `tests/unit/adapters/conformance/test_agent_port.py` | 138, 139 (aclose + capabilities) |
| `tests/unit/adapters/conformance/test_parser_port.py` | 138 (aclose on protocol) |

---

### Task 1: Add `aclose()` to `DeepAgentsLLMAdapter` (Issue 095)

**Files:**
- Modify: `src/karenina/adapters/langchain_deep_agents/llm.py:189` (end of class)
- Test: `tests/unit/adapters/langchain_deep_agents/test_llm.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/adapters/langchain_deep_agents/test_llm.py`:

```python
@pytest.mark.asyncio
async def test_aclose_exists_and_is_noop(self, deep_agents_model_config):
    """aclose() should exist and complete without error."""
    adapter = DeepAgentsLLMAdapter(deep_agents_model_config)
    await adapter.aclose()  # Should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain_deep_agents/test_llm.py::TestDeepAgentsLLMAdapter::test_aclose_exists_and_is_noop -v`
Expected: FAIL with `AttributeError: 'DeepAgentsLLMAdapter' object has no attribute 'aclose'`

- [ ] **Step 3: Implement `aclose()`**

Add at end of `DeepAgentsLLMAdapter` class in `src/karenina/adapters/langchain_deep_agents/llm.py` (after `with_structured_output`):

```python
async def aclose(self) -> None:
    """Close underlying resources.

    No resources to clean up: the LangChain model is created fresh
    per ainvoke() call. Provided for interface consistency with other
    adapters.
    """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain_deep_agents/test_llm.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/karenina/adapters/langchain_deep_agents/llm.py tests/unit/adapters/langchain_deep_agents/test_llm.py
git commit -m "fix(deep-agents): add aclose() to DeepAgentsLLMAdapter (095)"
```

---

### Task 2: Fix `convert_mcp_to_tools()` session lifetime (Issue 092)

**Files:**
- Modify: `src/karenina/adapters/langchain_deep_agents/mcp.py:73-94`
- Test: `tests/unit/adapters/langchain_deep_agents/test_mcp.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/adapters/langchain_deep_agents/test_mcp.py`:

```python
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.unit
class TestConvertMcpToToolsSessionLifetime:
    """Test that convert_mcp_to_tools keeps sessions alive via exit_stack."""

    @pytest.mark.asyncio
    async def test_sessions_remain_open_after_return(self):
        """Tools returned by convert_mcp_to_tools should have live sessions.

        The function should register sessions with the exit_stack rather than
        closing them when the function returns.
        """
        from karenina.adapters.langchain_deep_agents.mcp import convert_mcp_to_tools

        mock_tool = MagicMock(name="mock_tool")
        mock_client = AsyncMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "karenina.adapters.langchain_deep_agents.mcp.MultiServerMCPClient",
            return_value=mock_client,
        ):
            async with AsyncExitStack() as exit_stack:
                tools = await convert_mcp_to_tools(
                    {"test": {"type": "http", "url": "http://localhost:8080"}},
                    exit_stack,
                )
                assert len(tools) == 1
                # Client should NOT have been closed yet (exit_stack still open)
                mock_client.__aexit__.assert_not_called()

            # After exit_stack closes, client should be cleaned up
            mock_client.__aexit__.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain_deep_agents/test_mcp.py::TestConvertMcpToToolsSessionLifetime -v`
Expected: FAIL (signature mismatch or session closed prematurely)

- [ ] **Step 3: Implement the fix**

Replace `convert_mcp_to_tools()` in `src/karenina/adapters/langchain_deep_agents/mcp.py` (lines 73-94):

```python
async def convert_mcp_to_tools(
    mcp_servers: dict[str, Any] | None,
    exit_stack: AsyncExitStack,
) -> list[Any]:
    """Convert MCP server configs to LangChain tools via langchain-mcp-adapters.

    Creates a MultiServerMCPClient registered with the provided exit_stack
    so that MCP sessions remain open for the lifetime of the caller's
    exit_stack context.

    Args:
        mcp_servers: Dict mapping server names to MCPServerConfig.
        exit_stack: AsyncExitStack managing session lifecycles. Sessions
            remain open until the exit stack closes.

    Returns:
        List of LangChain BaseTool instances from all MCP servers.
    """
    server_params = build_mcp_server_params(mcp_servers)
    if not server_params:
        return []

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(server_params)  # type: ignore[arg-type,misc]
    await exit_stack.enter_async_context(client)
    return await client.get_tools()
```

Add the import at top of file:

```python
from contextlib import AsyncExitStack
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain_deep_agents/test_mcp.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/karenina/adapters/langchain_deep_agents/mcp.py tests/unit/adapters/langchain_deep_agents/test_mcp.py
git commit -m "fix(deep-agents): fix MCP session lifetime with AsyncExitStack (092)"
```

---

### Task 3: Wire up MCP/tools in Deep Agents `arun()` (Issue 091)

**Files:**
- Modify: `src/karenina/adapters/langchain_deep_agents/agent.py:128-240`
- Test: `tests/unit/adapters/langchain_deep_agents/test_agent.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/adapters/langchain_deep_agents/test_agent.py`:

```python
@pytest.mark.unit
class TestDeepAgentsMCPToolsWiring:
    """Test that arun() passes tools and MCP-derived tools to the agent."""

    @pytest.mark.asyncio
    async def test_explicit_tools_passed_to_agent(self, deep_agents_model_config, monkeypatch):
        """Explicit tools should be forwarded to create_deep_agent."""
        from karenina.ports import AgentConfig, Message, Tool

        captured_kwargs = {}

        def mock_create_deep_agent(**kwargs):
            captured_kwargs.update(kwargs)
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value={
                "messages": [MagicMock(content="Done", type="ai")],
                "is_last_step": False,
            })
            return mock_agent

        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent._create_deep_agent",
            mock_create_deep_agent,
        )
        # Mock trace/usage extraction to avoid LangChain imports
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.deep_agents_messages_to_raw_trace",
            lambda msgs: "trace",
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.extract_deep_agents_usage",
            lambda msgs, model: UsageMetadata(model=model),
        )
        monkeypatch.setattr(
            "karenina.adapters.langchain_deep_agents.agent.extract_actual_model",
            lambda msgs: None,
        )

        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        test_tool = Tool(name="test_tool", description="A test tool", input_schema={})

        await adapter.arun(
            messages=[Message.user("Hello")],
            tools=[test_tool],
            config=AgentConfig(max_turns=5),
        )

        assert "tools" in captured_kwargs
        assert len(captured_kwargs["tools"]) >= 1
```

(Also add necessary imports at top: `from unittest.mock import AsyncMock, MagicMock` and `from karenina.ports.usage import UsageMetadata`)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain_deep_agents/test_agent.py::TestDeepAgentsMCPToolsWiring -v`
Expected: FAIL (tools not passed to create_deep_agent because of `# noqa: ARG002`)

- [ ] **Step 3: Implement the wiring**

Modify `arun()` in `src/karenina/adapters/langchain_deep_agents/agent.py`:

1. Add import at top of file: `from contextlib import AsyncExitStack`

2. Remove `# noqa: ARG002` comments from lines 131-132:
```python
    tools: list[Tool] | None = None,
    mcp_servers: dict[str, MCPServerConfig] | None = None,
```

3. Replace the agent execution block (lines 199-240) with the AsyncExitStack pattern. After `agent_kwargs` is built (line 197), before creating the agent:

```python
        # Convert MCP servers to LangChain tools and combine with explicit tools
        all_tools: list[Any] = []
        if tools:
            all_tools.extend(tools)

        # Use AsyncExitStack for persistent MCP sessions. Sessions stay alive
        # for all tool calls during agent execution. Exceptions are captured
        # inside the block and re-raised after clean exit, because MCP session
        # cleanup can wrap errors in ExceptionGroup if an exception propagates
        # through the exit stack.
        deferred_error: Exception | None = None

        async with AsyncExitStack() as exit_stack:
            # Convert MCP servers to LangChain tools
            if mcp_servers:
                from .mcp import convert_mcp_to_tools

                try:
                    mcp_tools = await convert_mcp_to_tools(mcp_servers, exit_stack)
                    all_tools.extend(mcp_tools)
                    logger.info(
                        "Loaded %d MCP tools from %d servers",
                        len(mcp_tools),
                        len(mcp_servers),
                    )
                except Exception as e:
                    logger.warning("Failed to load MCP tools: %s", e)
                    deferred_error = AgentExecutionError(
                        f"Failed to initialize MCP tools: {e}"
                    )
                    deferred_error.__cause__ = e

            if deferred_error is None:
                # Pass tools to agent if any were collected
                if all_tools:
                    agent_kwargs["tools"] = all_tools

                # Create the agent
                agent = _create_deep_agent(**agent_kwargs)

                # Build invocation input
                invoke_input: dict[str, Any] = {
                    "messages": [{"role": "user", "content": prompt_string}],
                }

                # LangGraph config for recursion limit
                # Each tool call + response = 2 steps, so double max_turns
                langgraph_config: dict[str, Any] = {
                    "recursion_limit": config.max_turns * 2,
                }

                # Execute agent
                result: dict[str, Any] = {}
                limit_reached = False

                async def execute_agent() -> None:
                    nonlocal result, limit_reached
                    result = await agent.ainvoke(invoke_input, config=langgraph_config)
                    if result.get("is_last_step", False):
                        limit_reached = True

                try:
                    if config.timeout:
                        await asyncio.wait_for(execute_agent(), timeout=config.timeout)
                    else:
                        await execute_agent()

                except TimeoutError as e:
                    deferred_error = AgentTimeoutError(
                        f"Agent execution timed out after {config.timeout}s"
                    )
                    deferred_error.__cause__ = e
                except Exception as e:
                    mapped_error, was_limit = wrap_deep_agents_error(e)
                    if was_limit:
                        limit_reached = True
                        logger.warning("Agent hit turn limit: %s", e)
                    else:
                        deferred_error = mapped_error
                        deferred_error.__cause__ = e

        # exit_stack closed: MCP sessions cleaned up

        if deferred_error is not None:
            raise deferred_error
```

Also add `AgentExecutionError` to the imports from `karenina.ports` at the top of the file (line 21-30).

4. The rest of the method (extract messages, build traces, etc.) stays the same but now uses the `result` and `limit_reached` variables that were set inside the exit_stack block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain_deep_agents/test_agent.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run broader adapter tests**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain_deep_agents/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/karenina/adapters/langchain_deep_agents/agent.py tests/unit/adapters/langchain_deep_agents/test_agent.py
git commit -m "feat(deep-agents): wire up MCP/tools support in arun() (091)"
```

---

### Task 4: Add `aclose()` to port protocols (Issue 138)

**Files:**
- Modify: `src/karenina/ports/llm.py:129` (end of LLMPort)
- Modify: `src/karenina/ports/agent.py:313` (end of AgentPort)
- Modify: `src/karenina/ports/parser.py:133` (end of ParserPort)
- Test: `tests/unit/adapters/conformance/test_llm_port.py`, `test_agent_port.py`, `test_parser_port.py`

- [ ] **Step 1: Add `aclose()` to all three protocols**

In `src/karenina/ports/llm.py`, add after `with_structured_output` (before end of class):

```python
    async def aclose(self) -> None:
        """Close underlying resources.

        Implementations should release any held resources (HTTP connections,
        file handles, MCP sessions). Safe to call multiple times. The default
        is a no-op for adapters with no resources to clean up.
        """
        ...
```

In `src/karenina/ports/agent.py`, add after `run` (before end of class):

```python
    async def aclose(self) -> None:
        """Close underlying resources.

        Implementations should release any held resources (HTTP connections,
        file handles, MCP sessions). Safe to call multiple times. The default
        is a no-op for adapters with no resources to clean up.
        """
        ...
```

In `src/karenina/ports/parser.py`, add after `parse_to_pydantic` (before end of class):

```python
    async def aclose(self) -> None:
        """Close underlying resources.

        Implementations should release any held resources (HTTP connections,
        file handles, MCP sessions). Safe to call multiple times. The default
        is a no-op for adapters with no resources to clean up.
        """
        ...
```

- [ ] **Step 2: Run conformance tests**

Run: `cd karenina && uv run pytest tests/unit/adapters/conformance/ -v`
Expected: All PASS (all adapters already implement aclose)

- [ ] **Step 3: Run mypy on ports**

Run: `cd karenina && uv run mypy src/karenina/ports/`
Expected: No new errors

- [ ] **Step 4: Commit**

```bash
git add src/karenina/ports/llm.py src/karenina/ports/agent.py src/karenina/ports/parser.py
git commit -m "fix(ports): add aclose() to LLMPort, AgentPort, ParserPort protocols (138)"
```

---

### Task 5: Add `capabilities` to `AgentPort` (Issue 139)

**Files:**
- Modify: `src/karenina/ports/agent.py:248` (before `arun` in AgentPort)
- Test: `tests/unit/adapters/conformance/test_agent_port.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/adapters/conformance/test_agent_port.py`:

```python
def test_agent_port_has_capabilities(self, agent_adapter):
    """AgentPort should expose a capabilities property."""
    caps = agent_adapter.capabilities
    assert isinstance(caps, PortCapabilities)
```

(Add `from karenina.ports.capabilities import PortCapabilities` import if missing.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd karenina && uv run pytest tests/unit/adapters/conformance/test_agent_port.py::TestAgentPortConformance::test_agent_port_has_capabilities -v`
Expected: FAIL (AgentPort has no capabilities property)

- [ ] **Step 3: Add `capabilities` to AgentPort**

In `src/karenina/ports/agent.py`, add the import at top: `from karenina.ports.capabilities import PortCapabilities`

Add before `arun` method in AgentPort class:

```python
    @property
    def capabilities(self) -> PortCapabilities:
        """Declare what prompt features this agent adapter supports.

        Returns:
            PortCapabilities with adapter-specific feature flags.
            Defaults to PortCapabilities() (system prompts supported,
            structured output not supported).
        """
        return PortCapabilities()
```

- [ ] **Step 4: Add `capabilities` to `DeepAgentsAgentAdapter`**

Check if `DeepAgentsAgentAdapter` already has a `capabilities` property. If not, add one to `src/karenina/adapters/langchain_deep_agents/agent.py`.

Add the import at module level (near the existing ports imports):

```python
from karenina.ports.capabilities import PortCapabilities
```

Add the property to the class:

```python
    @property
    def capabilities(self) -> PortCapabilities:
        """Declare adapter capabilities.

        Returns:
            PortCapabilities with system_prompt=True.
        """
        return PortCapabilities(supports_system_prompt=True)
```

- [ ] **Step 5: Write test for Deep Agents agent capabilities**

Add to `tests/unit/adapters/langchain_deep_agents/test_agent.py`:

```python
@pytest.mark.unit
class TestDeepAgentsAgentCapabilities:
    def test_capabilities_returns_port_capabilities(self, deep_agents_model_config):
        """DeepAgentsAgentAdapter should expose a capabilities property."""
        from karenina.ports.capabilities import PortCapabilities

        adapter = DeepAgentsAgentAdapter(deep_agents_model_config)
        caps = adapter.capabilities
        assert isinstance(caps, PortCapabilities)
        assert caps.supports_system_prompt is True
```

- [ ] **Step 6: Run tests**

Run: `cd karenina && uv run pytest tests/unit/adapters/conformance/ tests/unit/adapters/langchain_deep_agents/test_agent.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/karenina/ports/agent.py src/karenina/adapters/langchain_deep_agents/agent.py tests/unit/adapters/langchain_deep_agents/test_agent.py
git commit -m "fix(ports): add capabilities property to AgentPort protocol (139)"
```

---

### Task 6: Centralize SDK error translation (Issue 068)

**Files:**
- Modify: `src/karenina/adapters/claude_agent_sdk/errors.py:128-134` (before generic fallback)
- Modify: `src/karenina/adapters/claude_agent_sdk/agent.py:365-373`
- Test: `tests/unit/adapters/claude_sdk/test_errors.py`

- [ ] **Step 1: Write tests for the new turn-limit case**

Add to `tests/unit/adapters/claude_sdk/test_errors.py`:

```python
def test_turn_limit_error_by_type_name(self) -> None:
    """MaxTurnsExceeded exception should map to AgentExecutionError with limit flag."""
    mock_error = MagicMock()
    mock_error.__class__.__name__ = "MaxTurnsExceededException"

    result = wrap_sdk_error(mock_error)

    assert isinstance(result, AgentExecutionError)
    assert result.limit_reached is True

def test_turn_limit_error_by_message(self) -> None:
    """Exception with 'max_turns' in message should map to limit error."""
    error = RuntimeError("Agent exceeded max_turns limit")

    result = wrap_sdk_error(error)

    assert isinstance(result, AgentExecutionError)
    assert result.limit_reached is True

def test_recursion_in_message(self) -> None:
    """Exception with 'recursion' in message should map to limit error."""
    error = RuntimeError("Hit recursion limit after 50 turns")

    result = wrap_sdk_error(error)

    assert isinstance(result, AgentExecutionError)
    assert result.limit_reached is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd karenina && uv run pytest tests/unit/adapters/claude_sdk/test_errors.py::TestWrapSdkError::test_turn_limit_error_by_type_name -v`
Expected: FAIL (no limit_reached attribute or wrong error type)

- [ ] **Step 3: Add `limit_reached` attribute to `AgentExecutionError`**

`AgentExecutionError` currently has only `message` and `stderr`. Add `limit_reached` to `src/karenina/ports/errors.py`:

```python
class AgentExecutionError(PortError):
    """Raised when an agent fails during execution.

    ...existing docstring...

    Args:
        message: Human-readable description of the error.
        stderr: Standard error output from the failed process, if available.
        limit_reached: True if the error was caused by hitting a turn/recursion limit.
    """

    def __init__(
        self, message: str, stderr: str | None = None, limit_reached: bool = False
    ) -> None:
        super().__init__(message)
        self.stderr = stderr
        self.limit_reached = limit_reached
```

The default `limit_reached=False` is backward-compatible with all existing callers.

- [ ] **Step 4: Implement the turn-limit case in `wrap_sdk_error()`**

In `src/karenina/adapters/claude_agent_sdk/errors.py`, add before the generic fallback (before line 129):

```python
    # Turn/recursion limit errors (by exception type name)
    if exc_type_name in ("MaxTurnsExceededException", "MaxTurnsExceeded"):
        return AgentExecutionError(
            message=f"Agent hit turn limit: {e}",
            limit_reached=True,
        )

    # Turn/recursion limit errors (by message content, last resort)
    error_lower = str(e).lower()
    if "recursion" in error_lower or "limit" in error_lower or "max_turns" in error_lower:
        return AgentExecutionError(
            message=f"Agent hit turn limit: {e}",
            limit_reached=True,
        )

- [ ] **Step 5: Update `arun()` in `agent.py` to use `wrap_sdk_error()`**

Replace lines 365-373 in `src/karenina/adapters/claude_agent_sdk/agent.py`:

```python
        except Exception as e:
            from .errors import wrap_sdk_error

            translated = wrap_sdk_error(e)
            if getattr(translated, "limit_reached", False):
                limit_reached = True
                logger.warning("Agent hit turn limit: %s", e)
            else:
                raise translated from e
```

- [ ] **Step 6: Run tests**

Run: `cd karenina && uv run pytest tests/unit/adapters/claude_sdk/test_errors.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/karenina/ports/errors.py src/karenina/adapters/claude_agent_sdk/errors.py src/karenina/adapters/claude_agent_sdk/agent.py tests/unit/adapters/claude_sdk/test_errors.py
git commit -m "fix(claude-sdk): centralize turn-limit detection via wrap_sdk_error (068)"
```

---

### Task 7: Fix operator precedence in Deep Agents errors (Issue 093)

**Files:**
- Modify: `src/karenina/adapters/langchain_deep_agents/errors.py:40`
- Test: `tests/unit/adapters/langchain_deep_agents/test_errors.py` (CREATE)

- [ ] **Step 1: Create test file and write the failing test**

Create `tests/unit/adapters/langchain_deep_agents/test_errors.py`:

```python
"""Tests for Deep Agents error wrapping."""

from __future__ import annotations

import pytest

from karenina.adapters.langchain_deep_agents.errors import wrap_deep_agents_error
from karenina.ports.errors import AgentExecutionError, AgentResponseError


@pytest.mark.unit
class TestWrapDeepAgentsError:
    def test_parse_alone_does_not_trigger_response_error(self):
        """'parse' alone should NOT map to AgentResponseError (093 regression)."""
        error = RuntimeError("failed to parse MCP configuration")
        mapped, was_limit = wrap_deep_agents_error(error)

        # Should be generic execution error, NOT AgentResponseError
        assert not isinstance(mapped, AgentResponseError)
        assert isinstance(mapped, AgentExecutionError)

    def test_parse_output_triggers_response_error(self):
        """'parse' + 'output' should map to AgentResponseError."""
        error = RuntimeError("failed to parse output from agent")
        mapped, was_limit = wrap_deep_agents_error(error)

        assert isinstance(mapped, AgentResponseError)
        assert was_limit is False

    def test_output_format_triggers_response_error(self):
        """'output' + 'format' should map to AgentResponseError."""
        error = RuntimeError("output format error in response")
        mapped, was_limit = wrap_deep_agents_error(error)

        assert isinstance(mapped, AgentResponseError)
        assert was_limit is False

    def test_recursion_limit_detected(self):
        """Recursion limit errors should set limit_reached=True."""
        error = RuntimeError("Hit recursion limit")
        mapped, was_limit = wrap_deep_agents_error(error)

        assert isinstance(mapped, AgentExecutionError)
        assert was_limit is True
```

- [ ] **Step 2: Run test to verify the precedence bug**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain_deep_agents/test_errors.py::TestWrapDeepAgentsError::test_parse_alone_does_not_trigger_response_error -v`
Expected: FAIL (currently "parse" alone triggers AgentResponseError)

- [ ] **Step 3: Fix the operator precedence**

In `src/karenina/adapters/langchain_deep_agents/errors.py`, change line 40 from:

```python
    if "parse" in error_str or "output" in error_str and "format" in error_str:
```

to:

```python
    if ("parse" in error_str and "output" in error_str) or ("output" in error_str and "format" in error_str):
```

- [ ] **Step 4: Run all error tests**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain_deep_agents/test_errors.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/karenina/adapters/langchain_deep_agents/errors.py tests/unit/adapters/langchain_deep_agents/test_errors.py
git commit -m "fix(deep-agents): fix operator precedence in error classification (093)"
```

---

### Task 8: Remove `_detect_tool_error` heuristic (Issue 067)

**Files:**
- Modify: `src/karenina/adapters/langchain/trace.py:215,224-249`
- Test: find or create trace test

- [ ] **Step 1: Find existing trace tests**

Check: `tests/unit/adapters/langchain/` for trace-related tests. Look for tests that call `_detect_tool_error` or `langchain_messages_to_trace_messages`.

- [ ] **Step 2: Write a test verifying false positives are gone**

Add a test (to existing trace test file or create `tests/unit/adapters/langchain/test_trace_tool_error.py`):

```python
"""Test that tool error detection no longer produces false positives."""

import pytest


@pytest.mark.unit
class TestToolErrorDetectionRemoved:
    def test_legitimate_tool_output_not_flagged(self):
        """Tool output containing 'error' in legitimate context should not be flagged."""
        from karenina.adapters.langchain.trace import _detect_tool_error

        # These should all return False (no false positives)
        assert _detect_tool_error("No errors found in the analysis") is False
        assert _detect_tool_error("timeout set to 30 seconds") is False
        assert _detect_tool_error("The exception handling looks correct") is False
        assert _detect_tool_error("Search failed to find any results") is False
```

- [ ] **Step 3: Run to confirm the false positives exist**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain/test_trace_tool_error.py -v`
Expected: FAIL (current heuristic flags these as errors)

- [ ] **Step 4: Remove the heuristic**

In `src/karenina/adapters/langchain/trace.py`:

Replace the `_detect_tool_error` function (lines 224-249) with:

```python
def _detect_tool_error(content: str) -> bool:
    """Check whether tool output represents an error.

    LangChain ToolMessage has no structural error indicator, so this
    always returns False. The previous heuristic (substring matching on
    keywords like 'error', 'failed') produced too many false positives
    on legitimate tool output.

    Args:
        content: The tool result content string.

    Returns:
        Always False. Retained for backward compatibility with callers.
    """
    return False
```

- [ ] **Step 5: Run test to verify**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain/test_trace_tool_error.py -v`
Expected: All PASS

- [ ] **Step 6: Run broader trace tests**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/karenina/adapters/langchain/trace.py tests/unit/adapters/langchain/test_trace_tool_error.py
git commit -m "fix(langchain): remove false-positive-prone tool error heuristic (067)"
```

---

### Task 9: Replace `PydanticOutputParser` with native Pydantic v2 (Issue 086)

**Files:**
- Modify: `src/karenina/benchmark/verification/evaluators/template/deep_judgment.py:133-136`
- Modify: `src/karenina/benchmark/verification/evaluators/template/evaluator.py:392,415-416`
- Create+Delete: `scripts/compare_schema_formats.py` (temporary)

- [ ] **Step 1: Write comparison script**

Create `scripts/compare_schema_formats.py`:

```python
"""Compare PydanticOutputParser vs model_json_schema output formats.

Run: cd karenina && uv run python scripts/compare_schema_formats.py
"""

import json

from pydantic import BaseModel, Field


class SampleAnswer(BaseModel):
    """Sample answer template for comparison."""

    gene_name: str = Field(description="Name of the gene")
    is_oncogene: bool = Field(description="Whether the gene is an oncogene")
    confidence: float = Field(description="Confidence score between 0 and 1")


# Method 1: PydanticOutputParser
try:
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=SampleAnswer)
    langchain_output = parser.get_format_instructions()
    print("=== PydanticOutputParser.get_format_instructions() ===")
    print(langchain_output)
    print()
except ImportError:
    print("langchain_core not installed, skipping PydanticOutputParser")
    langchain_output = None

# Method 2: model_json_schema
schema = SampleAnswer.model_json_schema()
pydantic_output = json.dumps(schema, indent=2)
print("=== json.dumps(model_json_schema(), indent=2) ===")
print(pydantic_output)
print()

# Method 3: model_fields
print("=== model_fields ===")
for name, field in SampleAnswer.model_fields.items():
    print(f"  {name}: {field.annotation.__name__} - {field.description}")
```

- [ ] **Step 2: Run the comparison script**

Run: `cd karenina && uv run python scripts/compare_schema_formats.py`

Examine the output. If `model_json_schema()` provides equivalent information (field names, types, descriptions), proceed with replacement. Otherwise, use `model_fields` approach.

- [ ] **Step 3: Replace in `deep_judgment.py`**

In `src/karenina/benchmark/verification/evaluators/template/deep_judgment.py`, replace lines 133-136:

```python
# Before:
from langchain_core.output_parsers import PydanticOutputParser
temp_parser = PydanticOutputParser(pydantic_object=RawAnswer)
json_schema = temp_parser.get_format_instructions()
```

With:

```python
import json as _json
json_schema = _json.dumps(RawAnswer.model_json_schema(), indent=2)
```

(Use `_json` alias to avoid shadowing any local `json` variable. Check for conflicts first.)

- [ ] **Step 4: Remove from `evaluator.py`**

In `src/karenina/benchmark/verification/evaluators/template/evaluator.py`:
- Remove the `from langchain_core.output_parsers import PydanticOutputParser` import (line 392)
- Remove `parser = PydanticOutputParser(pydantic_object=self.answer_class)` (line 415)
- Remove `format_instructions = parser.get_format_instructions()` (line 416)
- Check if `format_instructions` is passed anywhere. If it's passed to `deep_judgment_parse()`, remove that parameter too (deep_judgment.py generates its own now).

- [ ] **Step 5: Run deep judgment tests**

Run: `cd karenina && uv run pytest tests/ -x -q -k "deep_judgment"`
Expected: All PASS

- [ ] **Step 6: Delete comparison script**

```bash
rm scripts/compare_schema_formats.py
```

- [ ] **Step 7: Commit**

```bash
git add src/karenina/benchmark/verification/evaluators/template/deep_judgment.py src/karenina/benchmark/verification/evaluators/template/evaluator.py
git commit -m "fix(evaluators): replace PydanticOutputParser with native Pydantic v2 (086)"
```

---

### Task 10: Remove module-level `load_dotenv()` (Issue 090)

**Files:**
- Modify: `src/karenina/adapters/claude_tool/llm.py:20,32`
- Modify: `src/karenina/adapters/claude_tool/agent.py:23,52`

- [ ] **Step 1: Write test verifying no import side effect**

Add to `tests/unit/adapters/claude_tool/test_llm_adapter.py` (or create a new file):

```python
@pytest.mark.unit
class TestNoLoadDotenvSideEffect:
    def test_importing_llm_does_not_call_load_dotenv(self, monkeypatch):
        """Importing claude_tool.llm should not call load_dotenv()."""
        import importlib

        call_count = 0
        original_load = None

        def tracking_load(*args, **kwargs):
            nonlocal call_count
            call_count += 1

        monkeypatch.setattr("dotenv.load_dotenv", tracking_load)

        import karenina.adapters.claude_tool.llm
        importlib.reload(karenina.adapters.claude_tool.llm)

        assert call_count == 0, "load_dotenv() was called during module import"
```

- [ ] **Step 2: Remove `load_dotenv` from both files**

In `src/karenina/adapters/claude_tool/llm.py`:
- Remove `from dotenv import load_dotenv` (line 20)
- Remove `load_dotenv()` (line 32)
- Remove the comment above it

In `src/karenina/adapters/claude_tool/agent.py`:
- Remove `from dotenv import load_dotenv` (line 23)
- Remove `load_dotenv()` (line 52)
- Remove the comment above it

- [ ] **Step 3: Run tests**

Run: `cd karenina && uv run pytest tests/unit/adapters/claude_tool/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/karenina/adapters/claude_tool/llm.py src/karenina/adapters/claude_tool/agent.py tests/unit/adapters/claude_tool/test_llm_adapter.py
git commit -m "fix(claude-tool): remove module-level load_dotenv() calls (090)"
```

---

### Task 11: Add missing rubric task registration (Issue 094)

**Files:**
- Modify: `src/karenina/adapters/langchain_deep_agents/prompts/rubric.py:35-41`
- Test: `tests/unit/adapters/langchain_deep_agents/test_rubric_registration.py` (CREATE)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/adapters/langchain_deep_agents/test_rubric_registration.py`:

```python
"""Tests for Deep Agents rubric task registration."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestRubricTaskRegistration:
    def test_rubric_dynamic_presence_check_registered(self):
        """Deep Agents should register rubric_dynamic_presence_check like other adapters."""
        from karenina.adapters.langchain_deep_agents.prompts.rubric import _RUBRIC_TASKS

        assert "rubric_dynamic_presence_check" in _RUBRIC_TASKS

    def test_rubric_task_count_matches_other_adapters(self):
        """Deep Agents should register the same number of rubric tasks as Claude Tool."""
        from karenina.adapters.langchain_deep_agents.prompts.rubric import _RUBRIC_TASKS

        assert len(_RUBRIC_TASKS) == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain_deep_agents/test_rubric_registration.py -v`
Expected: FAIL (only 5 tasks, missing rubric_dynamic_presence_check)

- [ ] **Step 3: Add the missing task**

In `src/karenina/adapters/langchain_deep_agents/prompts/rubric.py`, add to `_RUBRIC_TASKS` list:

```python
_RUBRIC_TASKS = [
    "rubric_llm_trait_batch",
    "rubric_llm_trait_single",
    "rubric_literal_trait_batch",
    "rubric_literal_trait_single",
    "rubric_metric_trait",
    "rubric_dynamic_presence_check",
]
```

- [ ] **Step 4: Run test**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain_deep_agents/test_rubric_registration.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/karenina/adapters/langchain_deep_agents/prompts/rubric.py tests/unit/adapters/langchain_deep_agents/test_rubric_registration.py
git commit -m "fix(deep-agents): add missing rubric_dynamic_presence_check registration (094)"
```

---

### Task 12: Fix MCP config format detection (Issue 069)

**Files:**
- Modify: `src/karenina/adapters/claude_agent_sdk/agent.py` (`_convert_mcp_servers` method)
- Test: `tests/unit/adapters/claude_sdk/test_mcp_config.py`

- [ ] **Step 1: Write test for the false positive**

Add to `tests/unit/adapters/claude_sdk/test_mcp_config.py`:

```python
@pytest.mark.unit
class TestMCPConfigFormatDetection:
    def test_custom_config_with_type_key_not_mistaken_for_sdk(self):
        """A config with a 'type' key for karenina purposes should be converted, not passed through."""
        # MCPHttpServerConfig has 'type': 'http' and 'url' - this IS karenina format
        mcp_config = {
            "my_server": {"type": "http", "url": "http://localhost:8080"}
        }
        # Should be recognized as karenina format and converted
        # (not passed through as-is just because 'type' key exists)
```

**Note:** The implementer should check the current behavior of `_convert_mcp_servers()` with this input and verify the test catches the issue before fixing.

- [ ] **Step 2: Implement proper format detection**

Replace the format-sniffing logic in `_convert_mcp_servers()` with isinstance checks:

```python
from karenina.ports.agent import MCPHttpServerConfig, MCPStdioServerConfig

# Check if config values match karenina's MCPServerConfig format
first_config = next(iter(mcp_servers.values()), {})
if isinstance(first_config, dict):
    # Check for karenina TypedDict fields
    is_karenina_stdio = "command" in first_config and first_config.get("type", "stdio") == "stdio"
    is_karenina_http = "url" in first_config and first_config.get("type") in ("http", "sse")

    if is_karenina_stdio or is_karenina_http:
        # Karenina format: convert to SDK format
        return convert_mcp_config(mcp_servers)

# Assume already SDK format
return mcp_servers
```

- [ ] **Step 3: Run tests**

Run: `cd karenina && uv run pytest tests/unit/adapters/claude_sdk/test_mcp_config.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/karenina/adapters/claude_agent_sdk/agent.py tests/unit/adapters/claude_sdk/test_mcp_config.py
git commit -m "fix(claude-sdk): use proper format detection for MCP config (069)"
```

---

### Task 13: Warn when `max_retries` is ignored (Issue 142)

**Files:**
- Modify: `src/karenina/adapters/claude_agent_sdk/llm.py` (with_structured_output)
- Modify: `src/karenina/adapters/langchain_deep_agents/llm.py:171-189` (with_structured_output)
- Modify: `src/karenina/ports/llm.py:108-129` (docstring)
- Test: `tests/unit/adapters/langchain_deep_agents/test_llm.py`, `tests/unit/adapters/claude_sdk/test_errors.py` or similar

- [ ] **Step 1: Write tests**

Add to `tests/unit/adapters/langchain_deep_agents/test_llm.py`:

```python
def test_with_structured_output_warns_on_max_retries(self, deep_agents_model_config, caplog):
    """with_structured_output should warn when max_retries is provided."""
    import logging

    from pydantic import BaseModel, Field

    class Answer(BaseModel):
        value: str = Field(description="The answer")

    adapter = DeepAgentsLLMAdapter(deep_agents_model_config)

    with caplog.at_level(logging.WARNING):
        adapter.with_structured_output(Answer, max_retries=5)

    assert "max_retries" in caplog.text
    assert "ignored" in caplog.text.lower()
```

- [ ] **Step 2: Add warning to Deep Agents adapter**

In `src/karenina/adapters/langchain_deep_agents/llm.py`, modify `with_structured_output()`:

```python
def with_structured_output(
    self,
    schema: type[BaseModel],
    *,
    max_retries: int | None = None,
) -> DeepAgentsLLMAdapter:
    """Return a new adapter configured for structured output.

    Args:
        schema: A Pydantic model class defining the output structure.
        max_retries: Not supported by this adapter. A warning is emitted
            if a non-None value is provided.

    Returns:
        A new DeepAgentsLLMAdapter configured with the schema.
    """
    if max_retries is not None:
        logger.warning(
            "max_retries=%d ignored by langchain_deep_agents adapter; "
            "retry behavior is managed internally by LangChain",
            max_retries,
        )
    return DeepAgentsLLMAdapter(
        self._config,
        _structured_schema=schema,
    )
```

Remove the `# noqa: ARG002` comment since the parameter is now used (for the warning check).

- [ ] **Step 3: Add warning to Claude SDK adapter**

Same pattern in `src/karenina/adapters/claude_agent_sdk/llm.py`:

```python
if max_retries is not None:
    logger.warning(
        "max_retries=%d ignored by claude_agent_sdk adapter; "
        "retry behavior is managed internally by the SDK via max_turns",
        max_retries,
    )
```

Remove the `# noqa: ARG002` comment.

- [ ] **Step 4: Update LLMPort docstring**

In `src/karenina/ports/llm.py`, update the `with_structured_output` docstring (line 117):

```python
        Args:
            schema: A Pydantic model class defining the output structure.
            max_retries: Maximum retry attempts on validation failure.
                Not all adapters support this parameter. LangChain and Claude
                Tool adapters respect it; Claude SDK and Deep Agents adapters
                ignore it (with a warning). Check adapter documentation for
                details.
```

- [ ] **Step 5: Run tests**

Run: `cd karenina && uv run pytest tests/unit/adapters/langchain_deep_agents/test_llm.py tests/unit/adapters/claude_sdk/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/karenina/adapters/langchain_deep_agents/llm.py src/karenina/adapters/claude_agent_sdk/llm.py src/karenina/ports/llm.py tests/unit/adapters/langchain_deep_agents/test_llm.py
git commit -m "fix(adapters): warn when max_retries is ignored by adapter (142)"
```

---

### Task 14: Update adapter creation skills and documentation (Group 6)

**Files:**
- Modify: `.claude/skills/create-karenina-adapter/SKILL.md`
- Modify: `.claude/skills/adapter-design/SKILL.md`
- Modify: `.claude/skills/adapter-implement/SKILL.md`
- Modify: `.claude/skills/adapter-test/SKILL.md`
- Modify: `.claude/skills/adapter-review/SKILL.md`
- Modify: `docs/advanced-adapters/writing-adapters.md`
- Modify: `docs/advanced-adapters/ports.md`
- Modify: `docs/advanced-adapters/available-adapters.md`
- Modify: `docs/advanced-adapters/mcp-integration.md`

**Note:** These are documentation files. Read each one first, then make targeted updates.

- [ ] **Step 1: Read all 5 adapter skills**

Read each skill file to understand current content before modifying.

- [ ] **Step 2: Update `adapter-implement/SKILL.md`**

Add to the implementation checklist:
- `aclose()` is now a **required** protocol method on all three ports (LLMPort, AgentPort, ParserPort). Every adapter must implement it. No-op implementations are acceptable for adapters with no resources to clean up.
- `AgentPort` now requires a `capabilities` property returning `PortCapabilities`, matching LLMPort and ParserPort.
- For MCP session management, use `AsyncExitStack` pattern (reference: `langchain/agent.py` and `langchain_deep_agents/agent.py`). Never create MCP sessions inside `async with` that closes before tools are used.

- [ ] **Step 3: Update `adapter-design/SKILL.md`**

Add to concept mapping tables:
- `aclose()` lifecycle method for all three ports
- `capabilities` property for AgentPort

- [ ] **Step 4: Update `adapter-test/SKILL.md`**

Add cold test:
- C-new: Verify `hasattr(adapter, "aclose")` is True for all three adapter instances
- C-new: Verify `isinstance(agent_adapter.capabilities, PortCapabilities)` for agent adapters

- [ ] **Step 5: Update `adapter-review/SKILL.md`**

Add to review checklist:
- Verify `aclose()` is implemented on all three port implementations
- Verify `capabilities` property exists on agent adapter
- Verify `max_retries` warning is emitted in `with_structured_output` if the adapter doesn't support it

- [ ] **Step 6: Update `create-karenina-adapter/SKILL.md`**

Add to deliverables checklist:
- `aclose()` method on all port implementations
- `capabilities` property on agent adapter

- [ ] **Step 7: Read and update documentation files**

Read and update each doc file:

**`docs/advanced-adapters/ports.md`:**
- Update protocol signatures to include `aclose()` on all three ports
- Add `capabilities` property to AgentPort section

**`docs/advanced-adapters/writing-adapters.md`:**
- Add `aclose()` as required method in the step-by-step guide
- Document `AgentPort.capabilities`
- Note that `max_retries` behavior varies by adapter

**`docs/advanced-adapters/available-adapters.md`:**
- Update Deep Agents row: MCP and tools support is now wired up

**`docs/advanced-adapters/mcp-integration.md`:**
- Document `AsyncExitStack` session management pattern
- Reference Deep Agents as second MCP-capable adapter alongside LangChain

- [ ] **Step 8: Commit**

```bash
git add .claude/skills/ docs/advanced-adapters/
git commit -m "docs: update adapter skills and documentation for protocol changes"
```

---

### Task 15: Final verification

- [ ] **Step 1: Run full test suite**

Run: `cd karenina && uv run pytest tests/ -x -q`
Expected: All ~2705 tests PASS

- [ ] **Step 2: Run adapter-specific tests**

Run: `cd karenina && uv run pytest tests/ -x -q -k "adapter"`
Expected: All PASS

- [ ] **Step 3: Run mypy on ports**

Run: `cd karenina && uv run mypy src/karenina/ports/`
Expected: No errors

- [ ] **Step 4: Run pre-commit hooks**

Run: `cd karenina && uv run pre-commit run --all-files`
Expected: All PASS

- [ ] **Step 5: Verify issue-specific checks**

Spot-check key behaviors:
- `hasattr(DeepAgentsLLMAdapter(...), "aclose")` is True
- `_detect_tool_error("No errors found")` returns False
- `wrap_sdk_error(RuntimeError("max_turns exceeded"))` returns error with limit_reached
- `wrap_deep_agents_error(RuntimeError("failed to parse MCP configuration"))` does NOT return AgentResponseError
- Deep Agents `_RUBRIC_TASKS` has 6 entries

- [ ] **Step 6: Commit any fixups from verification**

If anything fails, fix and commit with descriptive message.
