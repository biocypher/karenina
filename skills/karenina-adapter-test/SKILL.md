---
name: karenina-adapter-test
description: Test a new karenina adapter using conformance suite and adapter-specific tests. Covers cold tests (mocked) and hot tests (live API). Use as Phase 4 of adapter creation.
---

# Phase 4: Test Karenina Adapter

Validate the adapter through systematic testing that covers every capability it claims to support. The test strategy is adapter-agnostic: it defines what to test by observable behavior, not by SDK internals.

## Prerequisites

- Adapter package implemented from `/karenina-adapter-implement`
- Registration complete (adapter appears in `AdapterRegistry.get_spec()`)

## Core testing principle

**For each capability the adapter claims to support, test it through observable behavior.** Do not rely on inspecting SDK internals (class names, private attributes, backend types). Instead, verify that the adapter produces correct output given specific inputs.

The most dangerous bugs are silent: the adapter appears to work for simple cases but fails for real benchmarking scenarios. The test strategy is specifically designed to catch these.

---

## Cold tests (no API calls, all mocked)

### C1: Protocol conformance

Run the shared conformance suite. This validates that every adapter satisfies the port protocols.

```bash
cd karenina && uv run pytest tests/unit/adapters/conformance/ -x -v
```

Wire your adapter in by creating `tests/unit/adapters/<name>/conftest.py`:

```python
"""Fixtures for <Name> adapter tests."""
from __future__ import annotations

import pytest
from karenina.schemas.config import ModelConfig


@pytest.fixture
def <name>_model_config() -> ModelConfig:
    return ModelConfig(
        id="test-<name>",
        model_name="claude-sonnet-4-6",
        model_provider="anthropic",
        interface="<interface>",
        temperature=0.0,
    )
```

Then for each port, create adapter instances in test files or in the conftest. The conformance suite at `tests/unit/adapters/conformance/` expects fixtures named `deep_agents_agent_adapter`, etc. (adapter-specific naming).

**Checks:** isinstance (runtime_checkable), method signatures, return types, `aclose()` lifecycle, `capabilities` property on all three ports, presence of `astream()` and `stream_invoke()` on the LLMPort class (they may raise `NotImplementedError` if streaming is not supported, but the attributes must exist).

### C1b: aclose() conformance

Verify that `aclose()` is implemented and callable on all three port adapters. This is a required protocol method:

```python
# AgentPort
agent = create_agent(config)
await agent.aclose()  # Must not raise

# LLMPort
llm = create_llm(config)
await llm.aclose()  # Must not raise

# ParserPort
parser = create_parser(config)
await parser.aclose()  # Must not raise
```

Calling `aclose()` twice must be safe (idempotent).

### C1c: capabilities conformance for agent adapters

Verify that the agent adapter exposes a `capabilities` property returning `PortCapabilities`, matching LLMPort and ParserPort:

```python
from karenina.ports.capabilities import PortCapabilities

agent = create_agent(config)
caps = agent.capabilities
assert isinstance(caps, PortCapabilities)

# LLM adapter capabilities must also declare streaming support.
llm = create_llm(config)
llm_caps = llm.capabilities
assert isinstance(llm_caps, PortCapabilities)
assert isinstance(llm_caps.supports_streaming, bool)
```

### C1d: streaming methods on LLMPort

Every LLM adapter must define `astream()` and `stream_invoke()`, even when the SDK has no streaming API (in which case they raise `NotImplementedError`). Verify the attributes exist and behave consistently with the declared capability:

```python
llm = create_llm(config)
assert callable(getattr(llm, "astream", None))
assert callable(getattr(llm, "stream_invoke", None))

if llm.capabilities.supports_streaming:
    # Real streaming: astream() must yield a StreamingLLMResponse
    from karenina.ports.llm import StreamingLLMResponse
    async with llm.astream([Message.user("hi")]) as sr:
        assert isinstance(sr, StreamingLLMResponse)
else:
    # No streaming: both methods must raise NotImplementedError
    with pytest.raises(NotImplementedError):
        llm.stream_invoke([Message.user("hi")], timeout=1.0)
```

### C2: Message conversion roundtrip

Verify: `Message.user("X")` -> provider format -> back to `Message` preserves content and role for all message types (user, assistant, system, tool).

### C3: Trace format

Mock the SDK to return a conversation with tool calls. Verify the raw_trace contains standard delimiters (`--- AI Message ---`, `--- Tool Call ---`, `--- Tool Result ---`) and that trace_messages preserves ordering.

### C4: Registration integrity

Verify `AdapterRegistry.get_spec("<interface>")` returns a spec with:
- All three factories (agent, llm, parser) non-None
- `supports_mcp`, `supports_tools` flags set correctly; `agent_tier` is `"tool_loop"` or `"deep_agent"`
- `fallback_interface` is either None or a registered interface

> If testing a standalone plugin before packaging, import the registration module directly: `import karenina_crewai.registration`

### C5: Configuration passthrough

This is where most silent bugs hide. Mock the SDK's agent creation function to **capture its arguments**, then verify that karenina's `AgentConfig` fields actually reach the SDK:

```python
captured_kwargs = {}

def capture_create(**kwargs):
    captured_kwargs.update(kwargs)
    return mock_agent  # returns mocked agent that produces a valid result

monkeypatch.setattr("<adapter_module>.<create_fn>", capture_create)

# Test 1: system_prompt passes through
await adapter.arun(
    messages=[Message.system("Be helpful."), Message.user("Hi")],
    config=AgentConfig(max_turns=5),
)
assert "system_prompt" in captured_kwargs or <verify SDK received it>

# Test 2: max_turns maps to SDK's limit mechanism
assert <verify recursion_limit or max_turns in captured_kwargs>

# Test 3: workspace_path configures real filesystem access
await adapter.arun(
    messages=[Message.user("test")],
    config=AgentConfig(max_turns=2, workspace_path=Path("/tmp/workspace")),
)
assert <verify backend/cwd/workspace configured for real filesystem>
```

**Patch target note**: If the adapter uses lazy imports inside methods (e.g., `from some_sdk import create_agent` inside `arun()`), patch the *source* module (`some_sdk.create_agent`), not the adapter module. `unittest.mock.patch` can only intercept attributes that already exist at module level.

**Why C5 matters:** Without this, an adapter can pass all other tests while silently ignoring configuration. The agent runs with wrong limits, no system prompt, or a virtual filesystem.

### C6: Error mapping

Mock the SDK to raise each type of error and verify karenina gets the correct exception:

| SDK behavior | Expected karenina exception |
|-------------|---------------------------|
| Recursion/turn limit exceeded | `AgentExecutionError` with `limit_reached=True` |
| Timeout | `AgentTimeoutError` |
| Malformed response | `AgentResponseError` |
| General failure | `AgentExecutionError` |

### C7: Usage extraction from mocked responses

Mock the SDK to return responses with known token counts. Verify the adapter extracts and aggregates them correctly. This catches adapters that return zero tokens, lose multi-turn aggregation, or drop the model name.

```python
# Mock a multi-turn conversation with known token counts per turn
mock_messages = [
    make_ai_message(content="Turn 1", input_tokens=100, output_tokens=20),
    make_tool_message(content="tool result"),
    make_ai_message(content="Turn 2", input_tokens=150, output_tokens=30),
]
# (make_ai_message adds usage to response_metadata or usage_metadata,
#  matching whatever the SDK provides)

result = await adapter.arun(...)

# Usage must aggregate across ALL turns, not just the last one
assert result.usage.input_tokens == 250, "Must sum input across turns"
assert result.usage.output_tokens == 50, "Must sum output across turns"
assert result.usage.total_tokens == 300
assert result.usage.model is not None, "Model name must be populated"
```

**Why C7 matters:** If usage is wrong, cost reporting is wrong. If only the last turn's tokens are reported, a 10-turn agent run that cost $0.50 shows as $0.02. Multi-turn aggregation is easy to get wrong.

For LLMPort and ParserPort, verify their results also carry usage:

```python
# LLMPort
response = await llm.ainvoke(messages)
assert response.usage.total_tokens > 0

# ParserPort
result = await parser.aparse_to_pydantic(messages, schema)
assert result.usage.total_tokens > 0
```

### C8: Streaming timeout partial capture (LLMPort)

For adapters that declare `supports_streaming=True`, verify that `stream_invoke()` (or its underlying `_astream_with_timeout()` helper) captures partial content when the stream is cut short, and flips both `is_partial` and `usage_unavailable` on the returned `LLMResponse`. The canonical test pattern lives in `tests/unit/adapters/<name>/test_streaming.py`:

```python
# Mock the SDK's streaming API to yield a few chunks fast then sleep
async def _fake_astream(messages):
    yield fake_chunk("First")
    yield fake_chunk("Second")
    await asyncio.sleep(10.0)  # blocks past the timeout
    yield fake_chunk("Never reached")

# Build the adapter with the mocked model (pattern differs by SDK; see
# test_streaming.py in the langchain, claude_tool, claude_agent_sdk, and
# langchain_deep_agents test directories for working examples).

result = await adapter._astream_with_timeout([Message.user("Hi")], timeout=1.0)

assert result.is_partial is True
assert result.usage_unavailable is True
assert "First" in result.content
assert "Second" in result.content
assert "Never reached" not in result.content
```

Also add a happy-path test that completes before the timeout and verifies both flags stay `False`. Adapters that do not support streaming skip C8 and instead assert that both methods raise `NotImplementedError` (covered by C1d).

### C9: Agent timeout partial recovery (AgentPort)

When the agent run hits a wall-clock timeout, the adapter should prefer returning a partial `AgentResult(timeout_reached=True)` over raising `AgentTimeoutError`. Mock the SDK's agent invocation so the first few tool responses arrive quickly and subsequent ones block past `config.timeout`, then verify the adapter:

1. returns an `AgentResult` (does not raise) when at least one message was accumulated
2. sets `result.timeout_reached is True`
3. includes the fast responses in `result.trace_messages` and `result.raw_trace`
4. appends a marker like `"[Note: Agent timed out"` to `raw_trace`
5. reports `result.usage.input_tokens > 0` for the turns that did complete

Working examples live in `tests/unit/adapters/langchain/test_agent_timeout.py` and `tests/unit/adapters/claude_tool/test_agent_timeout.py`. Also cover the empty-state case: when no messages were collected at all, the adapter must raise `AgentTimeoutError` with a message like `"timed out ... with no messages"`.

If the adapter's SDK exposes no way to read partial state (e.g., Deep Agents, Claude Agent SDK), document this in the adapter docs and fall back to raising `AgentTimeoutError` on every timeout. C9 then becomes: assert that a short timeout raises `AgentTimeoutError` and that no partial `AgentResult` is returned.

---

## Hot tests (live API, require user approval)

Ask the user before running these. They cost real API tokens.

**Before running hot tests, ask the user (via AskUserQuestion):**
1. Which model to use (e.g., `claude-haiku-4-5`, `claude-sonnet-4-6`, `gpt-4.1`)
2. Which provider (e.g., `anthropic`, `openai`, `google_genai`)
3. Confirm they have the API key configured

Then build the config:

```python
config = ModelConfig(
    id="hot-test",
    model_name="<user-chosen-model>",
    model_provider="<user-chosen-provider>",
    interface="<adapter-interface>",
)
agent = get_agent(config, auto_fallback=False)
llm = get_llm(config, auto_fallback=False)
parser = get_parser(config, auto_fallback=False)
```

**IMPORTANT: Run ALL hot tests, not just the simple one. Each tests a different failure mode.**

### Local MCP test server for adapters without built-in tools

Some adapters need external tools for agent tests (H1-H4, H8, H9). Create a minimal MCP server using FastMCP and start it before running hot tests:

```python
# scripts/mcp_test_server.py
from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP(
    "TestFileServer",
    stateless_http=True,
    json_response=True,
    host="127.0.0.1",
    port=8321,
)

@mcp.tool()
def read_file(path: str) -> str:
    """Read the contents of a file at the given path."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"

@mcp.tool()
def list_directory(path: str = ".") -> str:
    """List files and directories at the given path."""
    try:
        return "\n".join(sorted(os.listdir(path)))
    except FileNotFoundError:
        return f"Error: directory not found: {path}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

Start it as a subprocess and wait for port 8321 to accept connections before running tests. Pass the MCP config to adapters that need it:

```python
MCP_SERVER_URL = "http://127.0.0.1:8321/mcp"

# Which adapters need MCP vs have built-in tools:
# langchain: tool-loop adapter, NEEDS MCP or explicit tools
# claude_tool: Anthropic SDK tool_runner, has built-in tools
# claude_agent_sdk: Claude CLI, has native filesystem tools
# langchain_deep_agents: FilesystemBackend, has built-in tools
NEEDS_MCP = {"langchain"}

def mcp_servers_for(interface):
    if interface in NEEDS_MCP:
        return {"test_server": {"type": "http", "url": MCP_SERVER_URL}}
    return None
```

**Adapter tool categories (new adapters must declare which category they fall into):**

| Category | Needs MCP server? | Examples |
|----------|-------------------|----------|
| Tool-loop, no built-in tools | Yes | langchain |
| SDK tool_runner, no built-in tools | Yes (for multi-turn) | claude_tool |
| Has built-in filesystem tools | No | claude_agent_sdk, langchain_deep_agents |

**Known limitations by adapter:**
- **langchain**: MCP tools don't receive `workspace_path` from AgentConfig. File access tests (H2) will fail because the MCP server can't resolve relative paths to the workspace. This is an architectural limitation of the URL-based MCP approach.
- **claude_tool**: Without explicit tools or MCP servers, falls back to single-turn `messages.create()`. File access (H2) and tool trace (H3) tests will fail. Turn limit (H4) won't trigger because there's no tool loop.
- **claude_agent_sdk**: Requires the `claude_agent_sdk` Python package installed. Skip all tests if unavailable.
- **langchain_deep_agents**: Has built-in tools via `FilesystemBackend`. All tests should pass without MCP.

### Absolute paths in MCP test prompts

MCP servers do not receive `workspace_path` from `AgentConfig`. When testing file access with MCP tools, always include the absolute path in the prompt:

```python
# CORRECT: MCP server can resolve absolute path
f"Read {tmpdir}/test_data.txt and tell me the secret value."

# WRONG: MCP server doesn't know workspace_path
"Read test_data.txt and tell me the secret value."
```

Adapters with built-in filesystem tools (Deep Agents, Claude SDK) receive `workspace_path` via their own mechanisms and handle relative paths.

### Structured output serialization

Adapters implementing `with_structured_output()` must ensure `LLMResponse.content` contains valid JSON, not Python repr strings. When the underlying framework returns a Pydantic model:

```python
# Correct: serialize to JSON
if isinstance(response, BaseModel):
    content = response.model_dump_json()
elif isinstance(response, dict):
    content = json.dumps(response)

# Wrong: produces "MyModel(field='value')" which is not JSON
content = str(response)
```

### Timeout handling on no-tools path

If the adapter has a single-turn fallback when no tools are provided, the fallback must still honor `AgentConfig.timeout`. Wrap the API call in `asyncio.wait_for()`:

```python
if not tools:
    if config.timeout:
        response = await asyncio.wait_for(
            client.messages.create(**kwargs), timeout=config.timeout
        )
    else:
        response = await client.messages.create(**kwargs)
```

### H1: Simple knowledge query

Verifies basic end-to-end flow without tool use or filesystem.

```python
result = await agent.arun(
    messages=[Message.user("What is 7 * 8? Reply with just the number.")],
    config=AgentConfig(max_turns=5),
)

assert "56" in result.final_response
assert result.turns >= 1
assert result.usage.total_tokens > 0
assert result.limit_reached is False
assert "--- AI Message ---" in result.raw_trace
```

### H2: Workspace file access (CRITICAL)

Verifies the agent can read real files from disk. This catches the most dangerous class of silent failure: the adapter appears to work but the agent sees an empty filesystem and generates synthetic data instead of using real workspace files.

```python
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    test_file = Path(tmpdir) / "test_data.txt"
    test_file.write_text("The secret value is 42.")

    result = await agent.arun(
        messages=[Message.user(
            "Read the file test_data.txt and tell me what the secret value is."
        )],
        config=AgentConfig(max_turns=10, workspace_path=Path(tmpdir)),
    )

    assert "42" in result.final_response, (
        f"Agent could not read workspace file. Response: {result.final_response[:200]}"
    )
```

**Why H2 is critical:** H1 passes even with a broken virtual backend. Only H2 catches the bug where the agent's filesystem tools return empty results for real paths. This bug is silent because LLMs compensate by generating plausible synthetic data.

### H3: Tool use trace

For deep agent adapters, verify the trace captures tool calls. Ask a question that requires the agent to use its built-in tools:

```python
with tempfile.TemporaryDirectory() as tmpdir:
    (Path(tmpdir) / "numbers.csv").write_text("a,b\n1,2\n3,4\n5,6\n")

    result = await agent.arun(
        messages=[Message.user("Read numbers.csv and compute the sum of column a.")],
        config=AgentConfig(max_turns=15, workspace_path=Path(tmpdir)),
    )

    assert "--- Tool Call ---" in result.raw_trace, "Trace should contain tool calls"
    assert "--- Tool Result ---" in result.raw_trace, "Trace should contain tool results"
    assert result.turns >= 2, "Should take multiple turns with tool use"
```

### H4: Turn limit behavior

Set a very low `max_turns` and give a task that requires multiple actions (tool calls), not just lengthy generation. A simple generation prompt may complete in one turn without hitting any limit. The task must force the agent through its tool loop multiple times:

```python
with tempfile.TemporaryDirectory() as tmpdir:
    # Create files that require multiple tool calls to process
    for i in range(5):
        (Path(tmpdir) / f"data_{i}.txt").write_text(f"value={i * 10}")

    result = await agent.arun(
        messages=[Message.user(
            f"Read ALL five data files in {tmpdir} and compute the sum of all values."
        )],
        config=AgentConfig(max_turns=2, workspace_path=Path(tmpdir)),
    )

    assert result.limit_reached is True, "Should hit turn limit with max_turns=2"
```

For adapters without built-in tools (using MCP), use absolute paths in the prompt (see "Absolute paths in MCP test prompts" above).

### H5: LLM single-turn invocation (LLMPort)

The LLMPort is used for judge parsing, rubric evaluation, and deep judgment. These are the most-called paths in a benchmark run. Verify basic invocation and usage tracking:

```python
from karenina.adapters.factory import get_llm

llm = get_llm(config)
response = await llm.ainvoke([
    Message.system("You are a helpful assistant."),
    Message.user("What is 2 + 2? Reply with just the number."),
])

assert isinstance(response, LLMResponse)
assert "4" in response.content
assert response.usage.total_tokens > 0, "Usage must track tokens"
assert response.usage.input_tokens > 0, "Input tokens must be non-zero"
assert response.usage.output_tokens > 0, "Output tokens must be non-zero"
```

**Why H5 matters:** If LLMPort doesn't work, every parsing and rubric evaluation in the pipeline fails. If usage tracking is broken, cost reporting is wrong.

### H6: Structured output parsing (ParserPort)

The ParserPort extracts structured data from LLM responses. This is how the pipeline turns free-form agent traces into typed Pydantic models. Verify it produces valid parsed output:

```python
from pydantic import BaseModel, Field
from karenina.adapters.factory import get_parser

class BookInfo(BaseModel):
    title: str = Field(description="The book title")
    year: int = Field(description="Publication year")

parser = get_parser(config)
result = await parser.aparse_to_pydantic(
    messages=[
        Message.system("Extract structured data from the following text."),
        Message.user(
            "The novel '1984' by George Orwell was first published "
            "in 1949 and remains widely read today."
        ),
    ],
    schema=BookInfo,
)

assert isinstance(result.parsed, BookInfo)
assert "1984" in result.parsed.title
assert result.parsed.year == 1949
assert result.usage.total_tokens > 0, "Parser must track usage"
```

**Why H6 matters:** If parsing fails, the pipeline cannot extract answers from agent traces. Structured output behavior varies significantly across LLM providers (some use tool_use, some use JSON mode, some need prompt engineering). This test catches provider-specific incompatibilities.

### H7: LLM with structured output (with_structured_output)

Verify the LLMPort's structured output path works (used by some pipeline stages):

```python
class YesNo(BaseModel):
    answer: bool = Field(description="True for yes, False for no")
    confidence: float = Field(description="Confidence from 0 to 1")

structured_llm = llm.with_structured_output(YesNo)
response = await structured_llm.ainvoke([
    Message.user("Is the sky blue? Respond with yes/no and confidence."),
])

import json
data = json.loads(response.content)
assert "answer" in data
assert "confidence" in data
```

### H8: Usage tracking across all ports

Verify that every port returns meaningful, consistent usage data from live calls:

```python
# AgentPort: multi-turn usage should be larger than single-turn
agent_result = await agent.arun(
    messages=[Message.user("What color is the sky on a clear day?")],
    config=AgentConfig(max_turns=5),
)
assert agent_result.usage.input_tokens > 0, "Agent must report input tokens"
assert agent_result.usage.output_tokens > 0, "Agent must report output tokens"
assert agent_result.usage.total_tokens == (
    agent_result.usage.input_tokens + agent_result.usage.output_tokens
), "total_tokens must equal input + output"
assert agent_result.usage.model is not None, "Agent must report model name"

# LLMPort: single-turn usage
llm_response = await llm.ainvoke([Message.user("Say hello.")])
assert llm_response.usage.input_tokens > 0
assert llm_response.usage.output_tokens > 0
assert llm_response.usage.total_tokens > 0

# ParserPort: parsing usage
parser_result = await parser.aparse_to_pydantic(
    [Message.system("Extract."), Message.user("The novel 1984 was published in 1949.")],
    BookInfo,
)
assert parser_result.usage.total_tokens > 0, "Parser must report usage"

# Cross-check: LLM single-turn should use fewer tokens than a multi-turn agent
# (not a strict invariant, but a sanity check)
if agent_result.turns > 1:
    assert agent_result.usage.total_tokens > llm_response.usage.total_tokens, (
        "Multi-turn agent should use more tokens than a single LLM call"
    )
```

**Why H8 matters:** Usage data drives cost reporting, budget limits, and billing. If any port returns zeros, the user sees incorrect costs. If total_tokens != input + output, aggregation logic downstream will be inconsistent.

### H9: Error propagation

Verify that adapter errors surface correctly rather than being swallowed. A short timeout must yield one of two well-defined outcomes: a partial `AgentResult(timeout_reached=True)` if at least one message was collected, or an `AgentTimeoutError` if nothing arrived before the cutoff.

```python
from karenina.ports import AgentResult, AgentTimeoutError

try:
    result = await agent.arun(
        messages=[Message.user("Do something complex.")],
        config=AgentConfig(max_turns=50, timeout=0.001),  # impossibly short timeout
    )
    # Partial-recovery adapters (claude_tool, langchain) return a partial result.
    assert isinstance(result, AgentResult), (
        f"Expected AgentResult or AgentTimeoutError, got {type(result).__name__}"
    )
    assert result.timeout_reached is True, (
        "Timeout must set timeout_reached=True on the partial AgentResult"
    )
except AgentTimeoutError:
    pass  # Adapters without partial recovery (claude_agent_sdk, langchain_deep_agents)
          # raise instead, which is also correct.
except Exception as e:
    assert False, f"Expected AgentResult(timeout_reached=True) or AgentTimeoutError, got {type(e).__name__}: {e}"
```

---

## Pass/fail summary

| Test | Port | What it catches | Failure mode if skipped |
|------|------|----------------|----------------------|
| C1: Protocol conformance | All | Missing methods, wrong return types | Runtime errors in pipeline |
| C1d: Streaming methods present | LLM | Missing `astream`/`stream_invoke` | Pipeline crashes when it routes calls through the streaming path |
| C2: Message roundtrip | Agent | Content/role corruption | Wrong prompts sent to LLM |
| C3: Trace format | Agent | Missing delimiters | Broken trace display, regex highlighting |
| C4: Registration | All | Wrong flags, missing factories | Factory errors, wrong pipeline path |
| C5: Config passthrough | Agent | Ignored settings | Wrong limits, no system prompt, virtual filesystem |
| C6: Error mapping | Agent | Unhandled exceptions | Pipeline crashes instead of graceful failure |
| C7: Usage extraction | All | Zero tokens, missing aggregation, no model name | Wrong cost reporting, broken billing |
| C8: Streaming timeout partial capture | LLM | Lost content on stream timeout, missing `is_partial`/`usage_unavailable` flags | Answer-generation stage loses partial output on request timeout |
| C9: Agent timeout partial recovery | Agent | Opaque timeouts, lost partial traces | Users cannot debug why an agent ran out of time |
| H1: Simple query | Agent | Basic invocation broken | Nothing works |
| H2: Workspace access | Agent | Virtual filesystem, agent blind to files | Correct-looking but wrong results (synthetic data) |
| H3: Tool use trace | Agent | Tool calls not captured | Incomplete traces, debugging impossible |
| H4: Turn limit | Agent | Limit detection broken | Runaway agent, wasted tokens |
| H5: LLM invocation | LLM | Single-turn calls broken | All parsing/rubric evaluation fails |
| H6: Structured parsing | Parser | Schema extraction broken | Pipeline cannot extract answers from traces |
| H7: Structured output | LLM | with_structured_output broken | Structured pipeline stages fail |
| H8: Usage tracking | All | Tokens zero, partial, or inconsistent across ports | Wrong cost reporting, budget limits ineffective |
| H9: Error propagation | Agent | Errors swallowed silently, timeout recovery broken | Hanging, opaque errors, or lost partial traces |

**Minimum viable test set:** C1 + C1d + C4 + C5 + C7 + C8 (streaming-capable only) + C9 + H1 + H2 + H5 + H6 + H8. These cover all three ports, streaming and timeout recovery, usage tracking, and catch the most critical bugs across the full pipeline.

---

## Running all tests

```bash
# Cold tests
cd karenina && uv run pytest tests/unit/adapters/<name>/ tests/unit/adapters/conformance/ -x -v

# Hot tests (after user approval)
cd karenina && uv run python -c "<hot test script>"

# Full regression
cd karenina && uv run pytest tests/ -x -q
```

## Next Step

Run `/karenina-adapter-review` to begin Phase 5.
