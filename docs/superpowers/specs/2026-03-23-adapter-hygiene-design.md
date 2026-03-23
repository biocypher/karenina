# Chunk 10: Adapter & Port Protocol Hygiene

**Date**: 2026-03-23
**Status**: Approved
**Priority**: MEDIUM
**Scope**: 15 issues across 4 adapters, 3 port protocols, and cross-adapter behavior

## Summary

Fix fragile heuristics, missing protocol methods, operator precedence bugs, import-time side effects, and dead MCP integration code across the LangChain, Claude SDK, Claude Tool, and Deep Agents adapters, plus the port protocol layer.

## Design Decisions

### D1: Deep Agents MCP/Tools (091/092)

**Decision**: Wire up MCP/tools support for real (not set to False).

- Keep `supports_mcp=True` and `supports_tools=True` in registration
- Restructure `convert_mcp_to_tools()` to accept an `AsyncExitStack` parameter
- Mirror the LangChain adapter's `arun()` pattern: `AsyncExitStack`, deferred error capture, persistent MCP sessions
- Ensures sessions are properly cleaned up on exit (same ExceptionGroup-safe pattern as LangChain)

### D2: Tool Error Detection (067)

**Decision**: Remove the heuristic entirely (structural signals only).

- `_detect_tool_error()` returns `False` always or is removed
- LangChain `ToolMessage` has no structural error indicator; accept that some tool errors go undetected in traces
- Eliminates all false positives ("No errors found", "timeout set to 30 seconds")

### D3: SDK Limit Detection (068)

**Decision**: Route through `wrap_sdk_error()`.

- Add a turn/recursion limit case to `wrap_sdk_error()` that returns `AgentExecutionError` with `limit_reached=True` metadata
- In `arun()`, replace the `except Exception` block with:

```python
except TimeoutError as e:
    raise AgentTimeoutError(...) from e
except Exception as e:
    translated = wrap_sdk_error(e)
    if isinstance(translated, AgentExecutionError) and getattr(translated, "limit_reached", False):
        limit_reached = True
        logger.warning("Agent hit turn limit: %s", e)
    else:
        raise translated from e
```

- `wrap_sdk_error()` checks SDK exception type first, then falls back to string matching as last resort
- Centralizes all SDK error translation in one place

### D4: MCP Config Format (069)

**Decision**: Include in this chunk.

- Replace key-sniffing (`type` or `command` presence) with `isinstance` checks against `MCPServerConfig` TypedDicts from the port layer

### D5: Schema Format Instructions (086)

**Decision**: Use `model_json_schema()` directly, after verifying equivalence.

- Write an auxiliary script (`scripts/compare_schema_formats.py`) to compare both outputs side by side
- If outputs are functionally equivalent (same fields, types, descriptions), use `json.dumps(RawAnswer.model_json_schema(), indent=2)`
- If outputs differ meaningfully, use `RawAnswer.model_fields` to build a format string that preserves the original structure
- Fallback: if neither approach preserves equivalence, keep `PydanticOutputParser` and document why
- Drop `langchain_core` import from both `deep_judgment.py` and `evaluator.py`
- Clean up the auxiliary script after verification

### D6: MCP Session Pattern (092)

**Decision**: Mirror the LangChain adapter's `AsyncExitStack` + deferred error pattern.

- Same session lifecycle and cleanup guarantees as `langchain/agent.py:350-432`
- `async with AsyncExitStack() as exit_stack` in `arun()`
- Pass `exit_stack` to MCP conversion function
- Deferred error capture to prevent ExceptionGroup wrapping during cleanup

### D7: Execution Approach

**Decision**: Sequential by dependency order.

- ~15 focused commits, each reviewable in isolation
- Respects dependency chain: 095 -> 091/092 -> 138 -> 139

## Execution Order

Dependencies require: **095 → 091/092 → 138 → 139** (strict order).

All remaining issues (068, 093, 067, 086, 090, 094, 069, 142) are independent of each other and of 139. They can be done in any order after 138 completes. The linear sequence below is for reviewer convenience:

```
095 → 091/092 → 138 → 139 → {068, 093, 067, 086, 090, 094, 069, 142}
```

## Issue Specifications

### Group 1: Deep Agents Foundation

#### 095: Add `aclose()` to `DeepAgentsLLMAdapter`

**File**: `adapters/langchain_deep_agents/llm.py`

Add `async def aclose(self) -> None` with docstring noting no resources to clean up (model created per call). Follows the pattern of all other adapters.

#### 091: Wire up MCP/Tools in Deep Agents Agent

**File**: `adapters/langchain_deep_agents/agent.py`, `adapters/langchain_deep_agents/registration.py`

- Remove `# noqa: ARG002` comments from `tools` and `mcp_servers` params in `arun()`
- In `arun()`: create `AsyncExitStack`, convert MCP servers to tools via `convert_mcp_to_tools(exit_stack)`, combine with explicitly passed tools, pass combined tools to agent execution
- Use deferred error capture pattern from LangChain adapter

#### 092: Fix `convert_mcp_to_tools()` Session Lifetime

**File**: `adapters/langchain_deep_agents/mcp.py`

- Change signature: `convert_mcp_to_tools(mcp_servers, exit_stack: AsyncExitStack)`
- Replace `async with MultiServerMCPClient` with exit_stack-managed session creation
- Sessions remain open for the duration of the caller's exit_stack

### Group 2: Port Protocols

#### 138: Add `aclose()` to Port Protocols

**Files**: `ports/llm.py`, `ports/agent.py`, `ports/parser.py`

Add `async def aclose(self) -> None: ...` to `LLMPort`, `AgentPort`, and `ParserPort`. This formalizes the existing convention: all adapters already implement `aclose()` (including Deep Agents after 095). The protocol addition makes the contract explicit for static type checkers and new adapter authors. The registry's `hasattr` check remains as a backward-compatibility safety net for any third-party adapters that may not yet implement the method.

#### 139: Add `capabilities` to `AgentPort`

**File**: `ports/agent.py`

Add `@property def capabilities(self) -> PortCapabilities: return PortCapabilities()` to `AgentPort`, matching `LLMPort` and `ParserPort`.

### Group 3: Error Handling

#### 068: Centralize SDK Error Translation

**Files**: `adapters/claude_agent_sdk/agent.py`, `adapters/claude_agent_sdk/errors.py`

- In `errors.py`: add a case for turn/recursion limit detection (check SDK exception types first, fall back to structural signals like exit codes)
- In `agent.py`: replace the `except Exception` block that does string matching with a call to `wrap_sdk_error(e)`, then check the returned error type for limit-related errors

#### 093: Fix Operator Precedence

**File**: `adapters/langchain_deep_agents/errors.py`

Change line 40 from:
```python
if "parse" in error_str or "output" in error_str and "format" in error_str:
```
to:
```python
if ("parse" in error_str and "output" in error_str) or ("output" in error_str and "format" in error_str):
```

Ensures "parse" alone is not sufficient to trigger `AgentResponseError` classification.

### Group 4: Heuristic & Import Cleanup

#### 067: Remove `_detect_tool_error` Heuristic

**File**: `adapters/langchain/trace.py`

Remove or neutralize `_detect_tool_error()`. The function's callers should default to `is_error=False` for tool messages. No structural error signal exists in LangChain's `ToolMessage`, so we accept the trade-off of undetected tool errors over false positives.

#### 086: Replace `PydanticOutputParser` with Native Pydantic v2

**Files**: `benchmark/verification/evaluators/template/deep_judgment.py`, `benchmark/verification/evaluators/template/evaluator.py`

- First: write auxiliary script comparing both outputs; verify equivalence
- In `deep_judgment.py`: replace `PydanticOutputParser(pydantic_object=RawAnswer).get_format_instructions()` with `json.dumps(RawAnswer.model_json_schema(), indent=2)` (or equivalent)
- In `evaluator.py`: remove the `PydanticOutputParser` import and `format_instructions` generation entirely (unused by `deep_judgment_parse()`)
- Both files drop the `langchain_core` import

#### 090: Remove Module-Level `load_dotenv()`

**Files**: `adapters/claude_tool/llm.py`, `adapters/claude_tool/agent.py`

Delete `load_dotenv()` calls and `from dotenv import load_dotenv` imports. The caller is responsible for environment setup.

### Group 5: Missing Registrations & Warnings

#### 094: Add Missing Task Registration

**File**: `adapters/langchain_deep_agents/prompts/rubric.py`

Add `"rubric_dynamic_presence_check"` to the `_RUBRIC_TASKS` list, matching Claude Tool and Claude SDK which both register 6 tasks.

#### 069: Fix MCP Config Format Detection

**File**: `adapters/claude_agent_sdk/agent.py`

Replace key-sniffing (`type` or `command` presence) in `_convert_mcp_servers()` with `isinstance` checks against `MCPServerConfig` TypedDicts (`MCPStdioServerConfig`, `MCPHttpServerConfig`) from the port layer.

#### 142: Warn When `max_retries` Is Ignored

**Files**: `adapters/claude_agent_sdk/llm.py`, `adapters/langchain_deep_agents/llm.py`, `ports/llm.py`

- In both adapters' `with_structured_output()`: when `max_retries` is not None, emit `logger.warning("max_retries=%d ignored by %s adapter; retry behavior is managed internally", max_retries, adapter_name)`
- Update `LLMPort.with_structured_output()` docstring to note adapter support varies

### Group 6: Documentation & Skill Updates

After all code changes are complete and tests pass, review and update the adapter authoring skills and documentation to reflect the new protocol requirements and patterns.

#### Adapter Creation Skills

Five skills in `.claude/skills/` guide new adapter authors through the lifecycle. These must reflect the protocol changes from 138/139 and the MCP session pattern from 091/092:

| Skill | Path | Updates Needed |
|-------|------|----------------|
| `create-karenina-adapter` | `.claude/skills/create-karenina-adapter/SKILL.md` | Mention `aclose()` as required protocol method, `capabilities` on all three ports |
| `adapter-design` | `.claude/skills/adapter-design/SKILL.md` | Add `aclose()` and `AgentPort.capabilities` to concept mapping tables |
| `adapter-implement` | `.claude/skills/adapter-implement/SKILL.md` | Add `aclose()` to implementation checklist, document AsyncExitStack pattern for MCP session management |
| `adapter-test` | `.claude/skills/adapter-test/SKILL.md` | Add cold test for `aclose()` protocol conformance, verify `capabilities` on agent adapters |
| `adapter-review` | `.claude/skills/adapter-review/SKILL.md` | Add `aclose()` and `capabilities` to review checklist |

#### Documentation

| Doc | Path | Updates Needed |
|-----|------|----------------|
| `writing-adapters.md` | `docs/advanced-adapters/writing-adapters.md` | Add `aclose()` as required method, document `AgentPort.capabilities`, note `max_retries` adapter-dependent behavior |
| `ports.md` | `docs/advanced-adapters/ports.md` | Update protocol signatures to include `aclose()`, add `capabilities` to `AgentPort` section |
| `available-adapters.md` | `docs/advanced-adapters/available-adapters.md` | Update Deep Agents capability matrix (MCP/tools now wired up) |
| `mcp-integration.md` | `docs/advanced-adapters/mcp-integration.md` | Document AsyncExitStack session pattern, reference Deep Agents as second MCP-capable adapter |

## Verification

- Run full test suite: `cd karenina && uv run pytest tests/ -x -q`
- Run adapter-specific tests: `uv run pytest tests/ -x -q -k "adapter"`
- Run mypy on ports: `uv run mypy src/karenina/ports/`
- Run pre-commit hooks (ruff, vulture)
- Issue-specific checks listed in TASK.md verification section

## Files Modified

| File | Issues |
|------|--------|
| `adapters/langchain/trace.py` | 067 |
| `adapters/claude_agent_sdk/agent.py` | 068, 069 |
| `adapters/claude_agent_sdk/errors.py` | 068 |
| `adapters/claude_agent_sdk/llm.py` | 142 |
| `adapters/claude_tool/llm.py` | 090 |
| `adapters/claude_tool/agent.py` | 090 |
| `adapters/langchain_deep_agents/registration.py` | 091 |
| `adapters/langchain_deep_agents/agent.py` | 091 |
| `adapters/langchain_deep_agents/mcp.py` | 092 |
| `adapters/langchain_deep_agents/errors.py` | 093 |
| `adapters/langchain_deep_agents/prompts/rubric.py` | 094 |
| `adapters/langchain_deep_agents/llm.py` | 095, 142 |
| `benchmark/verification/evaluators/template/deep_judgment.py` | 086 |
| `benchmark/verification/evaluators/template/evaluator.py` | 086 |
| `ports/llm.py` | 138, 142 |
| `ports/agent.py` | 138, 139 |
| `ports/parser.py` | 138 |
| `.claude/skills/create-karenina-adapter/SKILL.md` | docs |
| `.claude/skills/adapter-design/SKILL.md` | docs |
| `.claude/skills/adapter-implement/SKILL.md` | docs |
| `.claude/skills/adapter-test/SKILL.md` | docs |
| `.claude/skills/adapter-review/SKILL.md` | docs |
| `docs/advanced-adapters/writing-adapters.md` | docs |
| `docs/advanced-adapters/ports.md` | docs |
| `docs/advanced-adapters/available-adapters.md` | docs |
| `docs/advanced-adapters/mcp-integration.md` | docs |
