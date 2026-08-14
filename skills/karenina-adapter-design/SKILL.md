---
name: karenina-adapter-design
description: Design a new karenina adapter based on gathered context. Produces a spec document with concept mapping and registry integration plan. Use as Phase 2 of adapter creation.
---

# Phase 2: Design Karenina Adapter

Produce a design spec from the context document created in Phase 1.

## Prerequisites

A context document at `docs/adapters/<name>-context.md` from `/karenina-adapter-gather-context`.

## Process

### 1. Read the context document

Load the context document and extract:
- SDK capabilities (async, MCP, tools, traces, structured output)
- Preliminary concept mapping
- Resolved design questions (deep agent, interface name, etc.)

### 2. Determine adapter characteristics

Based on the context, decide:

| Characteristic | Options |
|---------------|---------|
| `agent_tier` | `"deep_agent"` if SDK handles its own tool loop (e.g., Deep Agents, Claude Code); `"tool_loop"` if karenina orchestrates turns |
| `supports_mcp` | `True` if SDK can connect to MCP servers |
| `supports_tools` | `True` if SDK supports tool definitions |
| `fallback_interface` | `None` (require explicit install) or `"langchain"` (silent fallback) |

### 3. Create the concept mapping table

Map every karenina port concept to the SDK equivalent. Cover ALL three ports:

**AgentPort** (multi-turn agent loops):

| karenina concept | SDK equivalent |
|-----------------|----------------|
| `AgentConfig.max_turns` | ??? |
| `AgentConfig.system_prompt` | ??? |
| `AgentConfig.timeout` | ??? |
| `AgentConfig.workspace_path` | ??? |
| `MCPServerConfig` | ??? |
| `AgentResult.raw_trace` | ??? |
| `AgentResult.trace_messages` | ??? |
| `AgentResult.usage` | ??? |
| `AgentResult.limit_reached` | ??? |
| `AgentResult.timeout_reached` | ??? (set to True when a wall-clock timeout interrupts the run and partial trace is recovered) |
| `AgentResult.turns` | ??? |
| `AgentResult.actual_model` | ??? |
| `aclose()` | ??? (required: releases agent resources, MCP sessions, etc.) |
| `capabilities` property | ??? (required: returns `PortCapabilities`) |

**LLMPort** (single-turn calls for parsing/rubric):

| karenina concept | SDK equivalent |
|-----------------|----------------|
| `LLMPort.ainvoke(messages) -> LLMResponse` | ??? (single LLM call, no agent loop) |
| `LLMResponse.content` (string) | ??? |
| `LLMResponse.usage` (UsageMetadata) | ??? |
| `LLMResponse.is_partial` | ??? (defaults to False; set to True when a streaming call was cut short by timeout) |
| `LLMResponse.usage_unavailable` | ??? (defaults to False; set to True when token counts could not be captured, e.g., streaming timeout dropped the final chunk) |
| `with_structured_output(schema)` | ??? (returns new LLMPort with schema constraint) |
| `astream(messages)` | ??? (async context manager yielding `StreamingLLMResponse`; if the SDK has no streaming API, raise `NotImplementedError` and leave `capabilities.supports_streaming=False`) |
| `stream_invoke(messages, timeout)` | ??? (sync wrapper that streams and returns an `LLMResponse` with `is_partial=True` if the wall-clock timeout fires) |
| `capabilities.supports_system_prompt` | ??? |
| `capabilities.supports_structured_output` | ??? |
| `capabilities.supports_streaming` | ??? (True only if `astream()` and `stream_invoke()` are fully implemented) |
| `aclose()` | ??? (required: releases client resources) |

Key design question: does the LLMPort reuse the agent runtime (simpler but heavier) or use a direct LLM call (lighter but separate code path)?

**ParserPort** (structured extraction):

| karenina concept | SDK equivalent |
|-----------------|----------------|
| `aparse_to_pydantic(messages, schema) -> ParsePortResult[T]` | ??? |
| `ParsePortResult.parsed` (Pydantic model) | ??? |
| `ParsePortResult.usage` (UsageMetadata) | ??? |
| `aclose()` | ??? (required: releases parser resources) |

Key design question: does `with_structured_output()` preserve the raw LLM response for usage tracking, or discard it? If discarded, use `include_raw=True` or equivalent.

### 4. Design the core flow

Document the `arun()` flow step by step:
1. Message conversion (karenina Message -> SDK format)
2. MCP server setup (if applicable)
3. Model initialization
4. **Backend/workspace configuration** (CRITICAL: if the SDK has built-in file tools, ensure they access the real filesystem, not a virtual state)
5. Agent creation and invocation
6. Result extraction (traces, usage, final response)
7. **Usage aggregation** (must sum across all turns for multi-turn agents)
8. Error handling (including recursion limit with empty/partial state)

### 4b. Address known pitfall areas

These are areas where adapters commonly have silent bugs. The design must explicitly address each:

| Area | What to decide | Why it matters |
|------|---------------|----------------|
| **Filesystem backend** | If the SDK defaults to virtual/in-memory state, the design must specify switching to real filesystem | Agent sees empty directories, generates synthetic data instead of using real workspace files |
| **Usage in structured output** | If `with_structured_output()` discards the raw LLM response, the design must specify `include_raw=True` or equivalent | ParserPort returns `usage.total_tokens=0`, breaking cost reporting |
| **Recursion limit handling** | What happens when the limit is hit and the SDK returns empty state? | Adapter raises an unexpected error instead of returning `AgentResult(limit_reached=True)` |
| **Multi-turn usage aggregation** | How to sum tokens across all turns, not just the last one | A 10-turn run reporting only the last turn's tokens |
| **Structured output return type** | Does `with_structured_output()` return a Pydantic model, a dict, or a JSON string? The adapter must serialize to JSON regardless. | Callers do `json.loads(response.content)` and crash on Python repr strings like `MyModel(field='value')` |
| **Turn limit mapping** | What is the SDK's parameter for iteration limits, and what is the mapping ratio? (e.g., LangGraph `recursion_limit = max_turns * 2`) | `AgentConfig.max_turns` is accepted but never enforced; the agent runs indefinitely |
| **No-tools fallback** | What happens when `tools=None` and `mcp_servers=None`? Must the adapter fall back to single-turn, use built-in tools, or refuse? | Adapter crashes when called without tools, or the no-tools path bypasses timeout and other guards |
| **Timeout on all paths** | If the adapter has multiple execution paths (tool loop vs single-turn fallback), do all paths honor `config.timeout`? | The no-tools fallback bypasses `asyncio.wait_for()`, making timeout ineffective |
| **MCP tool category** | Does the adapter need external MCP tools for multi-turn tests, or does it have built-in filesystem tools? | Hot tests fail or are skipped because the adapter has no tools to exercise the agent loop |
| **Agent timeout partial recovery** | When `asyncio.wait_for` cancels the agent run, can the adapter recover the partial trace (messages collected so far) and return `AgentResult(timeout_reached=True, ...)`, or must it raise `AgentTimeoutError`? Recovery is strongly preferred: it lets the pipeline inspect how far the agent got. Raising is acceptable only when no messages were collected at all, or when the SDK provides no way to read partial state. | Users see opaque timeout errors instead of a partial trace they can debug; downstream stages lose the chance to score a partial response. |
| **Streaming support** | Does the LLM adapter implement `astream()` and `stream_invoke()` with partial-content capture on timeout? If the SDK has no streaming API, the adapter can raise `NotImplementedError` from both methods and leave `capabilities.supports_streaming=False`. If streaming exists, the adapter must set `LLMResponse.is_partial=True` and `LLMResponse.usage_unavailable=True` when the wall-clock timeout interrupts the stream. | The pipeline cannot capture partial LLM output on timeouts (e.g., answer generation via LLMPort), losing whatever tokens were produced before the cutoff. |

### 5. Invoke brainstorming for non-trivial decisions

If the concept mapping reveals ambiguity (multiple valid approaches), invoke `/superpowers:brainstorming` to explore options with the user.

### 6. Produce the spec document

Save to `docs/adapters/<name>-adapter-design.md`. If existing adapter design specs exist in `docs/adapters/` or `docs/superpowers/specs/`, use them as a structural reference.

Required sections:
1. Overview
2. Motivation
3. Design Decisions (table)
4. Adapter Package Structure
5. Registration (AdapterSpec code, including `requires_provider` field)
6. Agent Adapter Core Flow
7. Concept Mapping Table
8. LLM Adapter
9. Parser Adapter
10. Message Converter
11. Trace Extraction
12. Usage Extraction
13. MCP Conversion
14. Error Mapping
15. Dependencies
16. Integration Points (conditional on distribution model: Manual requires no files to edit; Built-in requires one import in `_load_builtins()`; Plugin requires an entry point in `pyproject.toml`)
17. Open Questions (with defaults)
18. Success Criteria

## Next Step

Run `/karenina-adapter-implement` to begin Phase 3 using this spec.
