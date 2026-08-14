---
name: karenina-adapter-gather-context
description: Gather requirements for a new karenina adapter. Collects SDK documentation, capabilities, and maps them to karenina ports. Use as Phase 1 of adapter creation.
---

# Phase 1: Gather Context for Karenina Adapter

Collect all information needed to design and implement a new karenina adapter.

## What you're building

A karenina adapter implements three port interfaces that the verification pipeline uses:

- **AgentPort**: Multi-turn agent loops with tool calling and MCP servers. Used for answer generation (the agent analyzes data, writes code, uses tools). This is the primary interface.
- **LLMPort**: Simple single-turn LLM calls without agent loops. Used for judge parsing, rubric evaluation, and deep judgment. Called many times per benchmark run.
- **ParserPort**: LLM-based structured output extraction. Takes free-form text and extracts typed Pydantic models. Used to parse agent traces into answer templates.

All three must be implemented. Read their protocol definitions at `karenina/src/karenina/ports/` (agent.py, llm.py, parser.py) to understand exact method signatures.

## Process

### 1. Identify the target SDK

Ask the user (via AskUserQuestion) for:
- **SDK name**: The framework or SDK to integrate (e.g., "LangChain Deep Agents", "CrewAI", "AutoGen")
- **Documentation URL**: Primary docs or GitHub repository
- **PyPI package name**: The pip-installable package

### 2. Fetch SDK documentation

Use WebFetch to retrieve:
- The SDK's main README or overview page
- API reference for agent creation, invocation, and tool use
- Any trace/observability documentation
- Message type documentation

### 3. Identify key capabilities

For each capability, determine if the SDK supports it:

| Capability | Question to answer |
|-----------|-------------------|
| Agent loops | Does the SDK run multi-turn agent loops with tool calling? |
| Async support | Does the SDK support async invocation (`await`)? |
| MCP integration | Can the SDK connect to MCP servers? How? |
| Tool definition | How are tools defined and passed to the agent? |
| Message types | What message types exist (human, AI, tool, system)? |
| Traces | How is conversation history accessible after invocation? |
| Usage/tokens | How are token counts and costs exposed? Are they per-turn or aggregated? Does structured output preserve usage metadata? |
| Streaming | Does the SDK expose a streaming API (async generator or callback)? Does the final chunk carry usage metadata, or is usage only available from a non-streaming call? Can partial content be captured if the stream is cut short? |
| Structured output | Does the SDK support constrained JSON output? Does `with_structured_output()` return just the parsed result, or can it include the raw LLM response (needed for usage tracking)? What type does it return: a Pydantic model, a dict, or a JSON string? |
| No-tools behavior | What happens when the agent is invoked with no tools and no MCP servers? Does it fall back to single-turn completion, raise an error, or hang? |
| System prompts | How are system prompts provided? |
| Recursion/turn limits | How does the SDK control max iterations? What happens when the limit is hit: does it raise, return partial state, or silently truncate? What is the mapping ratio (e.g., LangGraph uses 2 recursion steps per agent turn)? |
| Agent timeout recovery | If the agent run hits a wall-clock timeout mid-loop, can the SDK expose the partial trace collected so far (messages, tool calls, final assistant output), or does it lose everything? Is there a checkpointer / state accessor that survives an `asyncio.wait_for` cancellation? |
| Filesystem/workspace | Does the SDK have built-in file tools (read, write, ls)? If so, do they operate on real disk or a virtual/in-memory state? Can the working directory be configured? |
| Backend/runtime | Does the SDK use an abstract backend (e.g., virtual state vs real filesystem)? What is the default? Can it be switched? |

### 4. Ask targeted clarifying questions

**Before asking**: Review the user's original message for answers already provided. If the user said something like "it should work like the Claude SDK adapter," that resolves the deep-agent question. Skip questions that are clearly answered. When in doubt, ask; one extra question is better than a wrong assumption.

Use AskUserQuestion to ask about:
- Should this be deep agent (SDK handles tool loop) or scaffolded (karenina orchestrates)?
- Should built-in SDK tools be enabled or disabled during benchmarking?
- What interface name to use in the registry?
- Availability check strategy (import check, CLI check, etc.)
- Fallback behavior when SDK is unavailable
- **Distribution model**: How should this adapter be registered?
  - **Built-in** (shipped with karenina): adapter lives in `karenina/src/karenina/adapters/<name>/`, one import added to `_load_builtins()` in `registry.py`
  - **Plugin** (entry point): adapter lives in `karenina/src/karenina/adapters/<name>/` or a separate package, registered via `[project.entry-points."karenina.adapters"]` in `pyproject.toml`, auto-discovered at runtime
  - **Manual** (user calls `AdapterRegistry.register()`): simplest, no packaging changes, but caller must register before creating `ModelConfig`

### 5. Produce context document

Save a structured context document to `docs/adapters/<name>-context.md` with:

```markdown
# <SDK Name> Adapter Context

## SDK Overview
- Package: <pypi-name>
- Docs: <url>
- Version: >=<minimum>

## Capabilities
| Capability | Supported | Notes |
|...

## Key API Surface
- Agent creation: `create_agent(...)`
- Invocation: `agent.invoke(...)` / `agent.ainvoke(...)`
- Message types: ...
- Trace access: ...
- Usage access: ...

## Concept Mapping (preliminary)
| karenina concept | SDK equivalent |
|...

## Critical Integration Points
- Filesystem access: Real disk / virtual state / configurable
- Usage tracking: Per-turn metadata location / aggregation needed
- Structured output: Preserves raw response for usage? (include_raw?)
- Structured output return type: Pydantic model / dict / JSON string
- Recursion limit: SDK parameter name and mapping ratio (e.g., 2x for LangGraph)
- Recursion limit: Behavior on limit hit (exception / partial state / empty)
- No-tools behavior: Falls back to single-turn / raises error / other
- Streaming: Async iterator API / callback-based / none; where usage lands (final chunk vs separate call)
- Agent timeout: Can partial trace be recovered from the SDK state after `asyncio.wait_for` cancels the run?

## Design Questions (resolved)
- Natively agentic: Yes/No
- Built-in tools: Enabled/Disabled
- Interface name: `<name>`
- Availability: Import check / CLI check
- Fallback: None / langchain
- Filesystem backend: Real / virtual / needs configuration
- Distribution: built-in / plugin / manual
```

## Next Step

Run `/karenina-adapter-design` to begin Phase 2 using this context document.
