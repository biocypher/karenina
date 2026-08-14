---
name: karenina-adapter-create
description: Guide for creating a new karenina adapter. Provides the full lifecycle from requirements gathering to working, tested code. Use when adding support for a new LLM SDK or agentic framework.
disable-model-invocation: true
---

# Create Karenina Adapter

This skill guides you through creating a new adapter for karenina's verification pipeline. Each adapter implements three port interfaces (AgentPort, LLMPort, ParserPort) and integrates with the AdapterRegistry.

## Lifecycle Overview

The adapter creation process has five phases. Invoke each phase skill in order:

| Phase | Skill | Produces |
|-------|-------|----------|
| 1. Gather Context | `/karenina-adapter-gather-context` | Context document with SDK capabilities |
| 2. Design | `/karenina-adapter-design` | Design spec with concept mapping |
| 3. Implement | `/karenina-adapter-implement` | Adapter package (~13 source files) |
| 4. Test | `/karenina-adapter-test` | Conformance + adapter-specific tests |
| 5. Review | `/karenina-adapter-review` | Clean, convention-compliant code |

## Distribution Models

There are three ways to distribute an adapter, depending on where the code lives:

| Model | Where code lives | How it registers | When to use |
|-------|-----------------|-----------------|-------------|
| **Manual** | Any Python module | Call `AdapterRegistry.register()` directly in your code before creating a ModelConfig | One-off experiments, internal prototypes |
| **Built-in** | `karenina/src/karenina/adapters/<name>/` | Write `registration.py`; add import in `registry.py _load_builtins()` | Adapters shipped with karenina |
| **Installable plugin** | Separate Python package | Write `registration.py`; declare entry point under `[project.entry-points."karenina.adapters"]` in `pyproject.toml` | Third-party or optional adapters |

All three paths call the same `AdapterRegistry.register()` API. The only difference is how the registration module gets imported: manually, via `_load_builtins()`, or via entry point discovery.

## Canonical Adapter Package

Every adapter lives at `karenina/src/karenina/adapters/<name>/` and contains:

```
<name>/
├── __init__.py          # Lazy exports, __all__
├── availability.py      # SDK availability check
├── registration.py      # AdapterSpec + AdapterRegistry.register()
├── agent.py             # AgentPort implementation (core)
├── llm.py               # LLMPort implementation
├── parser.py            # ParserPort implementation
├── messages.py          # Bidirectional message conversion
├── trace.py             # Dual trace extraction (raw + structured)
├── usage.py             # UsageMetadata extraction
├── errors.py            # Exception mapping to karenina hierarchy
├── mcp.py               # MCPServerConfig conversion (if applicable)
├── initialization.py    # Model creation (if needed)
└── prompts/             # Adapter instruction registration
    ├── __init__.py
    ├── parsing.py
    ├── rubric.py
    └── deep_judgment.py
```

Not all files are required. `initialization.py` and `mcp.py` are only needed when the adapter uses LangChain model creation or MCP tools respectively.

## Deliverables Checklist

- [ ] Context document (`docs/adapters/<name>-context.md`)
- [ ] Design spec (`docs/superpowers/specs/<date>-<name>-design.md`)
- [ ] Adapter package (all source files)
- [ ] `AdapterRegistry.register()` called in `registration.py`
- [ ] `aclose()` implemented on all three port adapters (required protocol method)
- [ ] `capabilities` property implemented on all three port adapters (returns `PortCapabilities`, including the `supports_streaming` flag)
- [ ] `LLMPort.astream()` and `LLMPort.stream_invoke()` defined (real implementation, or raising `NotImplementedError` when the SDK has no streaming API)
- [ ] `AgentResult.timeout_reached` handling: either returns a partial result on timeout, or raises `AgentTimeoutError` with the limitation documented in the adapter's module docstring
- [ ] (Built-in only) Import added in `registry.py _load_builtins()`
- [ ] (Plugin only) Entry point declared in `pyproject.toml`
- [ ] Conformance tests pass
- [ ] Adapter-specific tests pass (including `test_streaming.py` if applicable and `test_agent_timeout.py` for partial-recovery adapters)
- [ ] Full test suite passes (no regressions)
- [ ] All files under 800 lines
- [ ] No cross-adapter imports

## Common Pitfalls (from hot testing all 4 adapters)

These bugs were found in multiple adapters during live API testing. They are systematic, not edge cases. The detailed patterns and code examples live in `/karenina-adapter-implement` (Phase 3); the review checklist in `/karenina-adapter-review` (Phase 5) verifies each one.

1. **Structured output must serialize to JSON.** Using `str()` on a Pydantic model produces Python repr, not JSON. Found in 3 of 4 adapters.
2. **Turn limits must be wired to the framework.** Accepting `max_turns` without passing it to the SDK's limit parameter means the agent runs indefinitely.
3. **No-tools fallback is required.** Adapters crash when called with `tools=None` and `mcp_servers=None` unless a fallback path exists.
4. **Timeout must apply to all execution paths.** If the adapter has a no-tools fallback, that path must also honor `config.timeout`.
5. **MCP sessions must outlive their tools.** Creating sessions in `async with` and returning tools after the block closes produces invalid tool handles.
6. **Deferred error pattern prevents ExceptionGroup wrapping.** Capture exceptions before `AsyncExitStack` cleanup, re-raise after.
7. **Streaming partial capture must flip both flags.** On a streaming wall-clock timeout, the returned `LLMResponse` must set both `is_partial=True` and `usage_unavailable=True`. Forgetting `usage_unavailable` lets downstream cost reporting treat zero-token partial responses as real (free) calls.
8. **Agent timeout should return a partial result when possible.** When `config.timeout` fires mid-run and at least one message was collected, return `AgentResult(timeout_reached=True)` instead of raising `AgentTimeoutError`. The pipeline propagates `timeout_reached` into `VerificationResult.response_timeout_partial` so downstream stages can score partial traces.

## Existing Adapters (for reference)

| Adapter | Interface | Deep Agent | Pattern |
|---------|-----------|-----------------|---------|
| LangChain | `langchain` | No (scaffolded) | Orchestrated tool loop |
| Claude Agent SDK | `claude_agent_sdk` | Yes | CLI-based agent |
| Claude Tool | `claude_tool` | No | Native structured output |
| Manual | `manual` | No | Pre-recorded traces |
| Deep Agents | `langchain_deep_agents` | Yes | LangGraph agent harness |

## Getting Started

Run `/karenina-adapter-gather-context <sdk-name>` to begin Phase 1.
