---
name: karenina-adapter-review
description: Review a new karenina adapter for quality, correctness, and convention compliance. Final phase before merge. Use as Phase 5 of adapter creation.
---

# Phase 5: Review Karenina Adapter

Comprehensive review of the adapter implementation for quality, correctness, and convention compliance.

## Prerequisites

- All tests from `/karenina-adapter-test` pass
- Adapter is registered and functional

## Review checklist

### 1. Code quality review

Dispatch the code-reviewer agent against the adapter directory:

```
Review the adapter at karenina/src/karenina/adapters/<name>/
against the spec at docs/superpowers/specs/<date>-<name>-design.md
```

### 2. File size check

All files must be under 800 lines (per CLAUDE.md):

```bash
find karenina/src/karenina/adapters/<name>/ -name "*.py" -exec wc -l {} +
```

### 3. No cross-adapter imports

Verify the adapter does not import from other adapter packages:

```bash
# Check for imports from ANY other adapter package (not just specific ones)
grep -rn "from karenina.adapters\." karenina/src/karenina/adapters/<name>/ \
    | grep -v "from karenina.adapters.<name>" \
    | grep -v "from karenina.adapters.registry"
```

Expected: No matches. The adapter should be self-contained. Imports from `karenina.adapters.registry` are allowed (needed for registration). Imports from other adapter packages (e.g., `from karenina.adapters.langchain.initialization`) are not; shared utilities should be copied, not imported across adapters.

### 4. Lazy import pattern

Verify `__init__.py` uses the `__getattr__` lazy import pattern, not eager imports:

```bash
head -80 karenina/src/karenina/adapters/<name>/__init__.py
```

Check: No top-level imports of adapter classes. All imports inside `__getattr__`.

### 5. `__all__` exports match implementations

Every name in `__all__` should resolve via `__getattr__`:

```python
from karenina.adapters.<name> import *  # Should not raise
```

### 6. Registration verification

```python
from karenina.adapters.registry import AdapterRegistry
spec = AdapterRegistry.get_spec("<interface>")
assert spec is not None
assert spec.agent_factory is not None
assert spec.llm_factory is not None
assert spec.parser_factory is not None
assert spec.agent_tier == "deep_agent"  # or "tool_loop"
assert spec.supports_mcp == <expected>
```

### 7. Plugin entry point verification

For plugin packages (adapters distributed as separate Python packages), verify that `pyproject.toml` declares the entry point correctly under `[project.entry-points."karenina.adapters"]`. The key should match the interface name and the value should point to the registration module.

### 8. Convention compliance

Check against CLAUDE.md:
- [ ] `from __future__ import annotations` at top of every file
- [ ] `logger = logging.getLogger(__name__)` after all imports
- [ ] Google-style docstrings on all public methods
- [ ] Lazy `%`-style log formatting (no f-strings in logger calls)
- [ ] Duck typing (no explicit Protocol inheritance)
- [ ] Exception chaining (`raise ... from e`)
- [ ] PEP 604 unions (`X | None`, not `Optional[X]`)
- [ ] Module-level docstrings on all `.py` files
- [ ] `aclose()` implemented on all three port adapters (required protocol method)
- [ ] `capabilities` property on all three port adapters (returns `PortCapabilities`)
- [ ] `LLMPort.astream()` and `LLMPort.stream_invoke()` defined (implemented, or raising `NotImplementedError`); `supports_streaming` capability flag matches reality
- [ ] `max_retries` warning emitted via `logger.warning()` if unsupported and caller passes it

### 9. Hot-test-proven correctness checks

These items correspond to bugs found in 3 or 4 out of 4 existing adapters during hot testing. Every one must be verified:

- [ ] **Structured output produces valid JSON**: Search for `str(response)` or `str(result)` in the `with_structured_output()` code path. If the framework returns a Pydantic model, the adapter must use `model_dump_json()`. If it returns a dict, it must use `json.dumps()`. Never `str()`.
- [ ] **`max_turns` is wired to the framework's limit mechanism**: Trace the code path from `AgentConfig.max_turns` to the SDK call. Verify the value actually reaches the SDK parameter (e.g., LangGraph `recursion_limit`, Anthropic tool_runner loop counter). An adapter that accepts `max_turns` but never passes it to the SDK will run indefinitely.
- [ ] **No-tools fallback exists and honors guards**: Call the agent adapter with `tools=None` and `mcp_servers=None`. It must not crash. The fallback path must still honor `config.timeout` (wrapped in `asyncio.wait_for()`).
- [ ] **Timeout applies to ALL execution paths**: If the adapter has multiple paths (tool loop vs single-turn fallback), verify both wrap their API call in `asyncio.wait_for()` when `config.timeout` is set.
- [ ] **MCP sessions outlive their tools**: If the adapter uses MCP, verify that sessions are managed with `AsyncExitStack` and that tools are not used after the session context closes. Check for `async with AsyncExitStack()` patterns where tools escape the block.
- [ ] **Deferred error pattern for MCP cleanup**: If the adapter uses `AsyncExitStack` for MCP, verify that exceptions during agent execution are captured and re-raised after the stack closes cleanly, preventing `ExceptionGroup` wrapping.
- [ ] **Streaming timeout capture (LLMPort)**: If `supports_streaming=True`, verify that `stream_invoke()` returns an `LLMResponse` with `is_partial=True` and `usage_unavailable=True` when the wall-clock timeout fires. Accumulated content must be preserved (not discarded), and the async context manager used by `astream()` must yield a `StreamingLLMResponse` that exposes `accumulated_content` even on partial reads. If streaming is not supported, both `astream()` and `stream_invoke()` must exist and raise `NotImplementedError`.
- [ ] **Agent timeout partial recovery (AgentPort)**: When `config.timeout` fires mid-run and the adapter has collected at least one message, verify that `arun()` returns a partial `AgentResult(timeout_reached=True)` instead of raising `AgentTimeoutError`. The partial result must include the messages collected so far, aggregated usage for the completed turns, and a marker in `raw_trace` (e.g., `"[Note: Agent timed out - partial response shown]"`). Raising `AgentTimeoutError` is only acceptable when no messages were collected, or when the SDK exposes no way to read partial state (document this limitation in the adapter's module docstring).

### 10. Full regression test

```bash
cd karenina && uv run pytest tests/ -x -q
```

All existing tests must pass with zero new failures.

### 11. Hot test suite verification

Run the minimum viable hot test set from `/karenina-adapter-test` (H1 + H2 + H5 + H6 + H8 + H9). This is the final gate: code review and cold tests cannot catch bugs that only manifest with live API calls.

Specifically verify:
- H1: Simple query produces correct answer
- H2: Agent can read a real file from a workspace directory (catches virtual filesystem bugs)
- H5: LLMPort returns valid response with non-zero usage tokens
- H6: ParserPort extracts structured data with non-zero usage tokens
- H8: Usage tracking is consistent across all three ports (total = input + output, model name present)
- H9: Short `config.timeout` either returns a partial `AgentResult(timeout_reached=True)` or raises `AgentTimeoutError` (never hangs, never swallows the error)

If any hot test fails, the adapter has a real bug that cold tests missed. Fix before merging.

## Revision loop

If any check fails:
1. Fix the issue
2. Re-run the specific check
3. Repeat until all checks pass (max 3 iterations)

If issues persist after 3 iterations, escalate to the user.

## Completion

Once all checks pass, the adapter is ready for merge. Consider:
- Updating the adapter docs (`karenina/docs/advanced-adapters/`) if behavior patterns changed
- Adding the adapter to CLAUDE.md's adapter table
- Updating the hub skill (`/karenina-adapter-create`) if the pattern has evolved
