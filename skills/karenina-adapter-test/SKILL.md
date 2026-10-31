---
name: karenina-adapter-test
description: Test a Karenina adapter according to its implemented ports and declared capabilities, including protocol conformance, cold tests, live workspace/tool calls, usage, cancellation, and requested model routes.
---

# Phase 4: Test the Adapter

Use the design's port/capability matrix as the test contract. Do not require LLMPort or ParserPort tests from an AgentPort-only adapter, and do not skip an implemented port.

Read [Protocol bridging](../karenina-adapter-create/references/protocol-bridge.md) for protocol-specific cases.

## Cold tests

### Registration and availability

- Interface is registered under the documented name.
- Exactly the implemented factories are non-`None`.
- Agent tier, MCP/tools flags, provider requirement, fallback, and runtime profile match behavior.
- Missing optional package/executable produces an informative unavailable result while the interface remains discoverable.
- Factory behavior for unsupported ports is documented and tested.

### Port conformance

For each implemented port:

- Runtime-checkable Protocol and method signatures
- `capabilities` values
- `aclose()` idempotence
- Required result type and non-negative usage
- Sync/async wrapper behavior
- Error hierarchy and exception chaining

### Protocol bridge

When applicable, test:

- successful and mismatched version negotiation
- optional/omitted capabilities
- input role/content serialization
- text and thought chunk coalescing by message id
- tool start/progress/completion and failed result pairing
- event ordering and duplicate update suppression
- usage normalization, including cumulative versus per-call and cache buckets
- stop-reason/limit mapping
- cancellation, partial trace, late events, and cleanup
- client callbacks: allowed, denied, and method-not-supported
- MCP transport conversion
- bounded malformed frames and stderr

Use protocol schema objects or transcript fixtures rather than vendor network calls for these unit tests.

### AgentPort behavior

- knowledge/no-tools path
- real-workspace path mapping
- system prompt and prior history
- built-in/static tools and MCP according to support
- timeout on every path
- partial timeout result when events exist
- turn-limit enforcement or documented best-effort behavior
- session/process cleanup on success, timeout, and failure
- trace ids and final-response extraction

### LLMPort behavior

- ordinary invocation and usage
- structured output is valid JSON
- retry/request-timeout mapping
- streaming chunks and usage when advertised
- interrupted streaming flags

### ParserPort behavior

- returns the requested Pydantic model
- usage survives structured extraction
- invalid output and retry behavior

## Live-test rules

Live calls spend quota and may expose data externally. Use them when requested or approved. Load credentials through existing environment/auth stores without printing, copying, or committing secrets.

Use only synthetic temporary workspaces and markers. Never send repository files unless the user explicitly authorizes their contents.

For each requested provider/subscription/model route, record:

- requested and actual model when observable
- exact assertion/marker
- tool/MCP trace present when expected
- input/output/cache usage
- turns, limit, and timeout flags
- artifact state on disk

### Minimum live matrix by capability

AgentPort:

1. Simple deterministic response
2. Read a synthetic file and return an exact marker
3. Write a synthetic file when read-write is advertised
4. Verify paired tool call/result in structured and raw traces
5. Verify non-zero usage and `total = input + output`
6. Short timeout: partial result or documented timeout error, never hang
7. Turn limit if enforceable; otherwise demonstrate and document the gap
8. MCP call if `supports_mcp=True` and a safe test server is available

LLMPort:

1. Deterministic single-turn response
2. Non-zero usage
3. Structured JSON if supported
4. Streaming and interrupted partial capture if supported

ParserPort:

1. Typed extraction from deterministic text
2. Non-zero usage
3. Invalid/ambiguous input behavior

## Commands

Use repository-native commands and extras. Typical sequence:

```bash
uv run --extra <adapter> pytest tests/unit/adapters/<name> -q
uv run --extra <adapter> pytest tests/unit/adapters/conformance -q
uv run --extra <adapter> ruff check src/karenina/adapters/<name> tests/unit/adapters/<name>
uv run --extra <adapter> mypy src/karenina/adapters/<name>
uv run --extra <adapter> pytest tests -q
```

Use the lockfile (`--locked`) after dependency resolution.

## Pass criteria

- All selected-port cold tests pass.
- Every advertised capability has positive evidence.
- Every unsupported capability is rejected or documented, never silently ignored.
- Requested live routes pass exact behavioral assertions.
- Full regression has no new failure.

Continue with `$karenina-adapter-review`.
