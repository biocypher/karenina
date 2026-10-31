---
name: karenina-adapter-implement
description: Implement a designed Karenina adapter with capability-scoped ports, protocol-aware mapping, safe lifecycle management, registration, and focused tests.
---

# Phase 3: Implement the Adapter

Implement the approved design. Do not expand the port matrix or transport scope without updating the design.

## Before editing

1. Use `$using-karenina` and read its advanced-adapter references.
2. Read repository instructions (`AGENTS.md`, `CLAUDE.md`, or equivalents) when present.
3. Read the design and [Protocol bridging](../karenina-adapter-create/references/protocol-bridge.md).
4. Read only the current port files being implemented, plus messages, usage, errors, capabilities, registry, factory, and agent runtime as relevant.
5. Inspect the closest adapter and current conformance tests. Reuse shared core utilities; do not import implementation helpers from another adapter package.
6. Check the worktree for user changes and preserve them.

## Implementation order

Adapt this order to the selected ports and transport:

1. Optional dependency/executable availability
2. Protocol/client transport and lifecycle
3. Message/content conversion
4. Usage and error mapping
5. MCP/tool conversion if supported
6. AgentPort, LLMPort, and/or ParserPort implementations
7. Runtime profile and `AdapterSpec`
8. Built-in import, plugin entry point, or manual registration
9. Adapter documentation and configuration examples
10. Focused tests

Create only justified modules. Keep package initialization import-safe when optional dependencies are absent.

## Implement the protocol bridge explicitly

For a standard protocol:

- Negotiate and validate its wire version.
- Derive behavior from advertised capabilities; omitted optional capabilities are unsupported.
- Coalesce streamed text/thought chunks by message id.
- Preserve tool correlation ids from start through result.
- Normalize raw tool inputs to dictionaries and serialize non-text outputs deterministically.
- Implement or reject every client callback the agent can invoke.
- Map cancellation before killing the transport.
- Keep extension metadata namespaced and isolated.
- Unit-test the mapper independently from the vendor runtime.

If the protocol has an official maintained Python SDK compatible with the repository, prefer it. If not, keep a minimal wire implementation small, versioned, and covered by transcript fixtures.

## Karenina semantic invariants

- `ToolUseContent.id` pairs exactly with `ToolResultContent.tool_use_id`.
- `AgentResult.trace_messages` and `raw_trace` describe the same ordered events.
- Final response is the last visible assistant text, with an explicit placeholder for a completed empty response.
- `UsageMetadata.total_tokens = input_tokens + output_tokens`; cache-read and cache-creation remain separate.
- Cumulative provider/protocol usage is differenced before mapping a single run.
- `actual_model` comes from runtime evidence when available; otherwise document that it is requested-model best effort.
- `turns` has one defined counting rule and tests.
- `limit_reached` comes from an enforced limit or an explicit stop reason, not guesswork.
- A timeout first sends cancellation, drains for a bounded grace period, returns collected partial state when possible, then closes tasks/processes.
- `uses_sandboxed_execution=True` only for real containment, not `cwd`, prompt text, or approval policy.

## Port requirements

Every implemented port provides `capabilities` and idempotent `aclose()`.

AgentPort:

- Accept no-tools calls or deliberately use built-in tools.
- Reject unsupported static tools/MCP rather than silently ignoring them.
- Honor workspace and overall timeout on every path.
- Wire `max_turns`, implement a client-side bound, or document/test the best-effort gap.

LLMPort:

- Preserve system messages according to capabilities.
- Return valid JSON for structured Pydantic/dict results.
- Implement both streaming methods when advertising streaming; interrupted streaming sets `is_partial=True` and `usage_unavailable=True`.
- If streaming is unsupported, define both methods to raise `NotImplementedError` and advertise false.

ParserPort:

- Return the requested Pydantic type, not a dict substitute.
- Preserve usage from the raw provider response.
- Map parse failures to Karenina errors and honor retry policy.

## Lifecycle and errors

- Keep MCP sessions alive for the full use of their tools, normally with `AsyncExitStack`.
- Drain subprocess stderr concurrently and bound it in raised errors.
- Re-raise the root execution error after cleanup; do not let cleanup `ExceptionGroup`s mask it.
- Bound interrupt, drain, close, and kill waits.
- Do not persist sessions unless the design requires resumption.

## Registration

Set each factory to a callable or `None` according to the port matrix. Test availability with and without optional dependencies. Add one guarded built-in import or one plugin entry point as designed.

## Verification while implementing

After each logical group run focused format/lint/tests. Then continue with `$karenina-adapter-test`; do not declare completion from mocks alone.
