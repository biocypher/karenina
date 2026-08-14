---
name: karenina-adapter-review
description: Audit a completed Karenina adapter against its supported-port contract, protocol mapping, safety claims, lifecycle, tests, documentation, and repository conventions before merge.
---

# Phase 5: Review the Adapter

Review evidence, not just code shape. Use a code-review agent only when one is available and repository instructions permit it; otherwise perform the audit directly.

## 1. Contract consistency

Compare context, design, source, `AdapterSpec`, docs, and tests:

- Same supported-port matrix everywhere
- `None` factories are intentional and documented
- No claim that availability fallback fills unsupported ports unless verified in factory code
- Agent tier, tools, MCP, provider requirement, and capabilities match live behavior
- Configuration examples use real field names and valid model/provider selectors

## 2. Protocol audit

Read [Protocol bridging](../karenina-adapter-create/references/protocol-bridge.md) and verify:

- Standard protocols were discovered before choosing a native path
- Transport decision is justified by semantic coverage
- Protocol version is negotiated/validated
- Capabilities are not assumed when omitted
- Every used wire concept has a documented Karenina mapping
- Chunks are coalesced; tool ids pair; ordering is stable
- Usage formulas reconcile semantic differences rather than copying names
- Cancellation precedes termination and partial state is retained
- Bidirectional callbacks are implemented or rejected correctly
- Extensions are namespaced and tested

## 3. Safety and lifecycle

- Workspace path is absolute and reaches the real tool runtime
- `cwd`/approvals are not described as an OS sandbox
- Read-only policy blocks write/execute paths it claims to block
- MCP sessions, async tasks, clients, streams, temp dirs, and subprocesses close on every exit
- Interrupt/drain/close/kill waits are bounded
- Stderr and tool outputs are bounded in diagnostics/traces
- Secrets are inherited or injected safely and never logged
- Sessions do not leak state between benchmark runs unless resumption is designed

## 4. Port correctness

For each implemented port, verify the current Protocol exactly.

AgentPort:

- no-tools behavior, workspace, MCP/static-tool handling
- timeout and partial-result behavior on every path
- real max-turn enforcement or an explicit, tested limitation
- final response, turns, limits, model, usage, and dual trace agree

LLMPort:

- structured output is JSON, not Python repr
- streaming methods/capability agree
- interrupted streaming preserves content and marks usage unavailable
- request timeout and retry policy reach the backend

ParserPort:

- requested Pydantic type and raw usage survive extraction
- retries and parse errors use Karenina conventions

All implemented ports expose accurate capabilities and idempotent `aclose()`.

## 5. Packaging and maintainability

- Optional imports do not prevent registry discovery
- Availability explains the missing executable/package and install extra
- No cross-adapter implementation imports
- Shared logic lives in genuine core utilities or remains self-contained
- Public methods have useful docstrings and types
- Logging is lazy and errors use exception chaining
- Files remain focused; split by responsibility when size harms reviewability
- Built-in import, plugin entry point, or manual instructions match the design
- Lockfile and optional dependency metadata are consistent

## 6. Test evidence

Require:

- Focused adapter tests
- Applicable conformance tests
- Protocol transcript/schema tests
- Format, lint, and type checks
- Full regression
- Capability-scoped live tests for every requested route

Inspect live assertions: a successful HTTP/model response is insufficient. Workspace/tool tests need exact file markers and trace evidence; usage needs non-zero input/output and correct cache separation; timeout tests must prove cancellation/cleanup.

## 7. Repository hygiene

- Review the diff and status for unintended files
- Preserve unrelated user changes
- Ensure docs/specs/tests are included in the intended PR
- Keep separate features in separate branches/PRs
- Rebase or merge according to the user's requested history policy, then rerun risk-proportionate tests

Fix every correctness issue found and rerun its focused test. Report any remaining limitation explicitly before proposing merge.
