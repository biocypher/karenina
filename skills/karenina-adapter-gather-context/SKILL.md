---
name: karenina-adapter-gather-context
description: Research a prospective Karenina adapter, select its supported ports and transport, and produce an evidence-backed context document with protocol-to-Karenina mappings.
---

# Phase 1: Gather Adapter Context

Produce `docs/adapters/<name>-context.md`. Prefer primary documentation, source, executable help/version output, and official protocol specifications.

## 1. Resolve known inputs

Extract from the request and repository before asking questions:

- Runtime/SDK/CLI and official URL
- Desired interface name and distribution model
- Requested providers, models, subscriptions, endpoints, and live tests
- Whether built-in tools, MCP, workspace writes, or sandboxing are expected
- Existing PR/branch or closest reference adapter

Ask only for a choice that cannot be discovered and would materially change the implementation. Never ask again for facts already provided.

## 2. Select the port matrix

Read the current definitions in `src/karenina/ports/`. For each port, record supported, unsupported, or uncertain with evidence:

- `AgentPort`: agent loop, workspace, tools/MCP, trace, cancellation
- `LLMPort`: direct single-turn calls, structured output, streaming
- `ParserPort`: typed Pydantic extraction and usage

Do not require all three. Identify how callers should configure duties the adapter does not implement. Verify the actual factory behavior before describing fallback.

## 3. Discover protocols first

Read [Protocol bridging](../karenina-adapter-create/references/protocol-bridge.md). Check the target for standard agent/model/tool protocols before selecting a vendor API. If a standard exists:

1. Record protocol name, version, transport, official client libraries, and runtime negotiation.
2. Compare it with native SDK/JSON/RPC modes using the semantic-coverage table.
3. Probe the real executable or a minimal client when possible.
4. Identify extensions and missing semantics.
5. Make a preliminary standard/native/hybrid recommendation.

## 4. Gather capability evidence

Cover only relevant ports, plus these cross-cutting areas:

| Area | Evidence to capture |
| --- | --- |
| Async/process model | Native async, thread wrapper, subprocess lifetime, concurrency limits |
| Messages | Roles, content types, history, system instructions, chunk/message ids |
| Tools | Built-in versus client-supplied, tool input/result ids, parallel calls |
| MCP | Supported transports and exact server schema |
| Workspace | Real disk, virtual FS, `cwd`, absolute paths, multiple roots |
| Security | OS sandbox, container, approval policy, client permission callbacks |
| Limits | Turn/request/token limits and observable stop reason |
| Timeout | Cancellation API, partial events, bounded cleanup |
| Usage | Per-call versus cumulative; input/output/cache/thought/cost semantics |
| Model routing | Provider/model selector and actual routed model reporting |
| Errors | Protocol errors, process exit, provider errors, retry behavior |
| Optional dependency | Install name/version and behavior when absent |

## 5. Inspect Karenina integration points

Read:

- `src/karenina/adapters/registry.py`
- `src/karenina/adapters/agent_runtime.py` for agent adapters
- Current factory behavior for missing factories and unavailable adapters
- The closest built-in adapter and its tests
- Adapter documentation under `skills/using-karenina/references/advanced-adapters/`

Preserve local repository instructions and unrelated worktree changes.

## 6. Write the context document

Include:

1. Runtime overview and authoritative links
2. Supported-port matrix
3. Standard protocol discovery and transport decision matrix
4. Capabilities with evidence
5. Preliminary protocol/API-to-Karenina bridge table
6. Dependency and availability strategy
7. Workspace/security boundary
8. Live-test matrix, including exact requested routes
9. Resolved decisions and genuine open questions

For usage, state formulas, not just field names. For safety, state whether isolation is enforced by the OS, a container, a permission gate, or only a prompt.

Continue with `$karenina-adapter-design`.
