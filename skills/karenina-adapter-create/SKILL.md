---
name: karenina-adapter-create
description: Guide the full lifecycle for adding or revising a Karenina model or coding-agent adapter, including protocol discovery, port selection, implementation, cold tests, live tests, and review.
---

# Create a Karenina Adapter

Use a five-phase workflow. Do not assume every backend implements every Karenina port.

| Phase | Skill | Deliverable |
| --- | --- | --- |
| 1 | `$karenina-adapter-gather-context` | `docs/adapters/<name>-context.md` |
| 2 | `$karenina-adapter-design` | `docs/adapters/<name>-adapter-design.md` |
| 3 | `$karenina-adapter-implement` | Source, registration, and dependency changes |
| 4 | `$karenina-adapter-test` | Capability-scoped cold and live evidence |
| 5 | `$karenina-adapter-review` | Final correctness and maintainability audit |

Read [Protocol bridging](references/protocol-bridge.md) before Phase 1 when the target is a coding agent, CLI, daemon, or service. If it implements a standard protocol, design the bridge from that protocol to Karenina explicitly. Do not jump straight to vendor-specific SDK objects.

## Start with a port matrix

Karenina has three independent ports:

| Port | Purpose | Implement when |
| --- | --- | --- |
| `AgentPort` | Agent loop, tools, workspace, MCP, trace | The backend is agentic or owns a tool loop |
| `LLMPort` | Single-turn generation, streaming, structured output | The backend exposes a direct model-call API with the required semantics |
| `ParserPort` | Typed Pydantic extraction | The backend can reliably return and validate schema-constrained output |

An adapter may implement any non-empty subset. Record unsupported ports in the context, design, registration tests, and user docs. `fallback_interface` handles an unavailable adapter; it does not automatically fill a `None` port factory. Callers needing unsupported duties must configure a separate model/interface unless the factory code explicitly provides port-level fallback.

## Distribution

| Model | Registration |
| --- | --- |
| Built in | Add `registration.py` and one guarded import in `AdapterRegistry._load_builtins()` |
| Plugin | Declare a `karenina.adapters` entry point in the plugin package |
| Manual | The caller imports a module that invokes `AdapterRegistry.register()` |

All models use `AdapterSpec`. Choose the distribution from the request or repository context; ask only if the choice materially changes the deliverable and cannot be inferred.

## Package shape

Create only the files the selected ports and transport require. Typical modules include:

```text
src/karenina/adapters/<name>/
  __init__.py
  availability.py
  registration.py
  agent.py          # AgentPort only
  llm.py            # LLMPort only
  parser.py         # ParserPort only
  messages.py       # when conversion is non-trivial
  usage.py          # when accounting needs a dedicated mapper
  trace.py          # when trace projection needs a dedicated mapper
  mcp.py            # when MCP is supported
  protocol.py       # standard/native wire bridge, when applicable
  errors.py         # when backend error mapping is non-trivial
```

Keep optional dependency imports lazy enough that registry discovery can report unavailability instead of losing the interface registration.

## Required outcomes

- Context and design documents state the supported-port matrix.
- Standard protocol discovery and the protocol-to-Karenina mapping are documented, or the absence of a suitable protocol is documented.
- Every implemented port satisfies its current Protocol, including `capabilities` and `aclose()`.
- Registration flags match behavior proved by tests.
- Agent timeouts cancel the backend and return partial traces when the transport exposes them.
- `UsageMetadata.total_tokens` follows Karenina's convention (`input_tokens + output_tokens`); cache buckets remain separate even if a provider/protocol defines a broader total.
- Cold tests cover every implemented port and every declared capability.
- Live tests exercise the actual provider/runtime, workspace, tools, traces, usage, timeout/limit behavior where supported, and each requested subscription/model route.
- Full regression tests, lint, format, and type checks pass in proportion to the change.

## Persistent pitfalls

- Treating all three factories as mandatory.
- Claiming fallback for an unsupported port without tracing the factory behavior.
- Wrapping a proprietary SDK before checking ACP, MCP, OpenAI-compatible APIs, JSON-RPC, or another published protocol.
- Copying a protocol field by name without reconciling semantics; token totals and turn counts commonly differ.
- Advertising sandboxing when the runtime only changes `cwd` or uses an approval prompt.
- Capturing streamed chunks as separate messages instead of coalescing them by protocol message id.
- Losing tool-call/result correlation ids.
- Accepting `max_turns`, timeouts, or MCP parameters without forwarding/enforcing them or documenting and testing a protocol limitation.
- Eager imports that make optional adapters disappear from the registry.
- Live tests that only ask a trivia question and never prove real workspace/tool behavior.

Begin with `$karenina-adapter-gather-context`.
