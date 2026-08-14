---
name: karenina-adapter-design
description: Turn adapter research into an implementable Karenina design covering selected ports, protocol bridging, lifecycle, safety, registration, tests, and known fidelity gaps.
---

# Phase 2: Design the Adapter

Read the Phase 1 context and [Protocol bridging](../karenina-adapter-create/references/protocol-bridge.md). Produce `docs/adapters/<name>-adapter-design.md`.

## 1. Freeze the supported-port contract

Create a table for AgentPort, LLMPort, and ParserPort with:

- implemented or unsupported
- factory name or `None`
- consumer use cases
- separate-interface requirement for unsupported duties

Do not imply that `fallback_interface` fills unsupported ports unless current factory code explicitly does so.

## 2. Freeze the transport decision

For a standard protocol, specify:

- negotiated wire version and incompatible-version behavior
- official client dependency or minimal wire implementation
- client and agent capabilities advertised
- standard methods/notifications used
- namespaced extensions used
- native API features intentionally not used
- semantic gaps and their observable consequences

If choosing native over standard, name the missing standard semantics that make native necessary. If hybrid, define which layer is authoritative for each field.

## 3. Map concepts for each implemented port

For AgentPort cover:

| Karenina | Backend/protocol mapping |
| --- | --- |
| messages and system prompt | exact serialization/content blocks |
| `max_turns` | enforced parameter, client cancellation, best effort, or unsupported |
| timeout | cancellation, drain grace, partial-result rule |
| workspace | path mapping and containment boundary |
| tools/MCP | definitions, transports, permission flow |
| trace messages | chunk grouping and event ordering |
| usage | formulas for input/output/total/cache/cost |
| turns/limits | exact counting and stop-reason mapping |
| session/model | `session_id` and `actual_model` source |
| cleanup | clients, sessions, tasks, processes |

For LLMPort cover invocation, content, usage, structured output return types, streaming/partial flags, timeout, retries, and cleanup.

For ParserPort cover schema submission, Pydantic validation, raw/usage preservation, retry routing, and cleanup.

Mark every lossy mapping. A documented unsupported behavior is preferable to a silent approximation.

## 4. Design bidirectional protocol callbacks

When the agent calls the client, define behavior for permissions, files, terminals, elicitation, hosted tools, and extensions. Advertise only callbacks that work. Map read-only/read-write policy without claiming OS sandboxing.

## 5. Define the execution sequence

Give an ordered flow from input conversion through cleanup. Include:

- optional dependency and executable checks
- session/process ownership
- model/provider selection
- MCP lifetime
- event collection before final response
- timeout cancellation before forceful termination
- deferred error handling so cleanup does not mask the root failure
- empty/malformed response behavior

## 6. Define package and registration

List only required files. Specify the complete `AdapterSpec`, including optional factories, availability, fallback, routing, runtime profile, MCP/tools flags, agent tier, and provider requirement.

Keep optional imports out of package initialization and registration paths that must work when the extra is absent.

## 7. Define tests before implementation

Create a requirement-to-test matrix:

- protocol negotiation/capabilities
- mapping of every wire event/content type used
- usage semantic normalization
- timeout/cancellation/late events
- unsupported callbacks/ports
- registration and availability
- live knowledge, workspace/tool, usage, and requested provider/model routes
- sandbox/permission claims

## Required sections

1. Overview and motivation
2. Supported-port matrix
3. Protocol/transport decision
4. Protocol-to-Karenina bridge
5. Core flows for implemented ports
6. Safety and lifecycle
7. Package/dependencies
8. Registration
9. Error mapping
10. Test plan
11. Known limitations
12. Success criteria

Continue with `$karenina-adapter-implement`.
