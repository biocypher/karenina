# Protocol Bridging to Karenina

Use this reference whenever a coding agent, CLI, daemon, or model service implements a published protocol. The adapter is a semantic bridge, not merely a subprocess or SDK wrapper.

## 1. Discover protocols before APIs

Inspect primary documentation and the executable/source for:

- Agent Client Protocol (ACP)
- Model Context Protocol (MCP)
- OpenAI Responses or Chat Completions-compatible APIs
- JSON-RPC/LSP-style agent protocols
- A2A or another versioned agent protocol
- A vendor-native event/RPC protocol

Separate protocols by role. MCP supplies tools/context; it does not itself replace an agent-turn protocol. A backend may implement more than one protocol.

## 2. Compare transports by semantic coverage

Build a decision table before implementation:

| Requirement | Standard protocol | Native API/RPC | Karenina need |
| --- | --- | --- | --- |
| Version negotiation |  |  | Fail clearly on incompatible wire versions |
| Capability negotiation |  |  | Set registration/runtime capabilities truthfully |
| Session lifecycle |  |  | `session_id`, cleanup, no cross-run leakage |
| System/user/history input |  |  | Preserve roles and current question |
| Text/thought streaming |  |  | Coalesced `Message` content and timeout salvage |
| Tool input/result ids |  |  | Paired `ToolUseContent` / `ToolResultContent` |
| Workspace roots |  |  | `AgentConfig.workspace_path` |
| MCP servers |  |  | `MCPServerConfig` transport mapping |
| Usage/cache/cost |  |  | `UsageMetadata` with Karenina semantics |
| Stop reasons/limits |  |  | `limit_reached`, `turns` |
| Cancellation |  |  | timeout cancellation and partial result |
| Permission callbacks |  |  | access-mode enforcement without false sandbox claims |
| Actual routed model |  |  | `actual_model`, or a documented best effort |

Prefer the standard protocol when it covers the required semantics and has a compatible, maintained client or a small auditable implementation. Use a native surface when the standard loses a critical invariant. A hybrid is valid: keep the standard lifecycle and read missing vendor fields from namespaced extension metadata. Document the decision and its compatibility cost.

## 3. Write the mandatory bridge table

Map wire concepts to Karenina concepts explicitly. At minimum cover:

| Wire concept | Karenina target | Questions |
| --- | --- | --- |
| Initialize/version/capabilities | Availability, registration, `PortCapabilities` | What is negotiated? What is optional? |
| New/load session | One `arun()` lifetime and `AgentResult.session_id` | Is state persisted or isolated? |
| Prompt/content blocks | `Message`, `Role`, content dataclasses | How are system instructions and history represented? |
| Text/thought updates | `TextContent`, `ThinkingContent` | How are chunks grouped and ordered? |
| Tool start/update/end | `ToolUseContent`, `ToolResultContent` | Which id pairs them? Is the raw tool name present? |
| Usage | `UsageMetadata` | Are counts per request or cumulative? Does total include cache/reasoning? |
| Stop/cancel/error | flags and port exceptions | Which outcomes are partial success versus failure? |
| Workspace/filesystem/terminal | `workspace_path`, runtime capabilities | Is containment real, advisory, or absent? |
| MCP configuration | `MCPServerConfig` | Which transports and fields survive? |
| Extensions | adapter-specific options | Are they versioned and namespaced? |

Never map fields by spelling alone. For example, Karenina defines `total_tokens` as input plus output and stores cache reads/creation separately; a wire protocol may include cache or thought tokens in its total.

## 4. Handle the client side of bidirectional protocols

Agent protocols may call back into the client for permissions, filesystem access, terminals, elicitation, or hosted tools. For every callback:

- Implement it, advertise it, or return the protocol's method-not-supported error.
- Do not advertise a capability with a stub that cannot honor it.
- Translate Karenina access mode into permission responses.
- Distinguish approval policy from OS sandboxing.
- Bound outputs and errors so traces do not leak secrets or grow without limit.

## 5. Preserve ordering and partial state

- Coalesce stream chunks using protocol message ids.
- Preserve event order across assistant, tool-use, tool-result, and final text.
- Retain completed events before a timeout.
- Send the protocol cancellation message before terminating the process/transport.
- Drain for a bounded grace period, then close all sessions and child processes.

## 6. Test both layers

Protocol tests:

- Supported and mismatched version negotiation
- Capability combinations and omitted optional fields
- Chunk coalescing
- Tool id correlation and failed tools
- Usage semantic normalization
- Cancellation and late updates
- Unsupported callbacks

Implementation tests:

- The target runtime speaks the claimed protocol version
- Model/provider selection reaches the requested route
- A real workspace read and, where authorized, write
- MCP transport if advertised
- Non-zero usage and correct cache buckets
- Partial timeout behavior
- Runtime/process cleanup

If a native fallback remains, run equivalent acceptance tests against both paths and state which one is authoritative.
