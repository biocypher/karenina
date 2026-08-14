# Oh My Pi ACP Adapter Design

## Overview

Add a built-in, AgentPort-only `omp` adapter that launches `omp acp` per Karenina run and bridges ACP v1 into Karenina's unified agent result. Keep OMP-specific concerns (command discovery and model/thinking configuration) outside the protocol mapper.

## Package structure

```text
src/karenina/adapters/omp/
  __init__.py
  acp_bridge.py      # ACP Client callbacks and event-to-Message projection
  agent.py           # AgentPort orchestration and subprocess lifetime
  availability.py    # CLI and Python-package checks
  mcp.py             # Karenina MCPServerConfig -> ACP schema
  messages.py        # prompt/history serialization and raw trace
  registration.py    # AdapterSpec and runtime profile
```

## Registration

- `interface="omp"`
- `agent_factory` only; `llm_factory` and `parser_factory` are `None`
- `fallback_interface="langchain"` applies only when OMP is unavailable; it does not synthesize unsupported LLM/Parser ports
- `supports_mcp=True`, `supports_tools=True`, `agent_tier="deep_agent"`, `requires_provider=True`
- Runtime capabilities advertise filesystem access, code execution only in read-write mode, and no OS sandbox

## Core flow

1. Validate that static Karenina tools were not supplied.
2. Resolve an absolute workspace (configured path or a temporary directory).
3. Convert MCP server dictionaries to ACP schema objects.
4. Start `omp acp` with deterministic context discovery defaults and the permission policy implied by `agent_runtime.access_mode`.
5. Initialize ACP v1 and verify the negotiated protocol version.
6. Create a session with the workspace and MCP servers.
7. Set OMP's ACP `model` option to `provider/model`; set `thinking` when configured.
8. Serialize system instructions and conversation history into one ACP text prompt.
9. Collect ACP updates into ordered Karenina assistant/tool messages while awaiting `session/prompt`.
10. On timeout, send `session/cancel`, allow a short drain, and return the partial trace with `timeout_reached=True`.
11. Map prompt usage and stop reason, close the ACP session/connection, and terminate the subprocess.

## Mapping and fidelity rules

- Text and thought chunks sharing a `messageId` are coalesced, never emitted as one Karenina message per token chunk.
- ACP has portable tool kinds but no mandatory raw tool name. Use `kind` as `ToolUseContent.name`; fall back to `other`.
- Preserve the exact `rawInput` dictionary. Serialize `rawOutput` as JSON when it is not text.
- A failed tool update becomes `ToolResultContent(is_error=True)`.
- ACP cache-read/write tokens map to Karenina cache fields. Karenina's `total_tokens` remains `input_tokens + output_tokens`; ACP's broader total (which includes cache buckets in OMP) is not copied into that narrower field.
- `max_turn_requests` and `max_tokens` set `limit_reached=True`; `cancelled` only sets `timeout_reached` when Karenina initiated cancellation for a timeout.
- `actual_model` is the requested OMP selector because ACP v1 does not standardize the routed model in the prompt response.

## Safety and workspace boundaries

OMP's native runtime is not an OS sandbox. Read-write mode is explicitly unattended and may operate on absolute paths the model supplies. Read-only mode keeps OMP's ACP permission gate active and rejects write/execute permission requests, but callers needing hard isolation must run Karenina/OMP in a container or other external sandbox. `uses_sandboxed_execution` therefore remains false.

## Error mapping

- Missing CLI/package -> adapter unavailable
- Protocol-version mismatch, malformed response, unsupported static tools -> agent response/execution error
- ACP/JSON-RPC request failure or non-zero early process exit -> agent execution error including bounded stderr
- Timeout before any useful event -> partial `AgentResult` when possible; otherwise timeout error

## Tests

- Unit-test prompt, MCP, trace, usage, permissions, registration, subprocess orchestration, protocol mismatch, cancellation, and malformed events.
- Run adapter conformance and the full unit suite.
- Live-test a real file-tool turn using `zhipu-coding-plan/glm-5.3` at high thinking.
- Live-test at least two different models through `opencode-go`.

## Success criteria

- The adapter uses ACP v1 end to end and documents every ACP-to-Karenina mapping.
- Structured trace tool-call ids pair with tool results.
- Live runs produce exact marker responses, actual file reads/writes, non-zero usage, and no leaked sessions/processes.
- The adapter-creation skills require protocol discovery, a mapping table, capability-gap decisions, and protocol conformance tests.
