# Oh My Pi Adapter Context

## Runtime overview

- Runtime: Oh My Pi (`omp`), a multi-provider coding agent CLI
- Primary documentation: <https://omp.sh/>
- Source: <https://github.com/can1357/oh-my-pi>
- Standard protocol: Agent Client Protocol (ACP) v1 over JSON-RPC 2.0/stdin/stdout
- Python client: `agent-client-protocol>=0.12.0,<0.13`
- Distribution: built in to Karenina, installed with the `omp` extra; the `omp` executable is installed separately

## Capabilities

| Capability | Supported | Notes |
| --- | --- | --- |
| Agent loop | Yes | OMP owns the model/tool loop and is a deep agent. |
| Async | Yes | Karenina uses the async ACP Python client over an OMP subprocess. |
| Standard protocol | Yes | ACP v1 provides initialization, capabilities, sessions, prompt turns, updates, cancellation, MCP, and usage. |
| MCP | Yes | ACP `session/new.mcpServers` supports stdio, HTTP, and SSE servers. |
| Built-in tools | Yes | OMP exposes file, search, shell, and other native tools. |
| Karenina `Tool` definitions | No | ACP v1 does not define arbitrary client-supplied callable tools. |
| Traces | Yes | `agent_message_chunk`, `agent_thought_chunk`, `tool_call`, and `tool_call_update` map to Karenina messages. |
| Usage | Yes | OMP returns per-prompt ACP `Usage` with input, output, total, cache-read, and cache-write tokens. |
| Cancellation | Yes | Karenina timeouts send ACP `session/cancel`, salvage collected updates, then stop the subprocess. |
| System prompt | Emulated | ACP has no system-message parameter; Karenina serializes system instructions and history into the prompt. |
| Turn limit | Partial | ACP reports `max_turn_requests`, but ACP v1 has no portable client-set maximum. |
| Workspace | Yes | ACP `session/new.cwd` receives an absolute path. OMP tools run against that workspace. |
| OS sandbox | No | Native OMP is a host process. Read-only mode uses OMP permission gates, not an OS sandbox. |

## Protocol-first bridge

| ACP concept | Karenina concept |
| --- | --- |
| `initialize` + negotiated `protocolVersion` | Availability/conformance gate before any run |
| `agentCapabilities` | Registered adapter capabilities (`supports_mcp`, deep-agent tier) |
| `session/new.cwd` | `AgentConfig.workspace_path` |
| `session/new.mcpServers` | `dict[str, MCPServerConfig]` |
| `session/set_config_option(model)` | `ModelConfig.model_provider/model_name` as `provider/model` |
| `session/set_config_option(thinking)` | `ModelConfig.extra_kwargs["thinking"]` |
| `session/prompt` | `AgentPort.arun(messages, ...)` |
| `agent_message_chunk` | `Message(role=ASSISTANT, TextContent)` |
| `agent_thought_chunk` | `ThinkingContent` |
| `tool_call.rawInput` | `ToolUseContent`; the standard tool kind is the portable name |
| completed/failed `tool_call_update.rawOutput` | `ToolResultContent` |
| `PromptResponse.usage` | `UsageMetadata`, including cache token fields |
| `PromptResponse.stopReason` | `limit_reached` and normal/cancelled completion |
| `session/cancel` | timeout cancellation and partial-trace recovery |
| ACP session id | `AgentResult.session_id` |

## Transport decision

OMP also exposes native JSON and RPC modes. The adapter uses ACP because OMP's implementation preserves every Karenina-critical semantic: tool inputs/results, streamed text/thought, MCP, cancellation, session identity, model/thinking selection, and per-prompt usage. Native RPC would create an OMP-specific bridge without adding required fidelity.

## Resolved design choices

- Interface: `omp`
- Ports: AgentPort only; verification parsing/judging must use a separate LLM/Parser model configuration
- Agent tier: `deep_agent`
- Provider required: yes
- Availability: both `omp` executable and Python `acp` package
- Fallback when unavailable: `langchain`
- Reproducibility: run with skills, rules, and extensions disabled by default; adapter-specific options may opt them back in
- Permission mode: `yolo` for read-write, `always-ask` with a deny-by-client policy for write/execute calls in read-only mode
