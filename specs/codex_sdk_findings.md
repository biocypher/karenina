# OpenAI Codex Python SDK — Step-0 validation against local vLLM

Contract for the karenina codex adapter. All facts verified in the worktree
`karenina-codex-worktree` on 2026-06-11 against the vLLM endpoint
`http://hl-codon-gpu-020:8000/v1` (model id `qwen3.5-122b-a10b`, 262k ctx,
keyless).

## Smoke result: PASS (with a vLLM-side compatibility shim)

- Both smoke turns drove the **full codex agent loop to completion**
  (`status=completed`): reasoning -> `exec_command` tool call -> shell run ->
  `function_call_output` -> final turn. `hello.txt` was created on disk with the
  correct contents (`'hello from codex\n'`, exit code 0).
- The SDK itself works end to end. The only blocker is that **stock vLLM
  Responses API is too strict for codex's prompt shape**; it needs a thin
  request-rewriting shim (see "Critical deviation" below). With the shim in
  place the run is clean and fast (~3s per turn).

## Versions

- SDK: `openai-codex == 0.1.0b2`
- Bundled runtime: `openai-codex-cli-bin == 0.132.0`; binary reports
  `codex-cli 0.132.0`.
- Bundled binary path (via `codex_cli_bin.bundled_codex_path()`):
  `<venv>/lib/python3.13/site-packages/codex_cli_bin/bin/codex`
- Install command that targets the worktree venv (NOT miniforge):
  `env -u VIRTUAL_ENV uv pip install --python .venv/bin/python openai-codex`
  (plain `uv pip install` landed in `~/miniforge3`; `--python .venv/bin/python`
  is required). Pulls pydantic 2.13.4.

## Critical deviation from the plan: wire_api MUST be "responses", not "chat"

- The planned `wire_api=chat` **does not work**. CLI 0.132.0 hard-rejects it:
  `` `wire_api = "chat"` is no longer supported. `` /
  `How to fix: set wire_api = "responses" in your provider config.`
- vLLM **does** serve `/v1/responses` (HTTP 200, real Responses payload with
  `output`, `usage`, `status`). So `wire_api="responses"` is the only viable
  setting and it reaches vLLM fine.
- BUT codex's Responses request shape trips three vLLM validation rules. Each
  was isolated by proxying codex -> vLLM and dumping bodies:
  1. **`role="developer"` rejected.** Codex sends its permissions/apps/skills
     system text as `developer`-role input messages. vLLM `/v1/responses`
     returns `400 "Unexpected message role."` for `developer` (it accepts
     `user`, `assistant`, `system`; `developer` passes the role enum but fails a
     later check). Fix in shim: fold all `developer`/`system` input messages and
     the top-level `instructions` into a single `instructions` string and drop
     those messages from `input`.
  2. **`400 "System message must be at the beginning."`** once developer→system
     produced multiple/late system messages. Folding into `instructions` (rule 1)
     resolves this too.
  3. **`reasoning` input items rejected** on follow-up turns. Codex echoes prior
     `reasoning` items back into `input`; vLLM cannot validate the `reasoning`
     variant and 400s. Fix in shim: strip `type=="reasoning"` items from `input`.
- Net: to run codex against THIS vLLM build you need an OpenAI-compatible proxy
  (or a vLLM patch) that (a) merges system/developer text into `instructions`,
  (b) strips `reasoning` input items. A reference proxy implementing exactly this
  lives at `/tmp/codex-proxy.py` (REWRITE_ROLES=1). The adapter should either
  ship such a shim, point at a patched vLLM, or require a vLLM build whose
  Responses API accepts `developer` role + `reasoning` items.

## 2a. ApprovalMode + default approval handler

`ApprovalMode` (in `_approval_mode.py`) has exactly two members:
- `ApprovalMode.deny_all` → `AskForApproval(never)`, no reviewer.
- `ApprovalMode.auto_review` → `AskForApproval(on_request)` +
  `ApprovalsReviewer.auto_review`. **This is the default** for `thread_start`
  (`approval_mode: ApprovalMode = ApprovalMode.auto_review`).

There is no separate "yolo / approve-all" enum member. Non-interactive runs are
achieved by **not supplying an approval handler**: the SDK's
`CodexClient._default_approval_handler` **auto-APPROVES** — it returns
`{"decision": "accept"}` for both `item/commandExecution/requestApproval` and
`item/fileChange/requestApproval`, and `{}` otherwise. So with no handler set,
escalation requests are auto-accepted and the run is fully non-interactive. (A
custom `approval_handler` can be passed to `CodexClient` directly, but the
public `Codex(...)` facade does not expose it — it always uses the
auto-approve default.) `auto_review` does NOT block; it routes escalations to an
auto-review subagent, but the transport-level approval callback still resolves
via the auto-approve default. In the smoke, commands ran without any prompt.

## 2b. TokenUsageBreakdown fields (generated/v2_all.py:4219)

Snake-case attribute (JSON alias):
- `cached_input_tokens` (`cachedInputTokens`)
- `input_tokens` (`inputTokens`)
- `output_tokens` (`outputTokens`)
- `reasoning_output_tokens` (`reasoningOutputTokens`)
- `total_tokens` (`totalTokens`)

`TurnResult.usage` is a `ThreadTokenUsage` (v2_all.py:6574), NOT a bare
breakdown:
- `.last: TokenUsageBreakdown` — usage of the most recent model call
- `.total: TokenUsageBreakdown` — cumulative usage for the turn/thread
- `.model_context_window: int | None` (`modelContextWindow`)

Real payload from smoke1 (`result.usage.model_dump(by_alias=True)`):
```json
{
  "last":  {"cachedInputTokens": 0, "inputTokens": 14753, "outputTokens": 49,
            "reasoningOutputTokens": 0, "totalTokens": 14802},
  "modelContextWindow": 258400,
  "total": {"cachedInputTokens": 0, "inputTokens": 29306, "outputTokens": 125,
            "reasoningOutputTokens": 0, "totalTokens": 29431}
}
```
For karenina usage mapping use `usage.total` (or `.last` for per-call). Note
vLLM/this model reports `reasoningOutputTokens: 0` even though reasoning items
are produced.

## 2c. TurnHandle mechanics — how to get the final TurnResult

`thread.turn(input, ...)` **starts the turn immediately** (it calls
`turn_start`) and returns a `TurnHandle(thread_id, id)`. The turn is already
running on the app-server; notifications are being routed to its queue.

Two ways to get the `TurnResult`:
- **`handle.run()`** — calls `handle.stream()` internally and feeds it to
  `_collect_turn_result`. Use this if you did NOT manually consume the stream.
- **Iterate `handle.stream()` yourself**, then collect. `stream()` is a generator
  that registers the turn queue and yields every `Notification` until it sees
  `turn/completed`, then stops. The notification objects are consumed **once**
  (single shared queue). So:
  - If you fully drain `.stream()` yourself, **do NOT then call `.run()`** — it
    would register a fresh stream on an already-completed/empty queue and block
    forever.
  - To both observe notifications AND get a `TurnResult`, tee the SAME stream:
    wrap `handle.stream()` in your own generator that records `ev.method` and
    re-yields, and pass that to
    `openai_codex._run._collect_turn_result(gen, turn_id=handle.id)`. This is
    what the smoke does and it works.
- `thread.run(input)` is sugar: it does `turn(...)` then `stream()` +
  `_collect_turn_result` for you. Use it for the simple blocking case.
- `handle.run()` does NOT re-run the turn (it never re-sends `turn/start`); it
  only collects the in-flight turn's notification stream.
- `TurnHandle.interrupt()` exists (calls `turn/interrupt`); `.steer(input)`
  exists (calls `turn/steer`). Confirmed `hasattr(handle, "interrupt") is True`.

`TurnResult` fields (`_run.py:21`): `id, status (TurnStatus), error (TurnError|
None), started_at, completed_at, duration_ms, final_response (str|None),
items (list[ThreadItem]), usage (ThreadTokenUsage|None)`.
`final_response` is derived from the last agentMessage item whose
`phase == final_answer` (else last unknown-phase agentMessage). In the smoke it
came back `''`/`None` because this model emitted an empty final agentMessage and
expressed the result through the command output, not a final-answer text. Do not
rely on `final_response` being populated; reconstruct from `items` if needed.

## 2d. Config plumbing: per-thread `config={}` vs `CodexConfig.config_overrides`

Two independent channels, both verified:

1. **`CodexConfig(config_overrides=("k=v", ...))`** → becomes **CLI `--config k=v`
   args** when launching `codex app-server` (`client.py:228-230`,
   `args.extend(["--config", kv])`). The codex CLI parses the value as **TOML**
   (`-c, --config <key=value>`, dotted path for nested, falls back to literal
   string if not valid TOML). This is the channel used for provider setup. So
   strings need embedded quotes: `model_providers.vllm.base_url="http://..."`.
   Nested dotted keys like `model_providers.vllm.base_url` work and were how the
   smoke configured the provider.

2. **`thread_start(config={...})`** → a per-thread `dict[str, Any]` carried in the
   `thread/start` JSON-RPC params as the `config` field (`ThreadStartParams.config:
   dict[str, Any] | None`, v2_all.py:6552). It is serialized verbatim via
   `model_dump(by_alias=True, exclude_none=True, mode="json")`
   (`client.py:_params_dict`). Because the field is freeform `dict[str, Any]`,
   the alias machinery does NOT rewrite nested keys — the dict you pass is sent
   as-is. So a nested per-thread dict like
   `config={"model_providers": {"vllm": {"base_url": "..."}}}` is transmitted
   literally. (The app-server then applies it as a config overlay; this path was
   NOT used in the smoke — provider config went through `config_overrides`/CLI.
   For the adapter, prefer `config_overrides` for provider/model-provider setup
   since the CLI TOML parsing and dotted-path semantics are well documented; use
   per-thread `config` only for per-thread overrides if needed.)

Model provider config keys recognized by the CLI (from binary strings):
`name`, `base_url`, `wire_api`, `env_key`, `env_key_instructions`,
`query_params`, `http_headers`, `env_http_headers`, `request_max_retries`,
`stream_max_retries`, `stream_idle_timeout_ms`, `requires_openai_auth`,
`supports_websockets`, `websocket_connect_timeout_ms`, `experimental_bearer_token`.

### env_key: not needed for keyless vLLM
The smoke ran with **no `env_key` and no `env`** at all. vLLM requires no auth,
codex sent no Authorization header, and requests succeeded. Do NOT set `env_key`
for a keyless endpoint (setting it would make codex require that env var to be
present and inject a bearer token). If a keyed endpoint is needed later, set
`model_providers.<p>.env_key="SOME_VAR"` and pass `CodexConfig(env={"SOME_VAR":
"..."})`.

## 2e. Sandbox values and cwd interaction

`Sandbox` enum (`_sandbox.py`), wire values in parens:
- `Sandbox.read_only` (`read-only`) — reads only.
- `Sandbox.workspace_write` (`workspace-write`) — reads anywhere; **writes
  restricted to `cwd` + configured writable roots**; network restricted.
- `Sandbox.full_access` (`full-access`) — no FS restrictions
  (maps to `dangerFullAccess`).

`cwd` + `workspace_write`: yes, writes are restricted to `cwd` and the writable
roots. In the smoke, with `cwd="/tmp/codex-smoke"` and
`sandbox=Sandbox.workspace_write`, the model's own permissions prompt listed the
writable roots as `/Users/carli/.codex/memories`, `/private/tmp`,
`/private/tmp/codex-smoke`, and the macOS temp dir. The command ran in
`/tmp/codex-smoke` and wrote `hello.txt` there without escalation. Set `cwd` on
`thread_start` (thread-level) and/or per-turn (`thread.turn(cwd=...)`); sandbox
can be set thread-level (`thread_start(sandbox=...)`, → `sandbox` mode) or per
turn (`turn(sandbox=...)`, → `sandbox_policy` override).

## 2f. MCP config shape

The Python SDK generated models do NOT define a typed `model_providers` or
`mcp_servers` config struct — provider and MCP config live in the **codex CLI
config schema** (Rust), reached via `config_overrides` (`-c`) or the per-thread
`config` dict. MCP-related strings in the binary: top-level `mcp_servers` table,
keys include `command`, `args`, `env`, `url` (for remote/streamable HTTP),
`bearer_token_env_var`, `http_headers`, `startup_timeout_sec`,
`supports_parallel_tool_calls`, `experimental_environment`, `oauth`,
`oauth_resource`, `scopes`, `client_id`. So an MCP server is configured like a
CLI override, e.g.
`config_overrides=('mcp_servers.fetch.command="uvx"',
'mcp_servers.fetch.args=["mcp-server-fetch"]')`, or via the per-thread `config`
dict `{"mcp_servers": {"fetch": {"command": "uvx", "args": [...]}}}`.
NOTE: codex auto-injects an internal `codex_apps` MCP and tool_search; the
tool catalog the model sees already includes `list_mcp_resources`,
`read_mcp_resource`, etc. plus `mcp__codex_apps__*` entries (observed in the
request `tools` array). The default tool the agent uses for shell is
`exec_command`.

## Exact working configuration (smoke, via shim proxy)

```python
from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

BASE = "http://hl-codon-gpu-020:8000/v1"   # or shim proxy in front of vLLM
config = CodexConfig(
    config_overrides=(
        'model_providers.vllm.name="vLLM (local)"',
        f'model_providers.vllm.base_url="{BASE}"',
        'model_providers.vllm.wire_api="responses"',   # NOT "chat"
        'model_providers.vllm.request_max_retries=2',
        'model_providers.vllm.stream_max_retries=2',
    ),
    env=None,            # keyless: no env_key, no Authorization header
)
codex = Codex(config)
thread = codex.thread_start(
    model="qwen3.5-122b-a10b",
    model_provider="vllm",
    sandbox=Sandbox.workspace_write,
    cwd="/tmp/codex-smoke",
    base_instructions="You are a helpful coding agent.",
    approval_mode=ApprovalMode.auto_review,   # default; auto-approve handler => non-interactive
)
result = thread.run("...prompt...")           # blocking collect
# or: handle = thread.turn("..."); iterate handle.stream(); collect via _collect_turn_result
codex.close()
```

## Items observed (ThreadItem shapes, real JSON)

`TurnResult.items` is a list of `ThreadItem` (a pydantic `RootModel` union;
unwrap with `.root`). Each inner item has a `.type` discriminator. Observed:

- `userMessage`: `{"type":"userMessage","content":[{"type":"text","text":"...",
  "text_elements":[]}]}`
- `reasoning`: `{"type":"reasoning","summary":[],"content":["...chain of thought..."]}`
  (fields: `id, content: list[str], summary: list[str]`)
- `agentMessage`: `{"type":"agentMessage","text":"...","phase":<commentary|final_answer|None>}`
  (fields: `id, text, phase (MessagePhase), memory_citation`)
- `commandExecution`:
  `{"type":"commandExecution","command":"/bin/zsh -lc \"echo 'hello from codex' > hello.txt && ls -la\"",
    "aggregated_output":"total 8\n...hello.txt...","exit_code":0,"status":"completed",
    "cwd":"root='/tmp/codex-smoke'","duration_ms":0}`
  (fields: `id, command, command_actions, cwd, aggregated_output, exit_code,
   duration_ms, status (CommandExecutionStatus: inProgress|completed|failed|declined),
   process_id, source, type`)
- `fileChange`: `{type:"fileChange", changes:[FileUpdateChange], status (PatchApplyStatus), id}`
  (not emitted this run; agent used shell redirection instead of apply_patch)
- `mcpToolCall`: `{type:"mcpToolCall", server, tool, arguments, result, error,
   status, duration_ms, id}` (not emitted this run)
- `webSearch`: `{type:"webSearch", query, action, id}` (not emitted)

Other ThreadItem variants in the union: `hookPrompt`, `plan`,
`dynamicToolCall`, `collabAgentToolCall`.

### Streaming notification methods (smoke2, `handle.stream()`, 206 events)
In order (deltas collapsed): `turn/started`, `item/started`, `item/completed`,
`item/started`, `item/reasoning/textDelta` x136, `item/completed`,
`item/started`, `item/agentMessage/delta` x62, `thread/tokenUsage/updated`,
`turn/completed`. So you get per-item lifecycle, reasoning text deltas, agent
message deltas, a token-usage update, and a terminal `turn/completed`. The
`_collect_turn_result` helper keys on `item/completed`
(`ItemCompletedNotification`), `thread/tokenUsage/updated`
(`ThreadTokenUsageUpdatedNotification`), and `turn/completed`
(`TurnCompletedNotification`).

## Agent loop / Responses request structure (for adapter awareness)
Codex drives a normal Responses tool loop. Per follow-up call it resends the
full running `input` (prior user/assistant messages + `function_call` +
`function_call_output` + `reasoning`), the top-level `instructions`, and a large
`tools` array. The shell tool is `exec_command`; output comes back as a
`function_call_output` whose `output` is a formatted chunk
(`Chunk ID / Wall time / Process exited with code N / Output:` + stdout). Tools
catalog injected by codex includes: `exec_command`, `write_stdin`,
`update_plan`, `request_user_input`, `view_image`, `spawn_agent`/`send_input`/
`resume_agent`/`wait_agent`/`close_agent` (multi-agent), `get_goal`/`create_goal`/
`update_goal`, plus MCP tools (`list_mcp_resources`, `read_mcp_resource`,
`mcp__codex_apps__*`) and freeform custom tools (apply_patch). The adapter may
want to disable unneeded tools via config (e.g. `disabled_tools`, multi_agent
features) to keep the prompt lean for a local model.

## Surprises / gotchas
- **`wire_api="chat"` is dead** in 0.132.0 — plan must change to `responses`.
- **vLLM Responses API is not codex-ready out of the box**: needs `developer`→
  `instructions` folding and `reasoning`-item stripping. This is the single
  biggest adapter risk; decide between a shim proxy vs a patched vLLM.
- `uv pip install` ignores the worktree venv unless `--python .venv/bin/python`
  is passed (it defaulted to `~/miniforge3`).
- The SDK stderr buffer is reachable via `codex._client._stderr_tail(n)`; it was
  empty on these failures (errors arrived as structured `turn.error.message`
  JSON via the JSON-RPC stream, not stderr). `CodexConfig(env={"RUST_LOG":
  "info"})` is available if deeper tracing is needed.
- `final_response` can be empty even on success; reconstruct from `items`.
- Existing `~/.codex/config.toml` (`model="gpt-5.5"`) is overridden by `-c`
  flags and by `thread_start(model=...)`; no conflict observed, but the adapter
  should pass model + provider explicitly rather than relying on user config.
- Approval is auto-accepted by the SDK default handler; the public `Codex`
  facade gives no way to inject a custom handler (would need `CodexClient`
  directly). Fine for non-interactive benchmarking.

## Repro
Smoke script: `scripts/codex_sdk_smoke.py` (untracked). Run with:
`CODEX_SMOKE_BASE_URL="http://127.0.0.1:8011/v1" \
 env -u VIRTUAL_ENV uv run --no-sync python scripts/codex_sdk_smoke.py`
with the shim proxy `/tmp/codex-proxy.py` (REWRITE_ROLES=1) in front of vLLM.
Pointing `CODEX_SMOKE_BASE_URL` directly at vLLM reproduces the
`Unexpected message role` 400.
```
