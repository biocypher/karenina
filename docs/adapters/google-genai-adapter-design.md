# Google GenAI (Gemini) Adapter Design

Phase 2 design for the `google_genai` adapter, built from `docs/adapters/google-genai-context.md`.
Status: draft for review, 2026-10-30.

## 1. Overview

A built-in karenina adapter on the `google-genai` SDK that implements all three ports:

- **AgentPort.** A karenina-owned tool loop. Each turn is one `generate_content` call. The adapter
  runs function and MCP calls itself, passes thought signatures back unchanged, and lets the
  model use Gemini's server-side tools (Google Search, URL reading) inside a call.
- **LLMPort.** Single calls, with optional structured output and streaming.
- **ParserPort.** Delegates to the LLM adapter's structured output.

Two backends: Vertex AI directly (application default credentials, the default) and the
Gemini API style (API key plus optional `base_url`, for example the EBI POC gateway).

## 2. Motivation

The first consumer is the citation-integrity judge. Gemini 4 Argon extracts the claims each
answer supports with a paper, then verifies each citation using the local Europe PMC mirror
(through MCP) with Google Search as a fallback. karenina has no Gemini-native adapter today.
LangChain's Gemini wrapper hides the parts this judge needs:

- the grounding metadata (search queries and sources) that form the judge's audit trail,
- the per-call usage, including thinking and cached tokens,
- control between turns, needed for a search budget.

The SDK's own automatic function calling (AFC) loses intermediate turns, never sums usage,
and cannot change tools mid-run (context doc, "Design questions").

## 3. Design Decisions

| # | Decision | Choice | Why |
|---|---|---|---|
| D1 | Interface name | `google_genai` | Agreed in Phase 1 |
| D2 | Agent tier | `tool_loop` (karenina owns the loop, AFC disabled) | Per-turn usage and grounding, search budget, partial trace on timeout |
| D3 | Backend | `vertex` by default (ADC, project, location `global`). `gemini_api` when `endpoint_base_url` is set or `backend="gemini_api"` | Vertex has no 30 s load-balancer cutoff. Verified on laptop and on codon compute nodes |
| D4 | Built-in tools | Off by default. Enabled per `ModelConfig` (`builtin_tools`). The citation judge turns on both Google Search and URL reading (user decision 2026-10-30), with the prompt asking it to use the mirror first | Search costs $0.014 per query, and page text is billed as input tokens. Only the citation judge needs them |
| D5 | Search budget | Counted per agent run from `grounding_metadata.web_search_queries`. Once the count reaches `max_search_queries`, Google Search is removed from the tools for the remaining turns | Prompt budgets are ignored (16 to 53 queries observed). See open question Q1: soft, a single call can go past it |
| D6 | Retries | `RetryExecutor` is the only retry layer. The SDK's own HTTP retries are switched off | Same rule as `claude_tool` (its design decision D1) |
| D7 | Streaming | Vertex: non-streamed calls by default. `gemini_api`: streamed calls with thought summaries on by default | The gateway cuts responses that stay silent for about 30 s |
| D8 | Server-side tool calls in the trace | Synthetic `ToolUseContent` and `ToolResultContent` pairs named `google_search` and `url_context` | The trace viewer, regex highlighting and the trait's parser already understand tool pairs. The queries and sources stay auditable |
| D9 | Usage | Summed over every call of a run. `cached_content_token_count` maps to `cache_read_tokens`. Thinking counts as output | Matches billing (thinking is billed at the output rate) |
| D10 | Distribution | Built in, registered in `_load_builtins()`, with a `google-genai` extra | Agreed in Phase 1 |
| D11 | Fallback | None (`fallback_interface=None`) | Silent fallback to LangChain would drop the grounding and the budget |
| D12 | Filesystem | None. No file or shell tools. Files only through an MCP server | Gemini's `code_execution` sandbox cannot see karenina's workspace |
| D13 | Concurrency | Every model call, including agent turns, holds one `GlobalLLMLimiter` permit | Respects `max_concurrent_requests` and Vertex quotas. Unlike `claude_tool`, agent calls are gated too |

## 4. Adapter Package Structure

```
src/karenina/adapters/google_genai/
├── __init__.py          # Public exports (adapter classes, settings)
├── registration.py      # AdapterSpec, availability check, ErrorRegistry patterns
├── settings.py          # GoogleGenAISettings: parse and validate ModelConfig fields + extra_kwargs
├── client.py            # Client construction per backend (vertex | gemini_api), HttpOptions, aclose
├── messages.py          # karenina Message <-> types.Content, system_instruction extraction
├── tools.py             # Tool/MCP -> FunctionDeclaration, name sanitising, built-in tools, search budget
├── grounding.py         # grounding_metadata / url_context_metadata -> synthetic tool pairs
├── calls.py             # One model call: streamed or not, finish-reason checks, limiter, retry wrapper
├── llm.py               # GoogleGenAILLMAdapter
├── parser.py            # GoogleGenAIParserAdapter
├── agent.py             # GoogleGenAIAgentAdapter (the turn loop)
├── usage.py             # usage_metadata -> UsageMetadata, aggregation
└── trace.py             # raw_trace via the shared messages_to_raw_trace
```

MCP connection code is shared, not copied: `adapters/claude_tool/mcp.py` moves to
`adapters/_mcp_http.py` and `claude_tool/mcp.py` re-exports it (Q3). Every file stays under 800 lines.

## 5. Registration

```python
# src/karenina/adapters/google_genai/registration.py
def _check_availability() -> AdapterAvailability:
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return AdapterAvailability(
            available=False,
            reason="google-genai not installed. Install with: pip install 'karenina[google-genai]'",
            fallback_interface=None,
        )
    return AdapterAvailability(available=True, reason="google-genai is installed")


_google_genai_spec = AdapterSpec(
    interface="google_genai",
    description="Google GenAI SDK adapter (Gemini on Vertex AI or the Gemini API) with a karenina-owned tool loop",
    agent_factory=_create_agent,
    llm_factory=_create_llm,
    parser_factory=_create_parser,
    availability_checker=_check_availability,
    fallback_interface=None,
    supports_mcp=True,
    supports_tools=True,
    requires_provider=False,
    agent_tier="tool_loop",
)
AdapterRegistry.register(_google_genai_spec)
```

No adapter-specific prompt instructions at first. The default instructions for parsing,
rubric, abstention and sufficiency apply. A `prompts/` subpackage can be added later if
Gemini needs different wording.

### Settings (`ModelConfig` -> `GoogleGenAISettings`)

| ModelConfig field | Use |
|---|---|
| `model_name` | `gemini-4-argon`, `gemini-3.8-flash`, ... (required) |
| `temperature` | `temperature` (Argon ignores it, Flash honours it) |
| `max_tokens` | `max_output_tokens`. Thinking and answer share this budget, so the default 16,384 is kept |
| `system_prompt` | `system_instruction` when the messages carry none |
| `endpoint_base_url`, `endpoint_api_key` | `gemini_api` backend: `HttpOptions(base_url)` and `Client(api_key)`. Without a key, `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| `request_timeout` | `HttpOptions(timeout=ms)` plus an `asyncio.wait_for` guard per call |
| `retry_policy` | `RetryExecutor` |
| `mcp_tool_filter`, `mcp_tool_description_overrides`, `mcp_http_timeout`, `mcp_sse_read_timeout` | Same meaning as in `claude_tool` |
| `extra_kwargs` | The keys below. Unknown keys raise `ValueError` when the adapter is built, so typos fail early |

| `extra_kwargs` key | Default | Meaning |
|---|---|---|
| `backend` | `"gemini_api"` if `endpoint_base_url` is set, else `"vertex"` | `vertex` or `gemini_api` |
| `project` | env `GOOGLE_CLOUD_PROJECT` | Vertex project |
| `location` | env `GOOGLE_CLOUD_LOCATION`, else `"global"` | Vertex location (Gemini 3.8 and 4 exist on `global` only) |
| `builtin_tools` | `[]` | Any of `"google_search"`, `"url_context"` |
| `max_search_queries` | `None` (no budget) | Google Search budget per agent run (D5) |
| `thinking_level` | `None` (model default) | Passed to `ThinkingConfig` |
| `thinking_budget` | `None` | Passed to `ThinkingConfig` |
| `include_thoughts` | `True` on `gemini_api`, `False` on `vertex` | Thought summaries in the response and the trace |
| `stream` | `True` on `gemini_api`, `False` on `vertex` | Use `generate_content_stream` |
| `resolve_grounding_urls` | `False` | Follow `grounding-api-redirect` links to the real URL (one `HEAD` per source) |
| `generate_content_config` | `{}` | Passed through to `GenerateContentConfig` (for example `seed`, `safety_settings`) |

The citation judge's config:

```python
ModelConfig(
    id="argon-judge", interface="google_genai", model_name="gemini-4-argon",
    mcp_urls_dict={"epmc": "http://localhost:<port>/mcp"},
    extra_kwargs={"project": "prj-int-dev-saez-ai-kbench", "builtin_tools": ["google_search", "url_context"],
                  "max_search_queries": 5, "include_thoughts": True},
)
```

## 6. Agent Adapter Core Flow

`GoogleGenAIAgentAdapter.arun(messages, tools, mcp_servers, config)`:

1. **Settings and client.** `config = config or AgentConfig()`. The client is built lazily per backend (`client.py`).
2. **System prompt.** System messages become `system_instruction`, falling back to `config.system_prompt`, then `ModelConfig.system_prompt`.
3. **Function tools.**
   - Static `tools` become `FunctionDeclaration(name, description, parameters_json_schema=input_schema)`.
   - MCP sessions are opened in an `AsyncExitStack` (`_mcp_http.connect_all_mcp_servers`) and their tools listed. `mcp_tool_filter` and the description overrides apply.
   - Names are sanitised to Gemini's pattern (`^[A-Za-z_][A-Za-z0-9_.-]{0,63}$`). A name map turns calls back into the original tool.
   - Two tools with the same sanitised name raise `AgentExecutionError` before any model call.
4. **Built-in tools.** `types.Tool(google_search=GoogleSearch())` and/or `types.Tool(url_context=UrlContext())` from `builtin_tools`.
5. **Contents.** The non-system messages become `types.Content` (`messages.py`).
6. **Turn loop** (inside `asyncio.wait_for(loop, config.timeout)` when a timeout is set). For each turn, `turns += 1`:
   1. Build the `GenerateContentConfig`:
      - `tools` = function tools + built-ins, without Google Search once the budget is spent,
      - `automatic_function_calling=disable`,
      - the thinking config,
      - `tool_config` left at AUTO.
   2. Call the model through `calls.generate(...)`, which:
      - holds a limiter permit for the call only,
      - streams when `stream` is on, accumulating parts, usage and grounding across chunks,
      - applies the `request_timeout` guard,
      - checks the finish reason (Section 14),
      - goes through `RetryExecutor`, so only the failed call is repeated, never the whole run.
   3. Append `candidate.content` to `contents` **unchanged**. Thought signatures must round-trip.
   4. Add to the trace:
      - the synthetic server-side tool pairs for this turn (`grounding.py`),
      - an assistant `Message` built from the parts (thought -> `ThinkingContent` with the base64 signature, text -> `TextContent`, function_call -> `ToolUseContent` with a generated id when the API gives none).
   5. Add this call's usage (Section 12). `search_queries += len(web_search_queries)`.
   6. If there are no function calls, stop: this is the final turn.
   7. Otherwise run each function call:
      - MCP calls go through `session.call_tool`, static tools through their callable,
      - an exception or `isError` becomes an error result, never an exception out of the loop,
      - results go back as one `user` `Content` of `Part.from_function_response(name, response={"result": ...})` (or `{"error": ...}`), and into the trace as `Message.tool_result(...)`.
   8. If `turns >= config.max_turns`, set `limit_reached=True` and stop.
7. **Timeout.** On `TimeoutError`, set `timeout_reached=True` and keep the trace collected so far. With no trace messages at all, raise `AgentTimeoutError`.
8. **No tools.** No function tools and no built-ins means the same loop ends after one turn. One code path, so the timeout, limiter and retries always apply.
9. **Result.**
   - `final_response` = the last assistant text without thoughts (`"[No final response extracted]"` if none).
   - `raw_trace` comes from `trace.py`, with the same notes as `claude_tool` when the turn limit, the timeout, `TOO_MANY_TOOL_CALLS` or `MAX_TOKENS` ended the run.
   - `actual_model` = the last `response.model_version`.
   - `session_id=None`.
10. **Cleanup.** The `finally` block closes the MCP exit stack. `aclose()` closes `client.aio` and the sync client.

`run()` uses the same sync pattern as the current LLM adapters:
- the portal from `async_lifecycle.get_async_portal()` when set,
- otherwise `run_coro_in_thread`, with `compute_sync_wrapper_timeout`.

## 7. Concept Mapping Table

### AgentPort

| karenina | google-genai |
|---|---|
| `AgentConfig.max_turns` | Adapter turn counter, 1 turn = 1 `generate_content` call (no ratio) |
| `AgentConfig.system_prompt` | `GenerateContentConfig.system_instruction` (fallback after system messages) |
| `AgentConfig.timeout` | `asyncio.wait_for` around the whole loop, plus a per-call guard from `request_timeout` |
| `AgentConfig.workspace_path` | Not enforced (no file tools). Left to the prompt, as for other adapters without file tools |
| `AgentConfig.extra` | Run-level overrides of `max_search_queries` and `builtin_tools` (the citation judge can set a budget per citation) |
| `MCPServerConfig` | HTTP/SSE sessions via `_mcp_http`, tools -> `FunctionDeclaration` |
| `AgentResult.raw_trace` | `messages_to_raw_trace(trace_messages)` plus notes |
| `AgentResult.trace_messages` | Built per turn: assistant messages, tool results, synthetic server-side tool pairs |
| `AgentResult.usage` | Sum of every call's `usage_metadata` |
| `AgentResult.limit_reached` | `turns >= max_turns`, or finish reason `TOO_MANY_TOOL_CALLS` |
| `AgentResult.timeout_reached` | `wait_for` fired and some trace was kept |
| `AgentResult.turns` | Number of model calls |
| `AgentResult.actual_model` | `response.model_version` |
| `aclose()` | Closes `client.aio` and the sync client. MCP sessions close per run |
| `capabilities` | `PortCapabilities(supports_system_prompt=True, supports_structured_output=True, supports_streaming=False, supports_file_tools=False, supports_code_execution=False)` |

### LLMPort

| karenina | google-genai |
|---|---|
| `ainvoke(messages)` | One `calls.generate` without function tools. Built-ins apply if configured (server-side, inside the call) |
| `LLMResponse.content` | Joined non-thought text parts |
| `LLMResponse.usage` | `usage.py` on the response (last chunk when streamed) |
| `LLMResponse.is_partial` | True when a streamed call is cut by the wall-clock timeout |
| `LLMResponse.usage_unavailable` | True when the cut stream never delivered usage |
| `with_structured_output(schema, max_retries)` | New adapter with `response_mime_type="application/json"`, `response_json_schema=schema.model_json_schema()`. Content is validated with `schema.model_validate_json` and retried with error feedback |
| `astream(messages)` | `generate_content_stream`, text chunks without thoughts |
| `stream_invoke(messages, timeout)` | Streams under a wall-clock timeout and returns the partial text with `is_partial=True` when it fires |
| `capabilities` | `supports_system_prompt=True`, `supports_structured_output=True`, `supports_streaming=True` |
| `aclose()` | Closes the clients |

The LLM adapter uses direct calls, not the agent loop: it is lighter, and `calls.py` keeps a
single code path for the wire call shared by all three ports.

### ParserPort

| karenina | google-genai |
|---|---|
| `aparse_to_pydantic(messages, schema)` | `llm.with_structured_output(schema, max_retries).ainvoke(messages)` |
| `ParsePortResult.parsed` | The validated model (`LLMResponse.raw`) |
| `ParsePortResult.usage` | The structured call's usage, summed over validation retries |
| `aclose()` | Closes the inner LLM adapter |

## 8. LLM Adapter

- **Text calls:** `ainvoke` -> `calls.generate` -> `LLMResponse(content, usage, raw=response)`.
- **Structured calls:**
  - `response_json_schema` is used, not a forced function call. Forced calls skip search on the OpenAI route, and JSON schema plus Google Search was verified natively.
  - Content is always `parsed.model_dump_json()`, so callers can `json.loads` it. `raw` is the parsed model.
  - **Risk:** Pydantic schemas with `$defs` / `$ref` or `anyOf`. If Gemini rejects one, the adapter retries once with `response_schema=<Pydantic class>`, which lets the SDK convert the schema. The hot tests run the parsing and rubric schemas karenina uses.
- **Sync `invoke`:** the same portal / thread pattern as `claude_tool`, wrapped in `with_llm_semaphore`.
- **Streaming:** `astream` yields text deltas and skips thought parts. `stream_invoke` accumulates under the timeout. `is_partial` and `usage_unavailable` are set as the port requires.

## 9. Parser Adapter

A thin wrapper over the LLM adapter, mirroring `ClaudeToolParserAdapter`:

- `ParseError` when validation fails after all retries,
- `PortError` subclasses pass through,
- usage is preserved from the structured response.

## 10. Message Converter

| karenina | google-genai |
|---|---|
| `Message.system` | Collected into `system_instruction` (joined if several) |
| `Message.user` text | `Content(role="user", parts=[Part(text=...)])` |
| Assistant `TextContent` | `Content(role="model")`, `Part(text=...)` |
| Assistant `ThinkingContent(thinking, signature)` | `Part(text=thinking, thought=True, thought_signature=b64decode(signature))` |
| Assistant `ToolUseContent(id, name, input)` | `Part(function_call=FunctionCall(id, name=sanitised(name), args=input))` |
| Tool `ToolResultContent(tool_use_id, content, is_error)` | `user` `Content` with `Part.from_function_response(name=<looked up from tool_use_id>, response={"result" or "error": content})` |
| Synthetic server-side pairs (`google_search`, `url_context`) | Dropped when converting back. The real `Content` objects with their signatures are what the loop sends |

Inside a run the loop sends the SDK's own `Content` objects back, never re-converted ones, so
signatures and grounding stay exact. The converter matters for incoming messages (prompt
assembly and replayed conversations).

## 11. Trace Extraction

- `trace_messages` are built while the loop runs, and `raw_trace` comes from the shared
  `benchmark.verification.utils.trace_formatting.messages_to_raw_trace`. Thought summaries are
  included only when `include_thoughts` is on.
- **Server-side tools (`grounding.py`).** For a turn whose candidate has grounding:
  - an assistant `ToolUseContent(id=f"google_search_{turn}", name="google_search", input={"queries": [...]})`,
  - then a tool `ToolResultContent` whose content is JSON: `{"server_side": true, "sources": [{"title", "uri"}], "supported_segments": n}`,
  - URL reading gives `url_context` pairs listing each `retrieved_url` and its status.

  These come before the model's text for that turn, the order in which they happened.
  `searchEntryPoint.renderedContent` (HTML) is dropped.

## 12. Usage Extraction

Per call (`usage.py`):

| UsageMetadata | From `usage_metadata` |
|---|---|
| `input_tokens` | `prompt_token_count` + `tool_use_prompt_token_count` (the prompt count already includes cached tokens) |
| `output_tokens` | `candidates_token_count` + `thoughts_token_count` |
| `total_tokens` | `total_token_count`, or the sum when absent |
| `cache_read_tokens` | `cached_content_token_count` |
| `reasoning_tokens` (new, C4) | `thoughts_token_count` |
| `web_search_queries` (new, C4) | `len(grounding_metadata.web_search_queries)` |
| `cost_usd` | `None`. karenina has no price table. The experiment computes cost from tokens and queries |
| `model` | `model_name` |

`aggregate_usage` sums every field, keeping `None` only when no call reported it. A streamed
call takes usage from the last chunk that carries it.

## 13. MCP Conversion

- MCP `inputSchema` goes straight into `parameters_json_schema` (JSON Schema is accepted
  natively).
- Calling a tool: `await session.call_tool(original_name, args)`. The text content blocks are
  joined. `isError` or an exception gives `{"error": text}` to the model and `is_error=True` in
  the trace.
- Tool results longer than a limit (`AgentConfig.extra["max_tool_result_chars"]`, default 50,000)
  are cut with a marker, so a full-text dump cannot blow up the context.

## 14. Error Mapping

| Condition | Handling |
|---|---|
| `google.genai.errors.ServerError` (5xx, including the gateway's HTML 500) | `SERVER_ERROR`, retried per call (type name registered) |
| `ClientError` 429 / `RESOURCE_EXHAUSTED` | `RATE_LIMIT` (built-in "429" rule plus a registered `resource_exhausted` substring) |
| Other `ClientError` (400, 403, 404) | `PERMANENT`, raised as `AgentExecutionError` / `PortError` with the API message |
| `google.auth` `RefreshError` / `DefaultCredentialsError` | `PERMANENT`. Message: "Google login missing or expired: run gcloud auth application-default login" |
| Stream ended with no `finish_reason` | `GeminiIncompleteResponseError`, registered as `SERVER_ERROR`, retried |
| Finish `STOP` | Normal |
| Finish `MAX_TOKENS` | Keep what came back, add a trace note. In structured mode, a `ParseError` follows if the JSON is incomplete |
| Finish `TOO_MANY_TOOL_CALLS` (server-side search loop) | Agent: end the run with `limit_reached=True` and a note. LLM: raise `PortError` with the finish reason. Never an empty success |
| Finish `MALFORMED_FUNCTION_CALL` / `UNEXPECTED_TOOL_CALL` | Agent: add a short corrective user note and continue (counts as a turn). LLM: `PortError` |
| Finish `SAFETY`, `RECITATION`, `PROHIBITED_CONTENT`, `BLOCKLIST`, `SPII`, `OTHER` | `PortError` (permanent) carrying the finish reason |
| Finish reasons unknown to the installed SDK | Compared by name string, so newer reasons map to "permanent with reason" without a crash |

The patterns are registered on the adapter's own `ErrorRegistry` instance, as `claude_tool` does.

## 15. Dependencies

- `google-genai>=1.56,<3`. The locked 1.68.0 already has everything this design uses (checked:
  `ThinkingConfig.thinking_level`, `include_thoughts`, `response_json_schema`,
  `FunctionDeclaration.parameters_json_schema`, `GroundingMetadata.web_search_queries`,
  `thoughts_token_count`).
- Version 2.x is not required. It is also blocked today, because the core dependency
  `langchain-google-genai` 4.2.1 pins `google-genai<2`. `langchain-google-genai` 4.4.2
  resolves with `google-genai` 2.34 if a later bump is wanted.
- New extra in `pyproject.toml`: `google-genai = ["google-genai>=1.56,<3"]`. It is already met
  through the core dependency, but it documents the direct requirement.
- `google-auth` (ADC) comes with `google-genai`. `mcp` is already a core dependency.

## 16. Integration Points

Adapter (built in):

1. `adapters/registry.py` `_load_builtins()`: one `try: from karenina.adapters.google_genai import registration` block.
2. `pyproject.toml`: the `google-genai` extra.
3. `adapters/_mcp_http.py`: the shared MCP helper moved out of `claude_tool` (re-exported there).
4. Docs: an adapter page next to the other adapters, and the interface listed in the verification skill's adapter table (ask before editing skills).

Companion changes the citation judge needs, as separate commits on this branch:

- **C1. `AgenticRubricTrait` uses the agent when the judge has MCP servers.** Today `trace_only`
  with no workspace always takes the single-call LLM path, so the judge could never reach the
  mirror tools. `needs_tools` becomes true when the judge's `mcp_urls_dict` is set.
  Built-in-only judges keep the LLM path, where Gemini still searches inside the call.
- **C2. The trait passes the judge's MCP servers.** `mcp_servers` is built from the judge's
  `mcp_urls_dict` exactly as `generate_answer` does (`{"type": "http", "url": url}`).
- **C3. The trait keeps the judge's usage.** `AgentResult.usage` and the LLM and parser usage
  reach the result's usage tracking. This is the Rikugan bug
  `bugs/agentic-rubric-usage-and-noncl-cache-reads.md`, part 1.
- **C4. `UsageMetadata` gains two optional fields:** `reasoning_tokens` and `web_search_queries`.
  Both default to `None`, appear in `to_dict()` only when set, and are summed wherever usage is
  aggregated. `claude_tool` can later fill `web_search_queries` from Anthropic's
  `server_tool_use.web_search_requests`.

- **C5. A failed investigation leaves a record.** Today `AgenticTraitEvaluator.evaluate_trait` catches any
  investigation exception, logs a warning and returns `(None, None)`. The result then holds no score, no trace and
  no reason. The change stores `[investigation failed: <error type>: <message>]` as the trait's trace and marks
  the row as an infrastructure failure, so it is excluded from the scores and re-run on resume, never counted as
  a judgment. Partial traces from timeouts and turn limits are already kept, because the adapter returns them
  instead of raising.

- **C6. The judge's structured trace is stored too.** `VerificationResultRubric` gains
  `agentic_trait_investigation_messages: dict[str, list[dict[str, Any]]] | None` (keyed by trait name, default
  `None`). It holds the serialised `AgentResult.trace_messages` in the same format as the answering model's
  `template.trace_messages`. The evaluator passes `trace_messages` through alongside `raw_trace` (the LLM-only
  path stores a single assistant message). Older result files stay readable. This makes the judge's tool calls,
  searches and page fetches readable by field rather than by pattern matching, and lets the GUI's structured
  trace viewer render judge traces. User decision 2026-10-30.

Outside karenina (the experiment folder `paper_examples/rubrics/agentic/citation_integrity/gemini_judge/`):

- A small MCP server (streamable HTTP), started by `codon_job.sbatch` beside the mirror replicas, with:
  - `europepmc_search`, `europepmc_get` and `europepmc_fulltext_find` against the mirror replicas;
  - `fetch_url`, which opens a web page with httpx and extracts its text with trafilatura.

  `fetch_url` exists because Gemini's `url_context` only opens addresses written in the request:
  given only a PMID or DOI, Argon searched instead, then claimed URLs it never opened. Its policy
  (Rikugan variable `gemini_judge_fetch_tool`, user decision 2026-10-30):
  - browser identity, with robots.txt not consulted;
  - at most 1 request per second per site, and pages cached per run;
  - Europe PMC addresses refused, because the mirror covers them;
  - 20k characters per page and 3 pages per citation.

  The fetch_url-tool spike is Rikugan run `fetch-tool-spike`. `url_context` stays on for links
  written in the answer. Page opens are audited from `url_context_metadata` and the tool log,
  never from the model's text.
- The per-citation unit (one judge run per citation, so the search budget is per citation) is
  the experiment's design. It is not part of the adapter.

## 17. Open Questions (with defaults)

| # | Question | Default |
|---|---|---|
| Q1 | How strict is the search budget? Gemini searches inside a single call, so the budget can only remove Google Search between calls | **Decided 2026-10-30: soft.** The budget removes Google Search between calls (D5), and the real query count goes into usage and the trace. A strict client-side `web_search` tool was considered and not chosen |
| Q2 | Streaming default per backend | Vertex off, gateway on with thought summaries |
| Q3 | Move the MCP helper to `adapters/_mcp_http.py`, or import it from `claude_tool` | Move it, and re-export from `claude_tool.mcp` |
| Q4 | Add `reasoning_tokens` and `web_search_queries` to `UsageMetadata` (C4) | Yes |
| Q5 | Resolve grounding redirect URLs | Off. The citation judge cites PMIDs from the mirror, not web URLs |
| Q6 | Thought summaries in agent traces | Off on Vertex by default. The citation judge turns them on so curators can see the reasoning |
| Q7 | Per-call `request_timeout` default when the pipeline sets none | 300 s on Vertex, so one stuck call cannot hold the whole run |

## 18. Success Criteria

- **Conformance:** the shared adapter conformance suite passes cold (mocked client) for all three ports.
- **Cold tests:**
  - the message round trip, including base64 thought signatures,
  - name sanitising and its reverse map,
  - the grounding-to-trace pairs,
  - usage aggregation over several turns, including cached and thinking tokens,
  - every finish reason in Section 14,
  - the search budget removing the tool after it is reached,
  - timeout with a partial trace, and the turn limit,
  - unknown `extra_kwargs` keys rejected.
- **Hot tests on Vertex** (opt-in, as for other adapters):
  - a plain call,
  - structured output with karenina's parsing schema,
  - an agent run over a test MCP server with at least 2 tool turns,
  - Google Search on, with queries in usage and the trace,
  - a forced timeout returning a partial trace,
  - an agent run on the gateway backend, streamed.
- **Citation smoke:** the 4-citation check from `gemini_judge/smoke_vertex_mirror.py`, run
  through karenina (`AgenticRubricTrait` + this adapter + the Europe PMC MCP server), gives the
  same verdicts as the scratch script. Its usage must match the scratch script's numbers for
  the same calls.
- **Tool hygiene:** ruff, mypy and the pre-commit hooks pass. No file over 800 lines.
