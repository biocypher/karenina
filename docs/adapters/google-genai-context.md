# Google GenAI (Gemini) Adapter Context

Phase 1 context for a Karenina adapter built on the `google-genai` SDK. The first consumer is an `AgenticRubricTrait` judge for the citation-integrity experiment: Gemini 3.8 Flash with Google Search and URL reading, reached through the EBI POC AI gateway.

All behaviour marked "verified" was probed live on 2026-10-30 against `gemini-3.8-flash-eu` through `https://poc-ai-gateway.ebi.ac.uk`.

## SDK overview

- Package: `google-genai` (import `google.genai`)
- Docs: <https://googleapis.github.io/python-genai/>
- Source: <https://github.com/googleapis/python-genai>
- Version: `>=2.30`. The latest is 2.34.0 (`requires_python >=3.11`). 2.x adds `ThinkingConfig.thinking_level`, which 1.46 lacks.
- Karenina's venv currently has 1.46.0, pulled in transitively by `google-adk` 1.16 and `google-cloud-aiplatform` 1.122. Whether the bump to 2.x conflicts with them must be checked in the design phase.

## Capabilities

| Capability | Supported | Notes |
| --- | --- | --- |
| Agent loop | Partial | The SDK's automatic function calling (AFC) runs a loop for Python callables and MCP `ClientSession`s (`maximum_remote_calls`, default 10). Server-side built-in tools (`google_search`, `url_context`, `code_execution`) run inside a single call. Verified: AFC with an async callable works (3.5 s). |
| Async | Yes | `client.aio.models.generate_content(...)` and `generate_content_stream(...)` (async iterator). `await client.aio.aclose()`. Verified. |
| MCP | Via AFC only | AFC accepts `mcp.ClientSession` objects in `tools` (experimental). 2.x also has `Tool.mcp_servers`, which is server-side and untested through the gateway. A Karenina-owned loop opens MCP sessions itself and converts their tools to `FunctionDeclaration`s. |
| Tool definition | Yes | `types.Tool(function_declarations=[types.FunctionDeclaration(name, description, parameters_json_schema={...})])`. Built-ins: `types.Tool(google_search=types.GoogleSearch())`, `types.Tool(url_context=types.UrlContext())`. Verified: built-in and function tools combine in one request, and the model chooses between them. |
| Message types | Yes | `types.Content(role="user" or "model", parts=[types.Part])`. A `Part` carries one of: `text`, `thought` (bool, with `text`), `thought_signature` (bytes), `function_call` (`name`, `args`, `id`), `function_response`, `executable_code`, `code_execution_result`. Tool results go back as a `user` turn with `Part.from_function_response(name=..., response={...})`. |
| Traces | Adapter-built | In manual mode the SDK holds no history. The adapter's `contents` list is the conversation. Each candidate carries `grounding_metadata` (`web_search_queries`, `grounding_chunks`, `grounding_supports`, `search_entry_point`) and `url_context_metadata` (`url_metadata[].retrieved_url`, `url_retrieval_status`). |
| Usage | Yes, per call | `usage_metadata`: `prompt_token_count`, `candidates_token_count`, `thoughts_token_count`, `tool_use_prompt_token_count`, `cached_content_token_count`, `total_token_count`. The adapter aggregates across turns. In streaming, usage arrives on chunks, and the last one holds the totals (verified). |
| Streaming | Yes, required | `generate_content_stream` yields `GenerateContentResponse` chunks. The final chunk carries `finish_reason` and usage. Grounding metadata arrives on the chunk after search. Partial content is kept if the stream is cut. See the gateway constraints below. |
| Structured output | Yes | `response_mime_type="application/json"` plus `response_json_schema` (dict) or `response_schema` (Pydantic class, which fills `response.parsed`). Usage is preserved on the same response. Verified: it works together with `google_search` (17.6 s, 5 queries, valid JSON). |
| No-tools behaviour | Single call | A plain completion. Without `include_thoughts`, long silent thinking hits the gateway's 30 s silence cutoff (verified failure at 30.5 s). With `include_thoughts=True`, thought summaries stream from about 2 s and keep the connection alive (verified: 101 s run succeeded). |
| System prompts | Yes | `GenerateContentConfig.system_instruction`. Verified. |
| Turn limits | Adapter-owned | AFC has `maximum_remote_calls`. A Karenina loop counts turns itself and returns the partial trace with `limit_reached=True` at `max_turns`. |
| Agent timeout recovery | Yes (adapter-owned loop) | The accumulated `contents` survive an `asyncio.wait_for` cancellation, so the partial trace is recoverable. AFC would lose intermediate turns. |
| Thinking | Yes | `ThinkingConfig(thinking_level=..., thinking_budget=..., include_thoughts=...)`. Summaries come back as `Part(thought=True)`. Gemini 3 requires `thought_signature`s to be sent back with function-call parts: append the model `Content` unchanged. Verified over a 2-turn loop. |
| Filesystem/workspace | No | No file tools. `code_execution` is a server-side Python sandbox with no access to Karenina's workspace. File access only through an MCP filesystem server. |
| Backend/runtime | Configurable | Gemini API style (`api_key`) or Vertex (`vertexai=True`, project, location). The gateway uses `api_key` plus `HttpOptions(base_url=...)` on the native `/v1beta/models/{model}:generateContent` and `:streamGenerateContent` routes. |

## EBI gateway constraints (verified)

- **Routes.** The client is `genai.Client(api_key=<POC key>, http_options=types.HttpOptions(base_url="https://poc-ai-gateway.ebi.ac.uk"))`. The native `/v1beta/models/...` routes work. The `/gemini/...` route returns 500, and `/vertex_ai/...` returns 401 because it wants OAuth.
- **Silence cutoff.** The front proxy closes any response that sends no bytes for about 30 s:
  - a non-streaming call dies at about 32 s with an HTML "Error: 500 | EMBL-EBI" page, which the SDK raises as `ServerError` 500;
  - a stream survives as long as chunks keep coming (a 54 s essay streamed fine).
- **Silent phases while searching.** With search tools on, nothing streams until the server-side searches finish. First bytes came at 7 to 22 s, and the slowest case made 10 queries plus URL reading. A heavy check can cross 30 s.
- **Runaway search, not a gateway cutoff.** A citation check that asked Gemini to read a PubMed page streamed thought summaries early, then ended after 50 to 72 s with no text, no `finish_reason` and no exception. The same happens directly on Vertex (see below), so the gateway is not the cause.
  - The cited PMID (19152484) does not exist: NCBI esummary "cannot get document summary", and it is absent from the Europe PMC mirror. The page fetch failed (`URL_RETRIEVAL_STATUS_ERROR`) because there was nothing to fetch.
  - PubMed does not block the fetcher: `url_context` read a real PubMed page in 4.2 s (`URL_RETRIEVAL_STATUS_SUCCESS`).
  - With only search and URL reading, Gemini does not conclude that a reference is fabricated. It keeps searching until `TOO_MANY_TOOL_CALLS`: 70 queries and 105 s when not streamed. When streamed, it ended with no `finish_reason`.
  - A stream that ends without a `finish_reason`, or with `TOO_MANY_TOOL_CALLS`, must be treated as a failure, never as an empty answer.
- **Forced function calls skip the search.** On the OpenAI-compatible route, a forced function call made Gemini skip search entirely (empty fields in 1.6 s). On the native route, use `response_json_schema` for structured output instead.
- **Source URLs.** Grounding source URLs are `vertexaisearch.cloud.google.com/grounding-api-redirect/...` links. A `HEAD` request without following redirects gives the real URL in `Location`. In one streamed run, `grounding_chunks` was empty while `web_search_queries` was filled.

## Direct Vertex AI (verified)

- **Client:** `genai.Client(vertexai=True, project="prj-int-dev-saez-ai-kbench", location="global")` with local Application Default Credentials (`authorized_user`).
- **Models:** `gemini-3.8-flash` and `gemini-4-argon` are on `global` only. `europe-west4` has only the 2.5 models.
- **No silence cutoff:** a non-streamed 53 s call succeeded (the same call fails through the gateway at 32 s).
- **Same runaway search on a fabricated PMID:** Gemini ran 70 searches and finished with `TOO_MANY_TOOL_CALLS` (105 s, no text). Streamed, the same check ran 156 s and ended with no `finish_reason`. With JSON output it ran 152 s, made 67 searches and also returned no text.
- **With Europe PMC mirror tools, the spiral is gone.** Three client-side function tools (`europepmc_search`, `europepmc_get`, `europepmc_fulltext_find`) were served from the local mirror replicas, with Google Search as fallback, AFC off, and a cap of 8 turns and 150 s. All 4 test citations finished with `STOP`:

  | Citation | Time | Turns | Verdict |
  |---|---|---|---|
  | misattributed PMID | 8.4 s | 3 | exact / not_backed, read the full text, 0 Google searches |
  | fabricated PMID | 76.5 s | 7 | no_match, "PMID does not exist" |
  | wrong journal and PMID (Sobhian, Cell vs Science) | 55.1 s | 7 | partial / not_backed |
  | paywalled database row (Lynch 2017) | 38.1 s | 2 | exact / cannot_tell |

- **The prompt's search budget is ignored.** "At most 5 Google searches" was exceeded with 16 to 29 queries per citation. A hard budget must be enforced by code, for example by dropping `google_search` from the tools once the cumulative `web_search_queries` count passes the budget.
- **Billing:** grounding with Google Search is billed per grounded request, to the GCP project.

## Key API surface

- **Client:** `genai.Client(api_key=..., http_options=types.HttpOptions(base_url=..., timeout=<ms>))`, async via `client.aio`.
- **Call:** `await client.aio.models.generate_content_stream(model=..., contents=[types.Content, ...], config=types.GenerateContentConfig(...))`.
- **Config:** `system_instruction`, `tools`, `tool_config`, `automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)`, `thinking_config`, `temperature`, `max_output_tokens`, `response_mime_type`, `response_json_schema`, `seed`, `safety_settings`.
- **Response:** `candidates[0].content.parts`, `candidates[0].finish_reason` (`STOP`, `MAX_TOKENS`, `SAFETY`, `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`, ...), `candidates[0].grounding_metadata`, `candidates[0].url_context_metadata`, `usage_metadata`, `model_version`.

## Concept mapping (preliminary)

| Karenina concept | google-genai equivalent |
| --- | --- |
| System message | `GenerateContentConfig.system_instruction` |
| `Message.user` | `Content(role="user", parts=[Part(text=...)])` |
| Assistant `TextContent` | `Content(role="model")`, `Part.text` with `thought` unset |
| `ThinkingContent(thinking, signature)` | `Part(thought=True, text=...)`. `Part.thought_signature` is bytes: base64 into `signature`, and must round-trip |
| `ToolUseContent(id, name, input)` | `Part.function_call` (`id` may be None, so the adapter generates one) |
| `ToolResultContent(tool_use_id, content, is_error)` | `Part.from_function_response(name, response)` in a `user` turn |
| Server-side search and URL reading | `grounding_metadata` and `url_context_metadata` per turn. To be rendered into `raw_trace` and `trace_messages` (design phase) |
| `UsageMetadata(input, output, total, cache_read)` | `prompt_token_count` (+ `tool_use_prompt_token_count`), `candidates_token_count` + `thoughts_token_count`, `total_token_count`, `cached_content_token_count`. The exact sums are settled in the design phase |
| `AgentConfig.max_turns` | Adapter turn counter |
| `AgentConfig.timeout` | `asyncio.wait_for` around the loop. The partial trace is returned with `timeout_reached=True` |
| `AgentResult.actual_model` | `response.model_version` |
| `ModelConfig.endpoint_base_url` / `endpoint_api_key` | `HttpOptions(base_url)` / `Client(api_key)`. Without them, fall back to the default Google endpoint and the `GEMINI_API_KEY` / `GOOGLE_API_KEY` env vars |
| `ModelConfig.temperature` / `max_tokens` | `temperature` / `max_output_tokens` |
| `ModelConfig.extra_kwargs` | Built-in tool list, `thinking_level`, `include_thoughts`, and other config passthrough |
| `ParserPort.aparse_to_pydantic` | One streamed call with `response_mime_type="application/json"` and `response_json_schema=schema.model_json_schema()` |
| `LLMPort.ainvoke` | One streamed call, no function tools (built-ins only if enabled) |

## Critical integration points

- **Filesystem access:** none built in. `supports_file_tools` and `supports_code_execution` are False unless MCP servers provide them.
- **Usage tracking:** one usage object per call, summed across turns by the adapter. Streaming usage is taken from the last chunk.
- **Structured output:** keeps the raw response, and therefore usage. The return type is a JSON string in `text`, plus `parsed` when `response_schema` is a Pydantic class.
- **Recursion limit:** an adapter-owned turn counter (no SDK parameter, no mapping ratio). On the limit, return the partial trace with `limit_reached=True`.
- **No-tools behaviour:** a single call. Set `include_thoughts=True` so long thinking streams and doesn't hit the gateway cutoff.
- **Streaming:** an async iterator. Usage and `finish_reason` are on the final chunk. A missing `finish_reason` means the stream was cut off, so retry.
- **Agent timeout:** the partial trace is recoverable from the adapter's `contents`.
- **Retries:** the gateway's HTML 500, silent truncation and transient 5xx must map onto `ErrorRegistry` / `RetryExecutor` as retryable.
- **Thought signatures:** the model's `Content` must be appended unchanged before function responses, or Gemini 3 rejects or degrades the next turn.
- **MCP:** with a Karenina-owned loop, the adapter opens the sessions itself. `adapters/claude_tool/mcp.py` is the closest existing pattern: a Karenina-run tool loop over the `mcp` client.

## Design questions (resolved with the user, 2026-10-30)

- **Natively agentic:** only for server-side built-ins. The agent loop is **Karenina-owned**: the adapter calls the model once per turn, executes function and MCP calls, and passes thought signatures back. AFC is disabled. Reconfirmed after reading the SDK 2.34 AFC loop:
  - each call overwrites the previous response, so intermediate turns' grounding and usage are lost;
  - usage is never summed;
  - the config is fixed for the whole loop, so no hard search budget is possible;
  - the history is lost on timeout;
  - the call limit is hit silently.

  Prompt caching is the same under both loops (append-only contents, so implicit cache hits). Only the accounting differs.
- **Backend:** selectable. The main route is direct Vertex AI (`vertexai=True`, project, location `global`, ADC), with no silence cutoff. The EBI gateway (`api_key` + `base_url`) is a second option, and it requires streaming plus the cutoff handling.
- **Search budget:** enforced by code. The adapter counts `web_search_queries` per turn and drops `google_search` from the tools once a configured budget is reached (Gemini ignored a prompt budget of 5 and used 16 to 29). Changing the tools ends implicit caching for the remaining turns, which is accepted.
- **Usage:** `cached_content_token_count` maps to `cache_read_tokens`, summed over all calls.
- **Built-in tools:** **off by default**, enabled per `ModelConfig` (for example `extra_kwargs={"builtin_tools": ["google_search", "url_context"]}`). The citation-integrity judge turns them on.
- **Interface name:** `google_genai`.
- **Ports:** AgentPort, LLMPort and ParserPort.
- **Availability:** an import check on `google.genai`.
- **Fallback:** none.
- **Filesystem backend:** none (MCP only).
- **Distribution:** built in (`karenina/src/karenina/adapters/google_genai/`, registered in `_load_builtins()`), with `google-genai>=2.30` in an optional extra.

## Open items for the design phase

- The `google-genai` 2.x bump against `google-adk` 1.16 and `google-cloud-aiplatform` 1.122, and which extra pulls those in.
- How server-side search and URL reading appear in `raw_trace` and `trace_messages`: synthetic tool-use and tool-result pairs, or a dedicated text block. Queries, sources and URL statuses must be kept, because they are the citation judge's audit trail.
- Whether to resolve grounding redirect URLs to real URLs inside the adapter or leave that to the caller.
- The retry policy for the gateway failure modes above, and whether to ask the gateway admins to raise the 30 s silence cutoff.
- How `AgenticRubricTrait` reaches the built-ins: the trait passes no tools, so the adapter must read them from `ModelConfig`.
- How the citation judge's Europe PMC mirror tools reach the agent. The trait passes no tools, so they should come in as an MCP server on the judge's `ModelConfig` (`mcp_urls_dict`): a small stdio or HTTP server in the experiment folder exposing `europepmc_search`, `europepmc_get` and `europepmc_fulltext_find` against the mirror replicas.
- Karenina gap found while checking caching: `AgenticRubricTrait` discards the judge agent's `AgentResult.usage` (and the parser's), so judge tokens and cost never reach the results. LangChain usage also ignores non-Anthropic cache reads. Both are logged in Rikugan `citation-integrity` at `bugs/agentic-rubric-usage-and-noncl-cache-reads.md`. The first matters for reporting the citation judge's cost.
