# Environment Variables Reference

This is the exhaustive reference for every environment variable recognized by karenina. For a tutorial-style introduction with grouped explanations and examples, see [Environment Variables Tutorial](../../workflows/configuration/environment-variables.md).

!!! note "Precedence"
    Environment variables are the **third priority** in the [configuration hierarchy](../../workflows/configuration/index.md): CLI arguments and preset values override them; they override built-in defaults.

---

## Complete Reference Table

### API Keys

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENAI_API_KEY` | `str` | — | OpenAI API key. Required when using the `langchain` interface with OpenAI models. |
| `ANTHROPIC_API_KEY` | `str` | — | Anthropic API key. Required when using `claude_agent_sdk`, `claude_tool`, or `langchain` with Anthropic models. Loaded via `dotenv` in Claude adapters. |
| `CLAUDE_CODE_OAUTH_TOKEN` | `str` | — | Claude subscription OAuth token (created by `claude setup-token`). When neither `ANTHROPIC_API_KEY` nor a model-level `anthropic_api_key` is configured, the `claude_agent_sdk` adapter forwards this from the host environment to the SDK subprocess, so the agent can authenticate through a Claude subscription instead of an API key. |
| `ANTHROPIC_AUTH_TOKEN` | `str` | — | Alternative Anthropic auth token forwarded alongside `CLAUDE_CODE_OAUTH_TOKEN` by the `claude_agent_sdk` adapter when no API key is configured. |
| `GOOGLE_API_KEY` | `str` | — | Google AI API key. Required when using the `langchain` interface with Google models (e.g., `google_genai` provider). |
| `OPENROUTER_API_KEY` | `str` | — | OpenRouter API key. Required when using the `openrouter` interface. Read via `os.environ.get()` in the OpenRouter model factory. |
| `TAVILY_API_KEY` | `str` | — | Tavily search API key. Required when deep judgment web search is enabled (`search_enabled=True`). Used by the Tavily search utility. |

### Embedding Check

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EMBEDDING_CHECK` | `bool` | `false` | Enable embedding similarity checking. When enabled, compares the LLM's parsed answer to ground truth using vector similarity. Read by `VerificationConfig.__init__()` to set `embedding_check_enabled` if not explicitly provided. The CLI defers to this env var when `--embedding-check` / `--no-embedding-check` is not passed. |
| `EMBEDDING_CHECK_MODEL` | `str` | `all-MiniLM-L6-v2` | SentenceTransformer model name for embedding computation. Read by `VerificationConfig.__init__()` to set `embedding_check_model` if not explicitly provided. The CLI defers to this env var when `--embedding-model` is not passed. |
| `EMBEDDING_CHECK_THRESHOLD` | `float` | `0.85` | Similarity threshold (0.0 to 1.0). Read by `VerificationConfig.__init__()` to set `embedding_check_threshold` if not explicitly provided. The CLI defers to this env var when `--embedding-threshold` is not passed. Invalid values fall back to the default. |

### Async Execution

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `KARENINA_ASYNC_ENABLED` | `bool` | `true` | Enable parallel verification execution across multiple questions. Also read by `VerificationConfig.__init__()` to set `async_enabled` if not explicitly provided. |
| `KARENINA_ASYNC_MAX_WORKERS` | `int` | `2` | Maximum number of parallel workers for async execution. Also read by `VerificationConfig.__init__()` to set `async_max_workers` if not explicitly provided. Invalid values fall back to the default. |
| `KARENINA_MAX_CONCURRENT_LLM_REQUESTS` | `int` | — | Global cap on concurrent LLM requests across all workers. Read by `VerificationConfig.__init__()` to set `max_concurrent_requests` if not explicitly provided. Useful for self-hosted inference servers (vLLM, SGLang). Invalid values fall back to no cap. |

### Path Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DB_PATH` | `str` | `./dbs/` | Directory for SQLite databases. Set by `karenina serve` relative to the working directory if not already defined (`os.environ.setdefault`). |
| `KARENINA_PRESETS_DIR` | `str` | `./presets/` | Directory where preset JSON files are stored and searched. Used by `resolve_preset_path()` to locate presets. Set by `karenina serve` if not already defined. Created by `karenina init`. |
| `MCP_PRESETS_DIR` | `str` | `./mcp_presets/` | Directory for MCP configuration preset files. Set by `karenina serve` relative to the working directory if not already defined. Created by `karenina init`. |

### Other Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AUTOSAVE_DATABASE` | `bool` | `true` | Automatically save verification results to the SQLite database after each run. Disable to keep results in-memory only. |
| `KARENINA_EXPOSE_GROUND_TRUTH` | `bool` | `false` | Include ground truth values in the judge LLM prompt during template evaluation. **Debugging only** — enabling this biases the judge and invalidates benchmark results. |
| `KARENINA_TRACE_TRUNCATION_THRESHOLD` | `int` | (module default) | Maximum number of characters retained per trace when materializing failure cases for error analysis or scenario trace materialization. Read by `benchmark/error_analysis/case_renderer.py` and `scenario/trace_materialization.py`. (`benchmark/error_analysis/materializer.py` forwards a `max_trace_chars` override but does not read the env var itself: the actual read happens in `case_renderer.py`.) The `analyze-errors` CLI accepts `--max-trace-chars` as an explicit override. Invalid values fall back to the module default. |
| `CLAUDE_CONFIG_DIR` | `str` | — | Forwarded by the `claude_agent_sdk` agent adapter to the spawned Claude CLI subprocess. When set on the parent process, the child inherits the same Claude config directory; otherwise the SDK falls back to its own resolution. |

---

## Reading Conventions

### Boolean Variables

There is no single shared boolean parser. The exact truthy set depends on the read site (all comparisons are case-insensitive):

- **Most variables** (`EMBEDDING_CHECK` and `KARENINA_ASYNC_ENABLED` as read by `VerificationConfig.__init__()`, and `AUTOSAVE_DATABASE`) accept `true`, `1`, `yes`.
- **`EMBEDDING_CHECK`** (via the embedding-check utility read) and **`KARENINA_EXPOSE_GROUND_TRUTH`** additionally accept `on`.
- **`KARENINA_ASYNC_ENABLED`** as read by `batch_runner` accepts exactly `true` (nothing else counts as true at that read point).
- **False**: anything not in the accepted set (including `false`, `0`, `no`, `off`, or unset).

### VerificationConfig Integration

Six environment variables are read directly by `VerificationConfig.__init__()` as fallbacks when the corresponding field is not set explicitly (via constructor, preset, or CLI):

| Env Var | VerificationConfig Field |
|---------|-------------------------|
| `EMBEDDING_CHECK` | `embedding_check_enabled` |
| `EMBEDDING_CHECK_MODEL` | `embedding_check_model` |
| `EMBEDDING_CHECK_THRESHOLD` | `embedding_check_threshold` |
| `KARENINA_ASYNC_ENABLED` | `async_enabled` |
| `KARENINA_ASYNC_MAX_WORKERS` | `async_max_workers` |
| `KARENINA_MAX_CONCURRENT_LLM_REQUESTS` | `max_concurrent_requests` |

These variables are only read when the field is **not present in the data dict** — explicit values always take precedence.

### Multiple Read Points

Some environment variables are read in more than one location in the codebase:

- **`KARENINA_ASYNC_ENABLED`** — Read by `VerificationConfig.__init__()`, `batch_runner.run_verification_batch()`, and the parallel execution base adapter.
- **`KARENINA_ASYNC_MAX_WORKERS`** — Read by `VerificationConfig.__init__()`, `VerificationExecutor.get_max_workers()`, and the parallel execution base adapter.
- **`EMBEDDING_CHECK` / `EMBEDDING_CHECK_MODEL` / `EMBEDDING_CHECK_THRESHOLD`** — Read by both `VerificationConfig.__init__()` and the embedding check utility functions.

In all cases, the effective precedence is: explicit argument > VerificationConfig field > environment variable > built-in default.

---

## Source Locations

| Variable | Primary Read Location |
|----------|----------------------|
| `OPENAI_API_KEY` | LangChain/LiteLLM (via provider SDK) |
| `ANTHROPIC_API_KEY` | `adapters/claude_tool/`, `adapters/claude_agent_sdk/` (via dotenv) |
| `CLAUDE_CODE_OAUTH_TOKEN` | `adapters/claude_agent_sdk/auth.py`, `adapters/claude_agent_sdk/llm.py`, `adapters/claude_agent_sdk/agent.py` |
| `ANTHROPIC_AUTH_TOKEN` | `adapters/claude_agent_sdk/auth.py`, `adapters/claude_agent_sdk/llm.py`, `adapters/claude_agent_sdk/agent.py` |
| `GOOGLE_API_KEY` | LangChain/LiteLLM (via provider SDK) |
| `OPENROUTER_API_KEY` | `adapters/langchain/models.py` |
| `TAVILY_API_KEY` | `benchmark/verification/utils/search_tavily.py` |
| `EMBEDDING_CHECK` | `schemas/verification/config.py`, `benchmark/verification/utils/embedding_check.py` |
| `EMBEDDING_CHECK_MODEL` | `schemas/verification/config.py`, `benchmark/verification/utils/embedding_check.py` |
| `EMBEDDING_CHECK_THRESHOLD` | `schemas/verification/config.py`, `benchmark/verification/utils/embedding_check.py` |
| `KARENINA_ASYNC_ENABLED` | `schemas/verification/config.py`, `benchmark/verification/batch_runner.py`, `adapters/_parallel_base.py` |
| `KARENINA_ASYNC_MAX_WORKERS` | `schemas/verification/config.py`, `benchmark/verification/executor.py`, `adapters/_parallel_base.py` |
| `KARENINA_MAX_CONCURRENT_LLM_REQUESTS` | `schemas/verification/config.py` |
| `DB_PATH` | `cli/serve.py` |
| `KARENINA_PRESETS_DIR` | `schemas/verification/config_presets.py`, `cli/serve.py` |
| `MCP_PRESETS_DIR` | `cli/serve.py` |
| `AUTOSAVE_DATABASE` | `benchmark/verification/batch_runner.py` |
| `KARENINA_EXPOSE_GROUND_TRUTH` | `benchmark/verification/evaluators/template/evaluator.py` |
| `KARENINA_TRACE_TRUNCATION_THRESHOLD` | `benchmark/error_analysis/case_renderer.py`, `scenario/trace_materialization.py` |
| `CLAUDE_CONFIG_DIR` | `adapters/claude_agent_sdk/agent.py` |

---

## Variable Count Summary

| Category | Count |
|----------|-------|
| API Keys | 7 |
| Embedding Check | 3 |
| Async Execution | 3 |
| Path Configuration | 3 |
| Other Settings | 4 |
| **Total** | **20** |

---

## Related

- [Environment Variables Tutorial (section 03)](../../workflows/configuration/environment-variables.md) — grouped explanations with usage examples
- [VerificationConfig Reference](verification-config.md) — fields that shadow embedding and async env vars
- [Configuration Hierarchy (section 03)](../../workflows/configuration/index.md) — how env vars fit into precedence
- [Workspace Initialization](../../getting-started/workspace-init.md) — `karenina init` creates `.env` template
