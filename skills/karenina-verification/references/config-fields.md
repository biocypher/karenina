# Configuration Field Reference

Complete field-by-field tables for VerificationConfig and ModelConfig, sourced from:
- `karenina/src/karenina/schemas/verification/config.py`
- `karenina/src/karenina/schemas/config/models.py`

---

## VerificationConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| **`answering_models`** | `list[ModelConfig]` | `[]` | Models that generate answers. Empty only when `parsing_only=True`. |
| **`parsing_models`** | `list[ModelConfig]` | (required) | Models that parse/judge responses. At least one is always required. |
| **`replicate_count`** | `int` | `1` | Number of independent runs per question/model combination. Must be >= 1. |
| **`evaluation_mode`** | `Literal["template_only", "template_and_rubric", "rubric_only"]` | `"template_only"` | Which evaluation stages run. There is no `"auto"` mode: attaching a rubric does NOT change the mode — to score attached rubrics you must set `"template_and_rubric"` (or `"rubric_only"`). `rubric_enabled` is a computed property derived from this field. Scenario runs warn and ignore `"rubric_only"` (mode auto-detected from rubric presence). |
| `parsing_only` | `bool` | `False` | Skip answer generation; use pre-recorded outputs (for TaskEval). |
| `rubric_trait_names` | `list[str] \| None` | `None` | Optional filter: evaluate only these trait names. |
| `rubric_evaluation_strategy` | `Literal["batch", "sequential"] \| None` | `"batch"` | `"batch"`: all LLM traits in one call. `"sequential"`: one call per trait. |
| `use_full_trace_for_template` | `bool` | `False` | Pass full MCP agent trace (not just final AI message) to template parsing. |
| `use_full_trace_for_rubric` | `bool` | `True` | Pass full MCP agent trace to rubric evaluation. |
| `allow_partial_trace_scoring` | `bool` | `True` | Permit template parsing/scoring on responses truncated by answer-generation timeout. |
| `abstention_enabled` | `bool` | `False` | Enable abstention/refusal detection guard. |
| `sufficiency_enabled` | `bool` | `False` | Enable trace sufficiency detection guard. |
| `include_extraction_hints` | `bool` | `True` | Append extraction hints to the parsing prompt. |
| `embedding_check_enabled` | `bool` | `False` | Enable semantic similarity fallback verification. Env: `EMBEDDING_CHECK`. |
| `embedding_check_model` | `str` | `"all-MiniLM-L6-v2"` | SentenceTransformer model for embeddings. Env: `EMBEDDING_CHECK_MODEL`. |
| `embedding_check_threshold` | `float` | `0.85` | Similarity threshold. Constrained to [0.0, 1.0]. Env: `EMBEDDING_CHECK_THRESHOLD`. |
| `async_enabled` | `bool` | `True` | Enable parallel execution of questions. Env: `KARENINA_ASYNC_ENABLED`. |
| `async_max_workers` | `int` | `2` | Number of parallel workers. Must be >= 1. Env: `KARENINA_ASYNC_MAX_WORKERS`. |
| `max_concurrent_requests` | `int \| None` | `None` | Global cap on concurrent LLM request setups. None = bounded by `async_max_workers` only. Env: `KARENINA_MAX_CONCURRENT_LLM_REQUESTS`. |
| `batch_timeout_seconds` | `float \| None` | `None` | Wall-clock ceiling for an entire parallel batch. None disables. Sequential runs are not bounded. |
| `task_ordering` | `Literal["auto", "prefix_cache", "distribute_answerers", "generation_order", "random"]` | `"auto"` | Task queue ordering strategy. |
| `answerer_concurrency_limits` | `int \| dict[str, int] \| None` | `None` | Per-answerer concurrency caps (int applies to all; dict keyed by `ModelConfig.id`). None disables. |
| `deep_judgment_mode` | `Literal["disabled", "reasoning_only", "full"]` | `"disabled"` | Template deep-judgment mode. `"disabled"`: off. `"reasoning_only"`: reasoning only (2 LLM calls). `"full"`: excerpts + reasoning (3+ LLM calls). |
| `deep_judgment_max_excerpts_per_attribute` | `int` | `3` | Max excerpts extracted per template attribute (ignored in reasoning-only mode). |
| `deep_judgment_fuzzy_match_threshold` | `float` | `0.80` | Fuzzy match similarity threshold for excerpt validation. |
| `deep_judgment_excerpt_retry_attempts` | `int` | `2` | Retry attempts for excerpt extraction. |
| `deep_judgment_search_enabled` | `bool` | `False` | Enable search validation for deep judgment excerpts. |
| `deep_judgment_search_tool` | `str \| Callable` | `"tavily"` | Search tool: built-in name or custom callable. |
| `deep_judgment_rubric_mode` | `Literal["disabled", "enable_all", "use_checkpoint", "custom"]` | `"disabled"` | Controls deep judgment for rubric traits. |
| `deep_judgment_rubric_global_excerpts` | `bool` | `True` | Enable/disable excerpts globally in `enable_all` mode. |
| `deep_judgment_rubric_config` | `dict[str, Any] \| None` | `None` | Per-trait config for `custom` mode. |
| `deep_judgment_rubric_max_excerpts_default` | `int` | `7` | Default max excerpts per rubric trait. |
| `deep_judgment_rubric_fuzzy_match_threshold_default` | `float` | `0.80` | Default fuzzy match threshold for rubric traits. |
| `deep_judgment_rubric_excerpt_retry_attempts_default` | `int` | `2` | Default retry attempts for rubric excerpt extraction. |
| `deep_judgment_rubric_search_tool` | `str \| Callable` | `"tavily"` | Search tool for rubric hallucination detection. |
| `few_shot_config` | `FewShotConfig \| None` | `None` | Few-shot prompting configuration. |
| `prompt_config` | `PromptConfig \| None` | `None` | Per-task custom instructions injected into pipeline LLM calls. |
| `agentic_parsing` | `bool` | `False` | Enable agentic parsing (Stage 7b): judge uses tools to verify artifacts. |
| `agentic_parsing_trigger` | `Literal["always", "dynamic"]` | `"always"` | When `agentic_parsing=True`: `"always"` runs the agentic investigation every time; `"dynamic"` escalates only when a final-message parse is insufficient. |
| `agentic_judge_context` | `Literal["workspace_only", "trace_and_workspace", "trace_only"]` | `"workspace_only"` | What context the investigation agent receives. |
| `agentic_parsing_max_turns` | `int` | `15` | Max turns for the investigation agent. Must be >= 1. |
| `agentic_parsing_timeout` | `float` | `120.0` | Timeout in seconds for the investigation agent. Must be >= 0.0. |
| `agentic_parsing_materialize_trace` | `bool` | `False` | Write the answering agent trace to a file for the investigation agent. |
| `agentic_parsing_persist_trace` | `bool` | `False` | Keep the materialized trace file after extraction (ignored when materialize is False). |
| `agentic_rubric_strategy` | `Literal["individual", "shared"]` | `"individual"` | How to evaluate agentic rubric traits. |
| `agentic_rubric_parallel` | `bool` | `False` | Enable parallel evaluation of agentic rubric traits (individual strategy only). |
| `workspace_copy` | `bool` | `True` | Copy question workspaces before execution (protects originals). |
| `workspace_cleanup` | `bool` | `True` | Delete working copies after the run. |
| `workspace_output_mode` | `Literal["none", "full", "produced"]` | `"none"` | Capture runtime workspaces as sidecars. `"full"`: complete final workspace. `"produced"`: new/modified files only. |
| `workspace_output_dir` | `Path \| None` | `None` | Directory for captured workspaces when `workspace_output_mode` is not `"none"`. |
| `workspace_output_exclude_patterns` | `list[str]` | `[]` | Extra fnmatch-style exclude patterns for workspace capture. |
| `db_config` | `Any \| None` | `None` | DBConfig instance for automatic result persistence. |
| `scenario_turn_limit` | `int` | `20` | Max turns before forced termination in scenario execution. Must be >= 1. |
| `request_timeout` | `float` | `120.0` | HTTP request timeout (seconds) for all pipeline LLM calls. None disables (provider SDK defaults). |
| `retry_policy` | `RetryPolicy` | `RetryPolicy()` | Per-category retry budgets for transient LLM errors. |
| `custom_error_patterns` | `list[ErrorPatternConfig]` | `[]` | User-defined error patterns for the ErrorRegistry. |
| `max_requeue_count` | `int` | `5` | Max times a task can be requeued in the parallel executor before generating fresh. Must be >= 1. |

### Environment Variable Overrides

These fields can be set via environment variables (lowest precedence, only used if the field is not explicitly set):

| Field | Environment Variable |
|-------|---------------------|
| `embedding_check_enabled` | `EMBEDDING_CHECK` (true/1/yes) |
| `embedding_check_model` | `EMBEDDING_CHECK_MODEL` |
| `embedding_check_threshold` | `EMBEDDING_CHECK_THRESHOLD` |
| `async_enabled` | `KARENINA_ASYNC_ENABLED` (true/1/yes) |
| `async_max_workers` | `KARENINA_ASYNC_MAX_WORKERS` |
| `max_concurrent_requests` | `KARENINA_MAX_CONCURRENT_LLM_REQUESTS` |

---

## ModelConfig Fields

### Minimal Valid Examples

All karenina Pydantic models use `extra="forbid"`. Passing an undefined field raises `ValidationError`.

```python
from karenina.schemas.config import ModelConfig

# Non-manual: id, model_name, and model_provider are required
ModelConfig(id="answerer", model_name="claude-haiku-4-5", model_provider="anthropic")

# Manual: only interface and manual_traces are required
# Import ManualTraces from the adapters module (NOT from schemas.config.models)
from karenina.adapters.manual import ManualTraces
manual_traces = ManualTraces(benchmark)  # benchmark argument is required
ModelConfig(interface="manual", manual_traces=manual_traces)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| **`id`** | `str \| None` | `None` | Model identifier. Required for non-manual interfaces. Defaults to `"manual"` for manual interface. |
| **`model_name`** | `str \| None` | `None` | Model name (e.g., `"claude-sonnet-4-20250514"`). Required for non-manual interfaces. |
| **`model_provider`** | `str \| None` | `None` | Provider identifier (`"anthropic"`, `"openai"`, `"google"`, `"litellm"`). Required when adapter's `requires_provider=True`. |
| **`interface`** | `str` | `"langchain"` | Adapter interface. See adapter selection reference. |
| `temperature` | `float` | `0.1` | Sampling temperature for generation. |
| `max_tokens` | `int` | `16384` | Maximum tokens in model response. |
| `system_prompt` | `str \| None` | `None` | Custom system prompt. Answering models get a default when omitted; parsing models keep `None` unless set explicitly. |
| `max_retries` | `int` | `2` | Max retries for template generation. |
| `mcp_urls_dict` | `dict[str, str] \| None` | `None` | MCP server URLs: `{"name": "http://host:port/mcp"}`. Enables agent mode. |
| `mcp_tool_filter` | `list[str] \| None` | `None` | Restrict available MCP tools to this list. |
| `mcp_tool_description_overrides` | `dict[str, str] \| None` | `None` | Override tool descriptions for GEPA optimization. |
| `mcp_http_timeout` | `float \| None` | `None` | HTTP timeout for MCP streamable-HTTP connections. None = MCP SDK default (30s). |
| `mcp_sse_read_timeout` | `float \| None` | `None` | SSE read timeout for MCP streamable-HTTP connections. None = MCP SDK default (300s). |
| `endpoint_base_url` | `str \| None` | `None` | Custom endpoint URL (for `openai_endpoint` interface). |
| `endpoint_api_key` | `SecretStr \| None` | `None` | API key for custom endpoint (for `openai_endpoint` interface). |
| `anthropic_base_url` | `str \| None` | `None` | Custom Anthropic API endpoint (for proxies, self-hosted). |
| `anthropic_api_key` | `SecretStr \| None` | `None` | Override `ANTHROPIC_API_KEY` env var. |
| `agent_runtime` | `AgentRuntimeConfig \| None` | `None` | Typed filesystem/sandbox/container settings for agent adapters. |
| `extra_kwargs` | `dict[str, Any] \| None` | `None` | Extra keyword arguments passed to the underlying model interface. |
| `manual_traces` | `Any` | `None` | Pre-recorded traces for manual interface. Required when `interface="manual"`. Import: `from karenina.adapters.manual import ManualTraces`. Constructor: `ManualTraces(benchmark)`. Excluded from serialization. |
| `agent_middleware` | `AgentMiddlewareConfig \| None` | `None` | Middleware config for MCP-enabled agents (retry, limits, summarization, caching). |
| `max_context_tokens` | `int \| None` | `None` | Token threshold for summarization middleware. |
| `agent_timeout` | `int \| None` | `None` | Timeout in seconds for agent execution (default 180s). |
| `request_timeout` | `float \| None` | `None` | HTTP request timeout for individual LLM calls. None = provider SDK default. Typically stamped from `VerificationConfig.request_timeout`. |
| `retry_policy` | `RetryPolicy \| None` | `None` | Per-category retry policy. None = use pipeline-level `RetryPolicy`. |

### AgentMiddlewareConfig (nested on ModelConfig.agent_middleware)

| Sub-config | Key Fields | Defaults |
|-----------|------------|----------|
| `limits` (AgentLimitConfig) | `model_call_limit`, `tool_call_limit`, `exit_behavior` | 25, 50, `"end"` |
| `model_retry` (ModelRetryConfig) | `max_retries`, `backoff_factor`, `initial_delay`, `max_delay`, `jitter`, `on_failure` | 2, 2.0, 2.0, 10.0, True, `"continue"` |
| `tool_retry` (ToolRetryConfig) | `max_retries`, `backoff_factor`, `initial_delay`, `on_failure` | 3, 2.0, 1.0, `"return_message"` |
| `summarization` (SummarizationConfig) | `enabled`, `model`, `trigger_fraction`, `trigger_tokens`, `keep_messages` | True, None, 0.8, None, 20 |
| `prompt_caching` (PromptCachingConfig) | `enabled`, `ttl`, `min_messages_to_cache`, `unsupported_model_behavior` | True, `"5m"`, 0, `"warn"` |

### AgentRuntimeConfig (nested on ModelConfig.agent_runtime)

Typed execution-runtime settings for agent adapters. Backend-specific validation is left to the adapter.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | `Literal["native", "filesystem", "container", "docker", "local_shell"] \| None` | `None` | Execution backend; None lets the adapter choose. |
| `access_mode` | `Literal["read_write", "read_only"]` | `"read_write"` | Filesystem access mode. |
| `sandbox_enabled` | `bool` | `True` | Enable sandboxing. |
| `container_runtime` | `Literal["docker", "singularity", "apptainer"]` | `"docker"` | Container runtime. |
| `container_image` | `str \| None` | `None` | Container image name. |
| `container_network` | `Literal["bridge", "none"]` | `"bridge"` | Container network mode. |
| `container_add_hosts` | `tuple[str, ...]` | `()` | Extra container host mappings. |
| `read_max_bytes` | `int \| None` | `None` | Max bytes readable from filesystem (None = unlimited). |

### FewShotConfig (nested on VerificationConfig.few_shot_config)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | `Literal["disabled", "question_pool", "global", "both"]` | `"both"` | Master source switch; `"disabled"` turns off all examples. |
| `pool_mode` | `Literal["all", "k-shot", "custom"]` | `"all"` | Default mode for all questions. |
| `pool_k` | `int` | `3` | Default k for `k-shot` mode. |
| `question_configs` | `dict[str, QuestionFewShotConfig]` | `{}` | Per-question overrides. |
| `global_examples` | `list[dict[str, str]]` | `[]` | External examples available to all questions. |

### PromptConfig (nested on VerificationConfig.prompt_config)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `generation` | `str \| None` | `None` | Custom instructions for answer generation. |
| `parsing` | `str \| None` | `None` | Custom instructions for answer parsing. |
| `abstention_detection` | `str \| None` | `None` | Custom instructions for abstention detection. |
| `sufficiency_detection` | `str \| None` | `None` | Custom instructions for sufficiency detection. |
| `rubric_evaluation` | `str \| None` | `None` | Fallback instructions for all rubric tasks. |
| `agentic_parsing` | `str \| None` | `None` | Fallback instructions for agentic parsing tasks (investigation + extraction). |
| `deep_judgment` | `str \| None` | `None` | Fallback instructions for all deep judgment tasks. |
