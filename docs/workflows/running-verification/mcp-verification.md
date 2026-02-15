# MCP-Enabled Verification

Model Context Protocol (MCP) allows the answering model to use external tools
during verification — web search, database queries, file operations, or any
custom tool exposed by an MCP server. This turns the answering model into an
**agent** that can gather information before producing its final response.

## When to Use MCP

MCP verification is useful when benchmark questions require:

- **Current information** — search the web for recent data
- **External data access** — query databases or APIs
- **File operations** — read documents or configuration files
- **Custom tools** — domain-specific tools for specialized benchmarks

Without MCP, the answering model can only rely on its training data. With MCP,
it can access live information through tool calls.

## Tool Configuration

MCP tools are configured on `ModelConfig`, not on `VerificationConfig`. This
means each answering model can connect to different MCP servers:

```python
from karenina.schemas import ModelConfig

model_config = ModelConfig(
    id="agent-gpt4o",
    model_name="gpt-4o",
    model_provider="openai",
    interface="langchain",
    mcp_urls_dict={
        "search": "http://localhost:3000/mcp",
        "database": "http://localhost:3001/mcp",
    },
)
```

### MCP Fields on ModelConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mcp_urls_dict` | `dict[str, str] \| None` | `None` | Mapping of server names to MCP endpoint URLs. Setting this activates agent mode. |
| `mcp_tool_filter` | `list[str] \| None` | `None` | Restrict which tools are available. Only tools whose names match this list are used. |
| `mcp_tool_description_overrides` | `dict[str, str] \| None` | `None` | Override tool descriptions sent to the model. Useful for improving tool selection. |
| `max_context_tokens` | `int \| None` | `None` | Maximum context window size. Used to trigger summarization when approaching the limit. Auto-detected from the model when not set. |

### Filtering Tools

If an MCP server exposes many tools but your benchmark only needs a few, use
`mcp_tool_filter` to restrict the set:

```python
model_config = ModelConfig(
    id="agent-claude",
    model_name="claude-sonnet-4-5-20250929",
    model_provider="anthropic",
    interface="langchain",
    mcp_urls_dict={"biotools": "http://localhost:3000/mcp"},
    mcp_tool_filter=["web_search", "query_gene_db"],
)
```

### Overriding Tool Descriptions

Tool descriptions from the MCP server may be generic. Override them to give the
model better context:

```python
model_config = ModelConfig(
    id="agent-claude",
    model_name="claude-sonnet-4-5-20250929",
    model_provider="anthropic",
    interface="langchain",
    mcp_urls_dict={"tools": "http://localhost:3000/mcp"},
    mcp_tool_description_overrides={
        "web_search": "Search the web for current biomedical research papers and FDA drug approvals.",
        "query_db": "Query the internal genomics database for gene function and chromosome location.",
    },
)
```

## Agent Middleware Settings

When MCP is enabled, the answering model runs as an agent — it makes multiple
LLM calls in a loop, invoking tools between calls. `AgentMiddlewareConfig`
controls the safety and performance of this loop:

```python
from karenina.schemas import ModelConfig
from karenina.schemas.config.models import AgentMiddlewareConfig

model_config = ModelConfig(
    id="agent-claude",
    model_name="claude-sonnet-4-5-20250929",
    model_provider="anthropic",
    interface="langchain",
    mcp_urls_dict={"search": "http://localhost:3000/mcp"},
    agent_middleware=AgentMiddlewareConfig(
        # All sub-configs have sensible defaults — only override what you need
    ),
)
```

`AgentMiddlewareConfig` only applies when `mcp_urls_dict` is set. It contains
five sub-configurations:

### Execution Limits (`AgentLimitConfig`)

Prevents runaway agent loops by capping the number of calls:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_call_limit` | `int` | `25` | Maximum LLM calls per verification question |
| `tool_call_limit` | `int` | `50` | Maximum tool calls per verification question |
| `exit_behavior` | `"end" \| "continue"` | `"end"` | `"end"` returns partial response gracefully; `"continue"` blocks exceeded calls but lets the agent finish |

### Model Retry (`ModelRetryConfig`)

Retries failed LLM calls with exponential backoff:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | `int` | `2` | Maximum retry attempts (total calls = retries + 1) |
| `backoff_factor` | `float` | `2.0` | Multiplier for exponential backoff |
| `initial_delay` | `float` | `2.0` | Initial delay in seconds before first retry |
| `max_delay` | `float` | `10.0` | Maximum delay between retries |
| `jitter` | `bool` | `True` | Add random jitter to retry delays |
| `on_failure` | `"continue" \| "raise"` | `"continue"` | `"continue"` returns partial response; `"raise"` raises exception |

### Tool Retry (`ToolRetryConfig`)

Retries failed tool calls (e.g., MCP server timeouts):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | `int` | `3` | Maximum retry attempts for tool calls |
| `backoff_factor` | `float` | `2.0` | Multiplier for exponential backoff |
| `initial_delay` | `float` | `1.0` | Initial delay in seconds before first retry |
| `on_failure` | `"return_message" \| "raise"` | `"return_message"` | `"return_message"` returns error as message to model; `"raise"` raises exception |

### Summarization (`SummarizationConfig`)

Automatically summarizes conversation history when approaching the context
window limit. Prevents out-of-context failures during long agent runs:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable automatic summarization (default for MCP agents) |
| `model` | `str \| None` | `None` | Model for summarization (defaults to a lightweight model) |
| `trigger_fraction` | `float` | `0.8` | Fraction of context window that triggers summarization (0.0–1.0) |
| `trigger_tokens` | `int \| None` | `None` | Token count that triggers summarization (overrides `trigger_fraction` if set) |
| `keep_messages` | `int` | `20` | Number of recent messages to preserve after summarization |

!!! note
    Summarization is specific to the **LangChain adapter**. The Claude Agent SDK
    and Claude Tool adapters handle context management through their own
    mechanisms.

### Prompt Caching (`PromptCachingConfig`)

Caches static prompt content (system prompts, tool definitions) on Anthropic's
servers to reduce costs and latency:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable prompt caching (default for Anthropic models) |
| `ttl` | `"5m" \| "1h"` | `"5m"` | Cache time-to-live |
| `min_messages_to_cache` | `int` | `0` | Minimum messages before caching starts |
| `unsupported_model_behavior` | `"ignore" \| "warn" \| "raise"` | `"warn"` | Behavior when using non-Anthropic models |

!!! note
    Prompt caching only applies to **Anthropic models** via the **LangChain**
    interface. The Claude Tool adapter handles prompt caching natively through
    its own `cache_control` mechanism.

## Adapter-Specific MCP Behavior

Each adapter implements MCP support differently. All three MCP-capable adapters
produce the same output format (unified `Message` traces), but their internal
mechanisms differ:

| Feature | LangChain | Claude Agent SDK | Claude Tool |
|---------|-----------|------------------|-------------|
| **MCP library** | `langchain-mcp-adapters` | SDK built-in MCP | Native MCP SDK (`mcp` package) |
| **Tool format** | LangChain `Tool` objects | SDK tool names (`mcp__server__tool`) | Wrapped `@beta_async_tool` |
| **Agent execution** | LangGraph agent + middleware stack | `ClaudeSDKClient` agent loop | `tool_runner()` multi-turn loop |
| **Summarization** | Configurable middleware | Handled by SDK | Not applicable |
| **Prompt caching** | Via middleware (`PromptCachingConfig`) | Handled by SDK | Native `cache_control` on tools |
| **Recursion limit** | `GraphRecursionError` + checkpoint recovery | Turn limit exceptions | Implicit in `tool_runner()` |
| **Session model** | Persistent client (reused across calls) | Per-run client instance | Session-per-run with cleanup |

### LangChain Adapter

The LangChain adapter builds a full middleware stack around MCP tools:

- Fetches tools via `langchain-mcp-adapters` using `streamable_http` transport
- Creates a LangGraph agent with configurable middleware (retry, summarization, prompt caching)
- Uses an `InMemorySaver` checkpointer to recover partial state on recursion limit errors
- Best choice for **multi-provider** MCP verification (works with OpenAI, Anthropic, Google models)

### Claude Agent SDK Adapter

The Claude Agent SDK adapter uses Anthropic's native agent infrastructure:

- Converts MCP URL dict to SDK's native MCP format (`{"type": "http", "url": "..."}`)
- Manages MCP connections through the SDK's built-in lifecycle
- When MCP servers are configured, restricts tools to only MCP tools (filters out default tools)
- Best choice for **Anthropic-native** workflows with minimal configuration

### Claude Tool Adapter

The Claude Tool adapter uses the Anthropic SDK directly with tool use:

- Creates fresh MCP sessions per run using `streamablehttp_client()` from the `mcp` package
- Wraps MCP tools as `@beta_async_tool` decorated functions for the Anthropic SDK
- Applies prompt caching natively via `cache_control` on tool definitions
- Manages session cleanup through `AsyncExitStack`
- Best choice for **direct Anthropic API** access with fine-grained control

## Trace Handling

When MCP is enabled, the answering model produces an **agent trace** — a
sequence of messages including LLM responses, tool calls, and tool results.
Karenina captures the full trace and lets you control what portion is passed to
the evaluation stages.

### Trace Input Options

Two flags on `VerificationConfig` control what the parsing and rubric evaluation
models see:

| Field | Type | Default | Controls |
|-------|------|---------|----------|
| `use_full_trace_for_template` | `bool` | `False` | What the **parsing model** receives for template evaluation |
| `use_full_trace_for_rubric` | `bool` | `True` | What the **rubric evaluation model** receives for rubric assessment |

```python
from karenina.schemas import VerificationConfig, ModelConfig

config = VerificationConfig(
    answering_models=[answering_model],
    parsing_models=[parsing_model],
    # Template parsing: only see the final answer (default)
    use_full_trace_for_template=False,
    # Rubric evaluation: see the full agent trace (default)
    use_full_trace_for_rubric=True,
)
```

**Why the different defaults?**

- **Template parsing** (`False`): The parsing model only needs the final answer
  to extract structured data. Sending the full trace would add noise and
  increase token costs.
- **Rubric evaluation** (`True`): Rubric traits often assess *how* the model
  arrived at its answer — tool usage, reasoning process, thoroughness. The full
  trace provides this context.

!!! note
    The full trace is **always** captured and stored in `raw_llm_response`
    regardless of these settings. These flags only control what input is provided
    to the evaluation models. You can always inspect the complete trace in
    results.

### Trace Validation

The pipeline includes a **trace validation auto-fail stage** (stage 4) that
checks whether agent traces end with an AI message. This runs automatically for
MCP-enabled verifications:

- If the trace ends correctly with an AI message, verification continues
- If the trace does not end with an AI message, verification is auto-failed
- Manual interface traces skip this validation (user-provided traces are trusted)

## Recursion Limits and Auto-Fail

Agent loops can potentially run indefinitely. Karenina prevents this through
execution limits and auto-fail behavior.

### How Limits Work

The `model_call_limit` and `tool_call_limit` in `AgentLimitConfig` cap the
number of calls per verification question. When a limit is reached:

1. The agent stops execution and returns whatever partial response it has
2. The pipeline's **recursion limit auto-fail stage** (stage 3) detects this
3. `verify_result` is set to `False` — the question is marked as failed
4. The trace and token usage are **preserved** for analysis
5. `completed_without_errors` remains `True` (the pipeline itself didn't error)

### Adjusting Limits

For benchmarks where questions require many tool calls (e.g., multi-step
research tasks), increase the limits:

```python
from karenina.schemas.config.models import AgentMiddlewareConfig, AgentLimitConfig

model_config = ModelConfig(
    id="agent-gpt4o",
    model_name="gpt-4o",
    model_provider="openai",
    interface="langchain",
    mcp_urls_dict={"research": "http://localhost:3000/mcp"},
    agent_middleware=AgentMiddlewareConfig(
        limits=AgentLimitConfig(
            model_call_limit=50,   # Allow more LLM calls
            tool_call_limit=100,   # Allow more tool calls
        ),
    ),
)
```

## End-to-End Example

This example shows a complete MCP-enabled verification workflow. It assumes an
MCP server is running at `http://localhost:3000/mcp` with a `web_search` tool.

```python
from karenina import Benchmark
from karenina.schemas import ModelConfig, VerificationConfig
from karenina.schemas.config.models import (
    AgentMiddlewareConfig,
    AgentLimitConfig,
    ModelRetryConfig,
)

# Load or create a benchmark
benchmark = Benchmark.load("my_benchmark.jsonld")

# Configure the answering model with MCP tools
answering_model = ModelConfig(
    id="agent-claude",
    model_name="claude-sonnet-4-5-20250929",
    model_provider="anthropic",
    interface="langchain",
    mcp_urls_dict={
        "search": "http://localhost:3000/mcp",
    },
    mcp_tool_filter=["web_search"],
    agent_middleware=AgentMiddlewareConfig(
        limits=AgentLimitConfig(model_call_limit=30, tool_call_limit=60),
        model_retry=ModelRetryConfig(max_retries=3),
    ),
)

# Configure the parsing model (no MCP needed for parsing)
parsing_model = ModelConfig(
    id="parser-gpt4o",
    model_name="gpt-4o",
    model_provider="openai",
    interface="langchain",
)

# Configure verification
config = VerificationConfig(
    answering_models=[answering_model],
    parsing_models=[parsing_model],
    use_full_trace_for_template=False,  # Parse only the final answer
    use_full_trace_for_rubric=True,     # Evaluate rubrics on the full trace
)

# Run verification — the answering model will use MCP tools
results = benchmark.run_verification(config)

# Inspect results
summary = results.get_summary()
print(f"Total: {summary['total']}, Passed: {summary['passed']}")

# Check for recursion limit hits
for result in results:
    if result.template and result.template.recursion_limit_reached:
        print(f"Question {result.metadata.question_id}: hit recursion limit")
```

## Related Pages

- [Running Verification](index.md) — verification workflow overview
- [Python API](python-api.md) — full Python API walkthrough
- [Multi-Model Evaluation](multi-model.md) — comparing models
- [Evaluation Modes](../core_concepts/evaluation-modes.md) — template, rubric, and combined modes
- [Verification Config Reference](../reference/configuration/verification-config.md) — all `VerificationConfig` fields
