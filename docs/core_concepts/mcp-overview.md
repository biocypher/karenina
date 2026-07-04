---
jupyter:
  jupytext:
    formats: docs/core_concepts//md,docs/notebooks/core_concepts//ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# MCP Integration

MCP (Model Context Protocol) transforms the answering model from a single-shot text generator into a multi-turn **agent** with tool access. Instead of relying solely on training data, the model can call external tools (web search, database queries, API endpoints, file operations) to gather information before producing its final response.

MCP integration is purely an **answering model concern**. It changes how the response is generated, not how the response is evaluated. The [verification pipeline](../verification-pipeline/), [answer templates](../answer-templates/), and [rubrics](../../../core_concepts/rubrics/) work identically regardless of whether the response came from a simple LLM call or a multi-turn agent session.

```python tags=["hide-cell"]
# Mock cell: ensures examples execute without live API keys.
# This cell is hidden in rendered documentation.
```

## 1. Why MCP Exists in Karenina

Standard verification sends a question to an LLM, receives a single response, and evaluates it. This works when the question can be answered from training data alone. Some evaluation scenarios, however, require the model to access external information:

- **Current information**: drug approval statuses, regulatory changes, recent publications
- **Structured databases**: genomics databases, clinical trial registries, knowledge graphs
- **Custom tools**: domain-specific calculators, internal APIs, file readers
- **Tool-use evaluation**: benchmarks that test the model's ability to choose and use tools effectively

MCP provides a standardized protocol for giving the model tool access. You configure MCP servers and Karenina handles connection, tool discovery, the agent loop, and trace capture automatically.

## 2. When to Use MCP

| Scenario | Recommendation | Why |
|----------|---------------|-----|
| Questions answerable from training data | Simple LLM (no MCP) | Faster, cheaper, more reproducible |
| Questions requiring current or real-time information | MCP with appropriate servers | Training data has a cutoff date |
| Questions needing data from specific databases | MCP with database tools | Model cannot access databases without tools |
| Benchmark tests tool-use ability itself | MCP required | The tool usage is the capability being evaluated |
| Reproducibility is the top priority | Simple LLM or [manual interface](../manual-interface/) | MCP results vary with external state |
| Cost and latency must be minimized | Simple LLM | MCP adds multiple LLM calls per question |

!!! tip "Litmus test"

    If the question includes phrases like "current status of," "latest data on," or "look up in [database]," MCP is likely appropriate. If the question asks about established knowledge ("What is the mechanism of action of aspirin?"), simple LLM is sufficient and more reproducible.

MCP verification costs more and takes longer than simple LLM verification. Each question triggers a multi-turn agent loop with multiple LLM calls and tool invocations. Use MCP only when the evaluation genuinely requires external information or when tool use is itself the capability being tested.

## 3. Architecture

MCP integration sits between the question and the [verification pipeline](../verification-pipeline/), replacing the single LLM call with an agent loop:

```
                          ┌─────────────────┐
                          │   MCP Servers    │
                          │  (external tools)│
                          └────────┬─────────┘
                                   │ tool calls & results
                                   ▼
┌──────────┐    question    ┌─────────────┐    response + trace    ┌────────────┐
│ Karenina │ ──────────────►│  AgentPort  │ ─────────────────────► │ Evaluation │
│          │                │  (adapter)  │                        │ Pipeline   │
└──────────┘                └─────────────┘                        └────────────┘
                                   │
                              multi-turn
                              agent loop
```

Three components work together:

| Component | Role | Configured via |
|-----------|------|----------------|
| **MCP Servers** | External processes that expose tools via the MCP protocol (HTTP or stdio transport) | `ModelConfig.mcp_urls_dict` |
| **AgentPort adapter** | Connects to servers, discovers tools, runs the multi-turn agent loop, captures the trace | `ModelConfig.interface` (selects the [adapter](../../../core_concepts/adapters/)) |
| **Agent middleware** | Retry logic, execution limits, conversation summarization, prompt caching | `ModelConfig.agent_middleware` |

The agent loop iterates: the model receives the question plus available tools, generates a response, optionally invokes tools, receives tool results, and continues until it produces a final answer or hits a configured limit.

### 3.1 The Abstraction Boundary

MCP changes the **generation** side of the pipeline. Everything downstream remains the same:

| Pipeline concern | With MCP | Without MCP |
|------------------|----------|-------------|
| **Response generation** | Multi-turn agent loop with tool calls | Single LLM call |
| **Trace capture** | Full conversation (messages + tool calls + results) | Single response |
| **Template parsing** | Identical: Judge LLM parses response into schema | Same |
| **Template verification** | Identical: `verify()` checks parsed values | Same |
| **Rubric evaluation** | Identical: traits assess observable response properties | Same |

The only pipeline-level difference is that two `VerificationConfig` fields control whether the full agent trace or only the final response is sent to the parsing and evaluation models (see [Section 7](#7-trace-handling)).

## 4. Configuration

MCP is configured on `ModelConfig`, not on `VerificationConfig`. Each answering model can have its own MCP server configuration:

```python
from karenina.schemas.config.models import ModelConfig

model_config = ModelConfig(
    id="agent-claude",
    model_name="claude-sonnet-4-5",
    model_provider="anthropic",
    interface="langchain",
    mcp_urls_dict={
        "biocontext": "https://mcp.biocontext.ai/mcp/",
        "web_search": "https://search-server.example.com/mcp/",
    },
)

print(f"Agent mode: {'enabled' if model_config.mcp_urls_dict else 'disabled'}")
print(f"MCP servers: {list(model_config.mcp_urls_dict.keys())}")
print(f"Tool filter: {model_config.mcp_tool_filter}")
print(f"Agent timeout: {model_config.agent_timeout}")
```

Setting `mcp_urls_dict` is the trigger that switches from simple LLM invocation to agent-based execution. All other MCP-related fields are optional refinements.

### 4.1 MCP Fields on ModelConfig

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `mcp_urls_dict` | `dict[str, str] \| None` | `None` | Map of server names to URLs. Setting this enables MCP agent mode. |
| `mcp_tool_filter` | `list[str] \| None` | `None` | Restrict which discovered tools the agent can use (by tool name). `None` means all tools are available. |
| `mcp_tool_description_overrides` | `dict[str, str] \| None` | `None` | Replace tool descriptions sent to the LLM. Useful for improving tool selection accuracy. |
| `max_context_tokens` | `int \| None` | `None` | Token threshold for triggering conversation summarization. When omitted, the adapter auto-detects from the model's context window. |
| `agent_middleware` | `AgentMiddlewareConfig \| None` | `None` | Controls retry behavior, execution limits, summarization, and prompt caching. Only used when `mcp_urls_dict` is set. |
| `agent_timeout` | `int \| None` | `None` | Timeout in seconds for agent execution. Overrides the default (180s). Set higher for complex questions with many tool calls. |

## 5. Agent Middleware

`AgentMiddlewareConfig` controls agent execution behavior and only applies when `mcp_urls_dict` is set. It groups five sub-configurations, each with sensible defaults:

```python
from karenina.schemas.config.models import (
    AgentMiddlewareConfig,
    AgentLimitConfig,
    SummarizationConfig,
)

middleware = AgentMiddlewareConfig(
    limits=AgentLimitConfig(
        model_call_limit=40,  # default: 25
        tool_call_limit=80,  # default: 50
    ),
    summarization=SummarizationConfig(
        trigger_fraction=0.7,  # default: 0.8
        keep_messages=30,  # default: 20
    ),
)

model_config = ModelConfig(
    id="agent-claude",
    model_name="claude-sonnet-4-5",
    model_provider="anthropic",
    interface="langchain",
    mcp_urls_dict={"biocontext": "https://mcp.biocontext.ai/mcp/"},
    agent_middleware=middleware,
)

print(f"Model call limit: {middleware.limits.model_call_limit}")
print(f"Tool call limit: {middleware.limits.tool_call_limit}")
print(f"Exit behavior: {middleware.limits.exit_behavior}")
print(
    f"Summarization: trigger at {middleware.summarization.trigger_fraction:.0%}, keep {middleware.summarization.keep_messages} messages"
)
print(f"Model retry: {middleware.model_retry.max_retries} retries, on_failure='{middleware.model_retry.on_failure}'")
print(f"Tool retry: {middleware.tool_retry.max_retries} retries, on_failure='{middleware.tool_retry.on_failure}'")
print(f"Prompt caching: enabled={middleware.prompt_caching.enabled}, ttl='{middleware.prompt_caching.ttl}'")
```

### 5.1 Sub-Configurations

| Component | Config Class | Key Fields | Defaults |
|-----------|-------------|------------|----------|
| **Execution limits** | `AgentLimitConfig` | `model_call_limit`, `tool_call_limit`, `exit_behavior` | 25 model calls, 50 tool calls, `"end"` |
| **Model retry** | `ModelRetryConfig` | `max_retries`, `backoff_factor`, `initial_delay`, `on_failure` | 2 retries, 2.0x backoff, 2.0s initial delay, `"continue"` |
| **Tool retry** | `ToolRetryConfig` | `max_retries`, `backoff_factor`, `initial_delay`, `on_failure` | 3 retries, 2.0x backoff, 1.0s initial delay, `"return_message"` |
| **Summarization** | `SummarizationConfig` | `enabled`, `trigger_fraction`, `keep_messages` | `True`, 0.8, 20 messages |
| **Prompt caching** | `PromptCachingConfig` | `enabled`, `ttl`, `unsupported_model_behavior` | `True`, `"5m"`, `"warn"` |

**Execution limits** prevent runaway agents. When `model_call_limit` or `tool_call_limit` is reached, the `exit_behavior` field determines whether the agent returns a partial response (`"end"`) or blocks further calls but continues (`"continue"`). The pipeline's RecursionLimitAutoFail stage (Stage 3) auto-fails verification if a limit was hit.

**Summarization** condenses conversation history when the token count approaches the context window, preventing failures on long multi-turn interactions. Only the `langchain` adapter exposes the full `SummarizationConfig`; the `claude_agent_sdk` and `claude_tool` adapters use built-in summarization.

**Prompt caching** reduces cost and latency for Anthropic models by caching static content (system prompts, tool definitions) on Anthropic's servers. Only applies to Anthropic models with the `langchain` interface. The `ttl` field accepts `"5m"` or `"1h"`.

## 6. Adapter-Specific MCP Behavior

Each [adapter](../../../core_concepts/adapters/) implements MCP differently. The adapter is selected by the `interface` field on `ModelConfig`.

| Feature | `langchain` | `claude_agent_sdk` | `claude_tool` |
|---------|:-----------:|:------------------:|:-------------:|
| **Transport** | HTTP/SSE | HTTP/SSE + stdio | HTTP/SSE |
| **MCP library** | `langchain-mcp-adapters` | Claude Agent SDK | `mcp` Python SDK |
| **Tool filtering** | Yes | Yes | Yes |
| **Description overrides** | Yes | Yes | Yes |
| **Configurable middleware** | Full (`AgentMiddlewareConfig`) | Limits only | Limits only |
| **Prompt caching** | Middleware-based (Anthropic only) | No | Native (`cache_control`) |
| **Summarization** | Configurable | Built-in (SDK-managed) | Built-in |
| **Stdio servers** | No | Yes | No |

### 6.1 Choosing an Adapter for MCP

| Scenario | Adapter | Why |
|----------|---------|-----|
| General-purpose MCP with configurable middleware | `langchain` | Full control over retry, summarization, caching behavior |
| Need local stdio-based MCP servers | `claude_agent_sdk` | Only adapter supporting stdio transport |
| Direct Anthropic API without framework overhead | `claude_tool` | Uses Anthropic SDK directly with native prompt caching |
| Non-Anthropic models (OpenAI, Google) with MCP | `langchain` | Multi-provider support via LangChain |
| Pre-recorded traces, no live tools | `manual` | Does not support MCP; raises `ValueError` if `mcp_urls_dict` is set |

The `openrouter` and `openai_endpoint` interfaces delegate to the `langchain` adapter internally, so they inherit the same MCP behavior.

For implementation details on how each adapter connects, discovers tools, runs the agent loop, and captures traces, see the [MCP Integration Deep Dive](../../../advanced-adapters/mcp-integration/).

## 7. Trace Handling

After agent execution, the full conversation trace (assistant responses, tool calls, tool results) is captured in two formats:

- **`trace_messages`**: structured list of `Message` objects with typed content blocks (text, tool use, tool result, thinking)
- **`raw_trace`**: legacy string serialization for backward compatibility and debugging

Two `VerificationConfig` fields control what the downstream evaluation models receive:

```python
from karenina.schemas.verification.config import VerificationConfig

# Minimal config to demonstrate trace handling fields
judge = ModelConfig(id="judge", model_name="claude-haiku-4-5", model_provider="anthropic", interface="langchain")
config = VerificationConfig(
    answering_models=[model_config],  # MCP-enabled model from Section 5
    parsing_models=[judge],
    use_full_trace_for_template=False,  # default: only final AI message for parsing
    use_full_trace_for_rubric=True,  # default: full trace for rubric evaluation
)

print(f"Template sees full trace: {config.use_full_trace_for_template}")
print(f"Rubric sees full trace: {config.use_full_trace_for_rubric}")
```

| Field | Type | Default | Effect |
|-------|------|---------|--------|
| `use_full_trace_for_template` | `bool` | `False` | `True`: template parsing and the abstention/sufficiency checks see the full trace. `False`: they see only the final AI message. |
| `use_full_trace_for_rubric` | `bool` | `True` | `True`: rubric evaluation sees the complete trace. `False`: only the final AI message is evaluated. |

!!! note

    The full trace is **always captured and stored** regardless of these settings. These flags only control what input the parsing and evaluation models receive. If set to `False` and the trace does not end with an AI message, the corresponding verification stage will fail.

The defaults reflect typical evaluation patterns: template parsing usually needs only the final answer (the model's conclusion after tool use), while rubric evaluation benefits from the full trace (to assess qualities like tool selection, reasoning process, or citation of retrieved information). Keep `use_full_trace_for_template=False` on long MCP runs: full traces can easily blow past a judge's context window.

## 8. MCP Execution Lifecycle

When verification runs with MCP enabled, the following lifecycle executes per question:

```
1. Connect       Adapter connects to MCP servers listed in mcp_urls_dict
       │
2. Discover      Available tools are fetched from each server
       │
3. Filter        mcp_tool_filter restricts the tool set;
       │         description overrides are applied
       │
4. Agent loop    LLM receives question + tools → generates response →
       │         calls tools → receives results → continues until
       │         final answer or limit reached
       │
5. Middleware     Retry handles transient failures; summarization
       │         condenses long conversations; limits cap execution
       │
6. Capture       All messages captured as trace_messages and raw_trace
       │
7. Return        AgentResult with final_response, traces, usage metadata,
                 and limit_reached flag
```

The `AgentResult` feeds into the standard [verification pipeline](../verification-pipeline/) at the GenerateAnswer stage (Stage 2). If `limit_reached` is `True`, the RecursionLimitAutoFail stage (Stage 3) marks verification as auto-failed while preserving the captured trace for inspection.

## 9. Next Steps

1. [MCP-enabled verification workflow](../../running-verification/mcp-agent-evaluation/): step-by-step configuration and execution
2. [MCP Integration Deep Dive](../../../advanced-adapters/mcp-integration/): adapter internals, connection lifecycle, trace capture, message conversion
3. [Adapters](../../../core_concepts/adapters/): adapter comparison and port/adapter architecture
4. [Evaluation modes](../evaluation-modes/): how MCP interacts with template and rubric evaluation
5. [Manual interface](../manual-interface/): alternative for reproducible testing without live tools
