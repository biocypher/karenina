---
jupyter:
  jupytext:
    formats: docs/workflows/running-verification//md,docs/notebooks/running-verification//ipynb
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

# MCP Agent Evaluation

This scenario evaluates tool-using agents that interact with MCP (Model Context Protocol) servers. You configure MCP tools on the answering model, set agent middleware parameters, and handle traces that include tool calls alongside natural language responses.

**What you'll learn:**

- Configure MCP tools for the answering model
- Set agent middleware parameters (limits, retries, summarization)
- Handle traces that contain tool calls
- Understand recursion limits and how to adjust them
- Compare adapter options for MCP evaluation

```python tags=["hide-cell"]
# Setup cell: creates mock results with MCP metadata.
# This cell is hidden in the rendered documentation.
import datetime
import tempfile
from pathlib import Path

from karenina import Benchmark
from karenina.schemas.config import ModelConfig
from karenina.schemas.results import VerificationResultSet
from karenina.schemas.verification import VerificationConfig, VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultTemplate,
)

_benchmark = Benchmark.create(
    name="Research Agent QA",
    description="Evaluating MCP-enabled research agents",
    version="1.0.0",
)
_questions = [
    ("Find the current population of Tokyo and compare it to New York City.", "Tokyo: ~14M, NYC: ~8.3M"),
    ("What is the latest published unemployment rate in the EU?", "6.0% (Eurostat, 2025)"),
    ("Search for recent clinical trials on GLP-1 receptor agonists for obesity.", "Multiple Phase 3 trials ongoing"),
    ("Look up the atomic mass of oganesson and verify it.", "294 u"),
    ("Find the current exchange rate of EUR to USD.", "Approximately 1.08"),
]
for q, a in _questions:
    _benchmark.add_question(question=q, raw_answer=a)

_tmp = Path(tempfile.mkdtemp()) / "benchmark.jsonld"
_benchmark.save(str(_tmp))

_answering = ModelIdentity(model_name="claude-sonnet-4-20250514", interface="claude_agent_sdk", tools=["brave_search"])
_parsing = ModelIdentity(model_name="claude-haiku-4-5", interface="langchain")
_ts = datetime.datetime.now(tz=datetime.UTC).isoformat()
_qids = _benchmark.get_question_ids()

_mcp_data = [
    {"verified": True, "recursion_hit": False, "iterations": 3, "tool_calls": 4, "tools": ["brave_search"], "suspect_failed": 0},
    {"verified": True, "recursion_hit": False, "iterations": 2, "tool_calls": 2, "tools": ["brave_search"], "suspect_failed": 0},
    {"verified": True, "recursion_hit": False, "iterations": 5, "tool_calls": 7, "tools": ["brave_search", "read_resource"], "suspect_failed": 1},
    {"verified": False, "recursion_hit": True, "iterations": 10, "tool_calls": 12, "tools": ["brave_search"], "suspect_failed": 3},
    {"verified": True, "recursion_hit": False, "iterations": 2, "tool_calls": 3, "tools": ["brave_search"], "suspect_failed": 0},
]

_trace = [
    {"role": "human", "content": "Question here"},
    {"role": "ai", "content": "Let me search for that."},
    {"role": "tool", "content": "Search results..."},
    {"role": "ai", "content": "Based on my research, the answer is..."},
]


def _make(qid, q_text, raw_ans, mcp):
    rid = VerificationResultMetadata.compute_result_id(qid, _answering, _parsing, _ts)
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=qid, template_id="tmpl_" + qid[:8],
            failure=None, caveats=[], question_text=q_text,
            raw_answer=raw_ans, answering=_answering, parsing=_parsing,
            execution_time=8.0, timestamp=_ts, result_id=rid,
        ),
        template=VerificationResultTemplate(
            raw_llm_response=f"Based on my research, {raw_ans.lower()}.",
            verify_result=mcp["verified"], template_verification_performed=True,
            parsed_gt_response={"answer": raw_ans},
            parsed_llm_response={"answer": raw_ans if mcp["verified"] else "unable to determine"},
            recursion_limit_reached=mcp["recursion_hit"],
            trace_messages=_trace,
            answering_mcp_servers=["brave_search"],
            agent_metrics={
                "iterations": mcp["iterations"],
                "tool_calls": mcp["tool_calls"],
                "tools_used": mcp["tools"],
                "suspect_failed_tool_calls": mcp["suspect_failed"],
                "suspect_failed_tools": ["brave_search"] if mcp["suspect_failed"] else [],
            },
        ),
        evaluation_input="Based on my research, " + raw_ans.lower() + ".",
        used_full_trace=False,
    )


_mock_results = [_make(qid, q, a, mcp) for qid, (q, a), mcp in zip(_qids, _questions, _mcp_data)]
_mock_result_set = VerificationResultSet(results=_mock_results)
_orig_run = Benchmark.run_verification
Benchmark.run_verification = lambda self, config, **kw: _mock_result_set
```

---

## When to Use MCP

MCP evaluation is needed when the answering model should use external tools:

- **Web search** — fact-checking, current events, real-time data
- **Database queries** — SQL execution, knowledge base lookup
- **API access** — external service integration
- **File operations** — reading documents, code analysis

---

## Configure MCP Tools

Attach MCP servers to the answering model via `ModelConfig`:

```python
from karenina import Benchmark
from karenina.schemas.verification import VerificationConfig
from karenina.schemas.config import ModelConfig

benchmark = Benchmark.load(str(_tmp))

config = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="claude-with-search",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="claude_agent_sdk",
            # MCP server configuration
            mcp_urls_dict={"brave_search": "http://localhost:3001/sse"},
            mcp_tool_filter=["brave_web_search", "brave_local_search"],
        )
    ],
    parsing_models=[
        ModelConfig(id="haiku-parser", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain",
                    temperature=0.0)
    ],
    evaluation_mode="template_only",
)

print(f"MCP servers: {config.answering_models[0].mcp_urls_dict}")
print(f"Tool filter: {config.answering_models[0].mcp_tool_filter}")
```

| Field | Description |
|-------|-------------|
| `mcp_urls_dict` | Map of server name → SSE endpoint URL |
| `mcp_tool_filter` | Restrict which tools the model can use (empty = all) |
| `mcp_tool_description_overrides` | Override tool descriptions for better prompting |

---

## Agent Middleware

`AgentMiddlewareConfig` controls agent execution behavior — limits, retries, and summarization:

```python
from karenina.schemas.config import ModelConfig
from karenina.schemas.config.models import AgentMiddlewareConfig, AgentLimitConfig

model = ModelConfig(
    id="claude-with-search",
    model_name="claude-sonnet-4-20250514",
    model_provider="anthropic",
    interface="claude_agent_sdk",
    mcp_urls_dict={"brave_search": "http://localhost:3001/sse"},
    # Agent middleware settings
    agent_middleware=AgentMiddlewareConfig(
        limits=AgentLimitConfig(
            model_call_limit=15,
            tool_call_limit=30,
        ),
    ),
)

print(f"Model call limit: {model.agent_middleware.limits.model_call_limit}")
print(f"Tool call limit:  {model.agent_middleware.limits.tool_call_limit}")
```

| Setting | Default | Description |
|---------|---------|-------------|
| `limits.model_call_limit` | `25` | Maximum LLM calls per agent invocation |
| `limits.tool_call_limit` | `50` | Maximum tool calls per agent invocation |
| `limits.exit_behavior` | `"end"` | Behavior when limit reached: `"end"` or `"continue"` |

---

## Run MCP-Enabled Verification

```python
results = benchmark.run_verification(config)
print(f"Total results: {len(results)}")
```

---

## Trace Handling

MCP agents produce multi-turn traces (human → AI → tool → AI → ...). By default, only the **final AI message** is passed to **template** evaluation, while **rubric** evaluation uses the **full trace**. You can configure both independently:

```python
config_full_trace = VerificationConfig(
    answering_models=[
        ModelConfig(
            id="claude-with-search",
            model_name="claude-sonnet-4-20250514",
            model_provider="anthropic",
            interface="claude_agent_sdk",
            mcp_urls_dict={"brave_search": "http://localhost:3001/sse"},
        )
    ],
    parsing_models=[
        ModelConfig(id="haiku-parser", model_name="claude-haiku-4-5",
                    model_provider="anthropic", interface="langchain",
                    temperature=0.0)
    ],
    # Pass full trace to evaluation (includes tool calls)
    use_full_trace_for_template=True,
    use_full_trace_for_rubric=True,
)

print(f"Full trace for template: {config_full_trace.use_full_trace_for_template}")
print(f"Full trace for rubric:   {config_full_trace.use_full_trace_for_rubric}")
```

| Setting | Default | When to change |
|---------|---------|---------------|
| `use_full_trace_for_template` | `False` | Set `True` if template needs to see tool call context |
| `use_full_trace_for_rubric` | `True` | Set `False` if rubric only needs the final answer |

Inspect what was passed to evaluation:

```python
for result in results[:2]:
    print(f"Q: {result.metadata.question_text[:40]}")
    print(f"  Used full trace: {result.used_full_trace}")
    if result.evaluation_input:
        print(f"  Evaluation input: {result.evaluation_input[:60]}...")
```

---

## Recursion Limits

When an agent exhausts its call budget or hits the framework's recursion sentinel, the recursion-limit autofail stage marks the result as failed. Budgets live on `ModelConfig`: `agent_timeout` (wall-clock seconds) and `AgentMiddlewareConfig.limits = AgentLimitConfig(model_call_limit=..., tool_call_limit=...)`. The autofail stage triggers on the recursion sentinel returned by the adapter (e.g. when LangChain's `recursion_limit` is hit), not on a single config field by that name:

```python
for result in results:
    t = result.template
    if t and t.recursion_limit_reached:
        print(f"RECURSION LIMIT: {result.metadata.question_text[:50]}")
        if t.agent_metrics:
            print(f"  Iterations: {t.agent_metrics['iterations']}")
            print(f"  Tool calls: {t.agent_metrics['tool_calls']}")
            print(f"  Tools used: {t.agent_metrics['tools_used']}")
```

### Agent Metrics

All MCP results include agent execution metrics:

```python
for result in results[:3]:
    t = result.template
    if t and t.agent_metrics:
        m = t.agent_metrics
        print(f"Q: {result.metadata.question_text[:40]}")
        print(f"  Iterations: {m['iterations']}, Tool calls: {m['tool_calls']}")
        print(f"  Tools: {m['tools_used']}")
        if m.get('suspect_failed_tool_calls', 0) > 0:
            print(f"  Suspect failures: {m['suspect_failed_tool_calls']} ({m['suspect_failed_tools']})")
```

---

## Adapter Comparison

Three adapters support MCP/agent workflows:

| Adapter | Interface | MCP Support | Best For |
|---------|-----------|-------------|----------|
| LangChain Agent | `langchain` | Via LangChain tools | OpenAI/Google models with tools |
| Claude Agent SDK | `claude_agent_sdk` | Native MCP | Claude models with MCP servers |
| Claude Tool | `claude_tool` | Via tool_use | Claude models with structured tool calling |

Choose the adapter via the `interface` field on `ModelConfig`. Claude Agent SDK provides the most direct MCP integration.

---

## CLI Equivalent

```python
# MCP verification via CLI:
# karenina verify benchmark.jsonld --preset mcp-preset.json \
#   --interface claude_agent_sdk

# The preset file should contain mcp_urls_dict and tool configuration.
# CLI does not support inline MCP configuration — use a preset.

print("CLI: karenina verify ... --preset mcp-preset.json --interface claude_agent_sdk")
```

---

## Related Pages

- [Basic Verification](basic-verification.ipynb) — Non-MCP verification walkthrough
- [Deep Judgment](deep-judgment.ipynb) — Add deep judgment to MCP results
- [Adapters](../../core_concepts/adapters.md) — Full adapter comparison
- [VerificationConfig Reference](../../reference/configuration/verification-config.md) — Trace handling fields

```python tags=["hide-cell"]
# Cleanup
Benchmark.run_verification = _orig_run
```
