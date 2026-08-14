# Guard Stages Reference

Guards are pipeline stages that auto-fail a question BEFORE template parsing or rubric evaluation. When a guard fires, no structured output is produced for that question, and subsequent evaluation stages are skipped. Guards exist to prevent wasted LLM calls on responses that cannot yield valid evaluations.

**Pipeline order of guards** (assembled from config — not a fixed stage count):

```
validate_template -> generate_answer -> recursion_limit_autofail -> trace_validation_autofail -> placeholder_retry_autofail -> [abstention_check] -> [sufficiency_check] -> parse_template -> ...
```

---

## 1. Recursion Limit Auto-Fail

**Stage name**: `RecursionLimitAutoFail`

**What triggers it**: The answering agent hit its maximum recursion depth during answer generation. This happens with MCP-enabled agents that enter infinite tool-calling loops.

**When it runs**: Always, when the `recursion_limit_reached` artifact is `True` (set by the answer generation stage).

**Auto-fail result**:
- `result.template.verify_result`: `False`
- `result.template.recursion_limit_reached`: `True`
- `result.metadata.failure`: `Failure(category="recursion_limit", stage="RecursionLimitAutoFail", reason=...)`
- Trace and token usage are preserved for analysis

**How to configure**: This guard cannot be disabled. Control the recursion threshold through `AgentLimitConfig` on the answering model's `agent_middleware`:

```python
from karenina.schemas.config import ModelConfig, AgentMiddlewareConfig, AgentLimitConfig

ModelConfig(
    id="agent",
    model_name="claude-sonnet-4-20250514",
    model_provider="anthropic",
    mcp_urls_dict={"tools": "http://localhost:8080/mcp"},
    agent_middleware=AgentMiddlewareConfig(
        limits=AgentLimitConfig(
            model_call_limit=25,   # Max LLM calls
            tool_call_limit=50,    # Max tool calls
        ),
    ),
)
```

**Debugging tip**: If many questions hit recursion limits, the agent may be stuck in a loop. Check the raw trace (`result.template.raw_llm_response`) to see the tool-calling pattern.

---

## 2. Trace Validation Auto-Fail

**Stage name**: `TraceValidationAutoFail`

**What triggers it**: An MCP agent trace does not end with a valid AI message. This indicates the agent was interrupted, timed out, or produced a malformed response.

**When it runs**: Always, after answer generation, but only applies to MCP-enabled responses (where `mcp_urls_dict` is set on the answering model). Regular LLM responses and manual traces skip this check.

**Auto-fail result**:
- `result.template.verify_result`: `False`
- `result.trace_extraction_error`: Error message describing what went wrong (root-level field on the result)
- `result.metadata.failure`: `Failure(category="trace_validation", stage="TraceValidationAutoFail", reason=...)`
- Internal pipeline artifacts (not exposed on the result object): `trace_validation_failed=True`, `trace_validation_error="..."`, `mcp_enabled=True`

**How to configure**: This guard cannot be disabled. It protects against evaluating incomplete agent traces that would cause downstream parsing errors.

**Debugging tip**: If you see trace validation failures, check the raw trace for truncation. Common causes: agent timeout, network errors during tool calls, or the agent returning only tool call results without a final summary message. Increasing `agent_timeout` on ModelConfig may help for complex tasks.

---

## 3. Placeholder Retry Auto-Fail

**Stage name**: `PlaceholderRetryAutoFail`

**What triggers it**: The agent trace ends with a LangChain `ModelRetryMiddleware` exhaustion placeholder — the LAST assistant message has text content beginning with `"Model call failed after "`. This occurs when retries are exhausted with `on_failure="continue"`, swallowing a connection/infrastructure failure instead of producing a real final answer.

**When it runs**: Always, after `TraceValidationAutoFail`, before the optional checks. Only applies when the trace's last assistant message matches the placeholder fingerprint.

**Auto-fail result**:
- `result.template.verify_result`: `False`
- `result.metadata.failure`: `Failure(category="connection", stage="PlaceholderRetryAutoFail", reason=...)` — the failure is reclassified as a `connection`-category error (so it is not misreported as a content failure)

**Debugging tip**: If this fires, the model never produced a real final answer — the root cause is an infrastructure/connection failure, not answer content. Inspect the placeholder message (first ~500 chars are logged) for the underlying error.

---

## 4. Abstention Check

**Stage name**: `AbstentionCheck`

**What triggers it**: The LLM refused to answer or abstained from responding. The check uses a judge LLM (parsing model) to analyze the raw response for refusal patterns.

**When it runs**: Only when `abstention_enabled=True`. Skips if recursion limit was already reached.

**Auto-fail result**:
- `result.template.verify_result`: `False`
- `result.template.abstention_check_performed`: `True`
- `result.template.abstention_detected`: `True`
- `result.template.abstention_reasoning`: The judge's reasoning (`"..."`)
- `result.metadata.failure`: `Failure(category="abstention", stage="abstention_check", reason=...)`

**How to configure**:

```python
# Enable abstention detection (default: disabled)
config = VerificationConfig(
    answering_models=[...],
    parsing_models=[...],
    abstention_enabled=True,
)

# Inject custom instructions for the abstention detector
from karenina.schemas.verification.prompt_config import PromptConfig

config = VerificationConfig(
    answering_models=[...],
    parsing_models=[...],
    abstention_enabled=True,
    prompt_config=PromptConfig(
        abstention_detection="Consider partial refusals as abstentions.",
    ),
)
```

**Debugging tip**: Check `result.template.abstention_reasoning` for the judge's explanation. If the detector is too aggressive (flagging valid but cautious responses as abstentions), disable it or tune the detection via `prompt_config.abstention_detection`.

---

## 5. Sufficiency Check

**Stage name**: `SufficiencyCheck`

**What triggers it**: The LLM response lacks sufficient information to populate the answer template. The check uses a judge LLM (parsing model) to compare the raw response against the template's JSON schema.

**When it runs**: Only when `sufficiency_enabled=True`. Skips if recursion limit was reached, trace validation failed, or abstention was detected.

**Auto-fail result**:
- `result.template.verify_result`: `False`
- `result.template.sufficiency_check_performed`: `True`
- `result.template.sufficiency_detected`: `False` (False means insufficient)
- `result.template.sufficiency_reasoning`: What the judge found missing (`"..."`)
- Note: `result.metadata.failure` may be `None` for a sufficiency auto-fail — the `sufficiency_*` template fields above are the reliable signal

**How to configure**:

```python
# Enable sufficiency detection (default: disabled)
config = VerificationConfig(
    answering_models=[...],
    parsing_models=[...],
    sufficiency_enabled=True,
)

# Inject custom instructions for the sufficiency detector
config = VerificationConfig(
    answering_models=[...],
    parsing_models=[...],
    sufficiency_enabled=True,
    prompt_config=PromptConfig(
        sufficiency_detection="Consider partial answers as sufficient if they address the main question.",
    ),
)
```

**Debugging tip**: `sufficiency_detected=False` means the response was judged insufficient. Check `sufficiency_reasoning` for what the judge found missing. If the check is too strict, the template schema may have overly specific required fields that responses cannot reasonably cover.

---

## Guard Interaction Summary

Guards run in order and short-circuit. If an earlier guard fires, later guards do not run:

```
recursion_limit_autofail    (if recursion limit hit, STOP)
         |
         v
trace_validation_autofail   (if MCP trace invalid, STOP)
         |
         v
placeholder_retry_autofail  (if model-retry placeholder, STOP)
         |
         v
abstention_check            (if abstention detected, STOP)
         |
         v
sufficiency_check           (if response insufficient, STOP)
         |
         v
parse_template              (proceed to evaluation)
```

**All guards set `result.template.verify_result=False`**. Check `result.metadata.failure.stage` to identify which guard fired first: the hard guards record their pipeline stage name (`"RecursionLimitAutoFail"`, `"TraceValidationAutoFail"`, `"PlaceholderRetryAutoFail"`), while an abstention auto-fail records `"abstention_check"`. Only the first guard to trigger is recorded (subsequent guards are skipped via short-circuiting). For a sufficiency auto-fail, `result.metadata.failure` may be `None` — rely on `result.template.sufficiency_detected=False` instead.

**`verify_result` has a third state**: `False` always means a guard or the verification stage explicitly recorded a failure. On a **stage error** (e.g. an auth or connection failure at `GenerateAnswer`) the stage never completes, so `result.template.verify_result` is `None` — no verdict was ever produced — and `result.metadata.failure` carries the failing stage name in `.stage` with a `category` reflecting the classified error type (`connection`, `rate_limit`, `server_error`, `timeout`; `unexpected_error` is the fallback).
