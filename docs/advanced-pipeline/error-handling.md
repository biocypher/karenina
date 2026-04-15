---
jupyter:
  jupytext:
    formats: docs/advanced-pipeline//md,docs/notebooks/advanced/pipeline//ipynb
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

# Error Handling and Retries

This page documents karenina's category-based error handling and retry system. It covers how exceptions raised inside the verification pipeline are classified, which categories are retried with which budgets, how to customize the classifier, how to escalate timeouts on retry, and how to read the per-pipeline retry counts that land on every `VerificationResult`.

For the higher-level pipeline overview see the [Pipeline Overview](index.md). For per-stage details see [Stages in Detail](stages.md).

```python tags=["hide-cell"]
# Mock cell: this notebook does not perform live LLM calls.
# All examples below construct policies, registries, and exceptions
# in-process and inspect their behavior. No API keys are required.
```

## Why a Single Mechanism

Earlier versions of karenina layered retries in several independent places: a `tenacity` decorator on LLM adapters, a separate decorator on streaming timeouts, ad-hoc retry loops on rubric evaluators, and a per-turn retry inside the scenario manager. Two consequences followed.

First, retries compounded. A timeout could trigger several layers of retry simultaneously, so a single failing call multiplied into many. Second, the result was opaque: different layers used different classification logic, so the final `VerificationResult` could not say which category of error fired or how many retries had been spent.

The harmonized system replaces those layers with one `ErrorRegistry` (classification), one `RetryPolicy` (budgets), and one `RetryExecutor` (the loop). Every adapter routes its calls through it, and per-pipeline retry counts flow into result metadata.

## ErrorCategory

Every exception caught inside the pipeline is mapped to one of five categories.

```python
from karenina.utils.errors import ErrorCategory

list(ErrorCategory)
```

The categories carry no special metadata; they exist purely as the routing key between classification and retry budgets. `is_retryable()` returns `True` for everything except `PERMANENT`.

```python
[(category.value, category.is_retryable()) for category in ErrorCategory]
```

| Category | Meaning | Default Retry Budget |
|----------|---------|----------------------|
| `CONNECTION` | DNS/network/portal failures, dropped sockets | 3 |
| `TIMEOUT` | Wall-clock or streaming timeouts | 3 |
| `RATE_LIMIT` | 429, overloaded responses, zero-content streaming timeouts | 5 |
| `SERVER_ERROR` | 5xx responses from the model provider | 2 |
| `PERMANENT` | Anything else (bad input, schema errors, etc.) | 0 (never retried) |

## ErrorRegistry: How Errors Are Classified

`ErrorRegistry` is the classifier that maps a live exception object to an `ErrorCategory`. It is intentionally extensible: you start from the built-in defaults and add your own rules whenever a new model provider, MCP server, or wrapper library exposes a non-standard exception type.

### The Match Order

`registry.classify(exc)` walks five rule layers in order and returns the first hit. Knowing the order matters when you need to override a built-in default with a custom rule.

| # | Rule | Source | Match by |
|---|------|--------|----------|
| 1 | User-registered exception classes | `register(SomeException, category)` | `isinstance(exc, SomeException)` |
| 2 | User-registered type names | `register("SomeError", category)` or `register_pattern(..., match_type="type_name")` | `type(exc).__name__ == "SomeError"` |
| 3 | User-registered message substrings | `register_pattern(..., match_type="message_substring")` | lowercased substring of `str(exc)` |
| 4 | Built-in type rules (MRO) | hardcoded | walks `type(exc).__mro__` for known names |
| 5 | Built-in message substrings | hardcoded | lowercased substring of `str(exc)` |
| 6 | Default | none | returned as `PERMANENT` |

User rules always run first and short-circuit before the built-ins, so a custom registration can override any default classification.

### Built-in Type Rules

The built-in type matcher walks the exception's MRO, so it catches not only well-known framework types but also any subclass of them. This is how `karenina.exceptions.StreamingTimeoutError` ends up classified as `TIMEOUT` automatically without needing its own entry: it inherits from `TimeoutError`, which is in the built-in table.

| Type Name | Category |
|-----------|----------|
| `ConnectionError`, `APIConnectionError` | `CONNECTION` |
| `TimeoutError`, `ReadTimeout`, `ConnectTimeout`, `APITimeoutError`, `StreamingTimeoutError` | `TIMEOUT` |
| `RateLimitError`, `OverloadedError` | `RATE_LIMIT` |
| `InternalServerError`, `HTTPError` | `SERVER_ERROR` |

### Built-in Message Substrings

When no type rule matches, the registry falls back to substring matching on the lowercased exception message. The substrings are checked in this fixed order; the first match wins.

| Substring | Category |
|-----------|----------|
| `connection`, `network`, `dns`, `event loop`, `portal`, `temporary failure` | `CONNECTION` |
| `timeout`, `timed out` | `TIMEOUT` |
| `rate limit`, `429`, `overloaded` | `RATE_LIMIT` |
| `500`, `502`, `503` | `SERVER_ERROR` |

The `event loop` and `portal` keywords exist because anyio portal lifecycle errors propagate as plain `RuntimeError` and would otherwise fall through to `PERMANENT`. Treating them as `CONNECTION` errors makes the executor retry them, which masks transient races during worker startup.

### Defaults in Action

```python
from karenina.utils.errors import ErrorRegistry

registry = ErrorRegistry()

# Built-in type rule (CONNECTION)
print(registry.classify(ConnectionError("reset by peer")))

# Built-in subclass via MRO (TIMEOUT)
print(registry.classify(TimeoutError("read timeout")))

# Built-in message substring (RATE_LIMIT)
print(registry.classify(RuntimeError("HTTP 429: too many requests")))

# Built-in message substring for portal lifecycle (CONNECTION)
print(registry.classify(RuntimeError("portal already closed")))

# No match: PERMANENT
print(registry.classify(ValueError("unparseable input")))
```

### Custom Rules

Real-world deployments usually need a handful of custom rules to handle exceptions thrown by specific providers, MCP servers, or wrapper libraries. Three registration paths exist.

**Register by exception class.** Use this when you can import the class. It is matched by `isinstance`, so subclasses of your registered class also match.

```python
class VllmQueueTimeout(Exception):
    pass

registry = ErrorRegistry()
registry.register(VllmQueueTimeout, ErrorCategory.RATE_LIMIT)

print(registry.classify(VllmQueueTimeout("queue full")))
```

**Register by type name.** Use this when the exception class lives in an optional dependency you do not want to import at module load time. The name is compared exactly against `type(exc).__name__`.

```python
registry = ErrorRegistry()
registry.register_pattern(
    "BedrockThrottlingException",
    ErrorCategory.RATE_LIMIT,
    match_type="type_name",
)
```

**Register by message substring.** Use this when the exception type is too broad (often a bare `Exception` or `RuntimeError`) and only the message reveals the underlying problem. Matching is lowercased substring containment.

```python
registry = ErrorRegistry()
registry.register_pattern(
    "context length exceeded",
    ErrorCategory.PERMANENT,
    match_type="message_substring",
)
registry.register_pattern(
    "model is currently overloaded for unprivileged users",
    ErrorCategory.RATE_LIMIT,
    match_type="message_substring",
)

print(registry.classify(RuntimeError("context length exceeded: 200000 > 128000")))
```

A subtle and useful consequence of the priority order is that you can demote a built-in classification. For example, the built-in `timeout` substring sends "timed out" messages to `TIMEOUT`. If a particular provider's "timed out" really means "queue saturated, retry slowly with backoff", you can register the same provider's exception class with `RATE_LIMIT` and that custom rule wins because it sits above the built-in matcher.

### Customizing Classification from VerificationConfig

The pipeline does not expect callers to construct an `ErrorRegistry` by hand. `VerificationConfig.custom_error_patterns` accepts a list of declarative `ErrorPatternConfig` entries that the runner converts into registry rules at the start of each verification call. This is the supported entry point for customization in production code, and it is what gets serialized into checkpoint files and YAML presets.

```python
from karenina.utils.retry_policy import ErrorPatternConfig

custom_patterns = [
    ErrorPatternConfig(
        pattern="VllmQueueTimeout",
        category="rate_limit",
        match_type="type_name",
    ),
    ErrorPatternConfig(
        pattern="context length exceeded",
        category="permanent",
        match_type="message_substring",
    ),
]
custom_patterns
```

You then pass `custom_error_patterns=custom_patterns` to `VerificationConfig` alongside the rest of your pipeline configuration. The runner builds an `ErrorRegistry` once per pipeline call, registers every entry from `custom_error_patterns`, and threads it through `VerificationContext`. Every adapter therefore sees the same classification rules within a single verification run, even when it dispatches across multiple worker threads.

### Inspecting a Classification Decision

If you are unsure why an exception is being classified one way or another, the simplest debugging recipe is to feed it through a fresh `ErrorRegistry` and print the result. Because the registry has no hidden state beyond what you register on it, the classification is fully reproducible.

```python
def explain(exc):
    return ErrorRegistry().classify(exc)

cases = [
    ConnectionError("connection reset"),
    TimeoutError("operation timed out"),
    RuntimeError("HTTP 503 Service Unavailable"),
    RuntimeError("portal already closed by worker shutdown"),
    ValueError("invalid argument"),
]

for exc in cases:
    print(f"{type(exc).__name__:20s} -> {explain(exc).value}")
```

A classification of `permanent` is the registry's way of saying "I have no rule for this". If that result surprises you, the fix is almost always to add a rule via `register_pattern` (locally) or `custom_error_patterns` (in `VerificationConfig`).

## RetryPolicy: Budgets and Backoff

`RetryPolicy` groups one `CategoryRetryConfig` per retryable category. Each `CategoryRetryConfig` has four knobs.

```python
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy

CategoryRetryConfig.model_fields.keys()
```

| Field | Default | Meaning |
|-------|---------|---------|
| `max_attempts` | varies | Number of retries (not total calls). 0 disables retry for this category. |
| `backoff_min` | 1.0 | Lower bound for the per-attempt backoff in seconds. |
| `backoff_max` | 10.0 | Upper bound for the per-attempt backoff in seconds. |
| `backoff_multiplier` | 2.0 | Exponential growth factor between attempts. |

The actual delay before each retry is `random.uniform(0, min(backoff_min * multiplier^attempt, backoff_max))`. The randomization gives every retry decorrelated jitter, which prevents synchronized thundering herds when multiple workers hit the same rate limit.

### Default Policy

```python
policy = RetryPolicy()
{
    "connection": policy.connection.max_attempts,
    "timeout": policy.timeout.max_attempts,
    "rate_limit": policy.rate_limit.max_attempts,
    "server_error": policy.server_error.max_attempts,
}
```

`RATE_LIMIT` gets the largest budget because rate limits are usually the most recoverable. `SERVER_ERROR` gets the smallest because a sustained 5xx usually indicates an outage that retrying will not fix.

### Configuring a Custom Policy

```python
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy

custom_policy = RetryPolicy(
    connection=CategoryRetryConfig(max_attempts=5, backoff_min=2.0, backoff_max=20.0),
    timeout=CategoryRetryConfig(max_attempts=4, backoff_min=10.0, backoff_max=60.0),
    rate_limit=CategoryRetryConfig(max_attempts=8, backoff_min=10.0, backoff_max=60.0),
    server_error=CategoryRetryConfig(max_attempts=1),
)

custom_policy.timeout.max_attempts, custom_policy.rate_limit.max_attempts
```

You then pass `retry_policy=custom_policy` to `VerificationConfig` alongside the rest of your pipeline configuration. The pipeline-level `retry_policy` is stamped onto every `ModelConfig` before tasks run, so a single setting at the top governs all per-model adapter calls. A `ModelConfig` can override this by setting its own `retry_policy`; if left as `None` it inherits the pipeline-level policy.

`RetryPolicy.derive_sdk_max_retries()` returns the maximum `max_attempts` across all categories. This is the value that gets passed into SDK clients (Anthropic, OpenAI, ...) that accept a single `max_retries` parameter, ensuring no category is starved by a too-low SDK-level cap.

```python
RetryPolicy().derive_sdk_max_retries()
```

## TimeoutEscalation: Growing the Budget on Retry

A flat `request_timeout` is sometimes the wrong tool. A model that needs ten extra seconds to finish reasoning will keep timing out at the same boundary on every retry. `TimeoutEscalationConfig` lets the per-attempt timeout grow on each `TIMEOUT`-category retry while leaving the other categories alone.

Three strategies are available.

| Strategy | Formula | Required Field |
|----------|---------|----------------|
| `additive` | `timeout(n) = min(base + increment * n, max_timeout)` | `increment > 0` |
| `multiplicative` | `timeout(n) = min(base * multiplier^n, max_timeout)` | `multiplier > 1.0` |
| `linear` | `timeout(n) = base + (max_timeout - base) * n / max_attempts` | `max_timeout` |

`n` is the number of `TIMEOUT` retries used so far (0 on the original call, k on the k-th retry). `additive` and `multiplicative` use `max_timeout` as an optional cap; `linear` requires it as the endpoint of the interpolation.

```python
from karenina.utils.retry_policy import (
    RetryPolicy,
    TimeoutEscalationConfig,
    compute_escalated_timeout,
)

config = TimeoutEscalationConfig(
    strategy="additive",
    increment=15.0,
    max_timeout=180.0,
)

# How the timeout grows for retries 0..4 with base=30s and max_attempts=3
[
    compute_escalated_timeout(base_timeout=30.0, timeout_attempt=n, config=config, max_attempts=3)
    for n in range(5)
]
```

```python
# Multiplicative variant capped at 120s
config = TimeoutEscalationConfig(strategy="multiplicative", multiplier=2.0, max_timeout=120.0)
[
    compute_escalated_timeout(base_timeout=15.0, timeout_attempt=n, config=config, max_attempts=4)
    for n in range(5)
]
```

```python
# Linear interpolation between base and max across the timeout retry budget
config = TimeoutEscalationConfig(strategy="linear", max_timeout=120.0)
[
    compute_escalated_timeout(base_timeout=30.0, timeout_attempt=n, config=config, max_attempts=3)
    for n in range(4)
]
```

Escalation only fires inside `RetryExecutor.execute_with_timeout` and `aexecute_with_timeout`, which are the entry points used by `stream_invoke`. Other adapter calls keep using the base `request_timeout` for every retry.

## RetryExecutor: How the Loop Runs

`RetryExecutor(policy, registry)` wraps a callable and re-invokes it on retryable exceptions. Each call creates fresh per-category counters, so the budgets reset between calls. The pseudocode is small enough to read whole.

```
budgets = {}
while True:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        category = registry.classify(exc)
        config = policy[category]
        used = budgets.get(category, 0)
        if not category.is_retryable() or used >= config.max_attempts:
            raise
        budgets[category] = used + 1
        record_retry(category)
        sleep(compute_backoff(config, used))
```

Four entry points are available.

| Method | Sync/Async | Timeout Escalation |
|--------|------------|--------------------|
| `execute(fn, *args, **kwargs)` | sync | no |
| `aexecute(fn, *args, **kwargs)` | async | no |
| `execute_with_timeout(fn, *args, timeout=..., **kwargs)` | sync | yes |
| `aexecute_with_timeout(fn, *args, timeout=..., **kwargs)` | async | yes |

The `_with_timeout` variants forward `timeout=current_timeout` into the wrapped callable on every attempt, escalating it for `TIMEOUT` retries when `policy.timeout_escalation` is set.

### A Working Example

We can exercise the executor end-to-end without any LLM calls by wrapping a callable that fails twice with a connection error and then succeeds.

```python
from karenina.utils.errors import ErrorRegistry
from karenina.utils.retry_policy import (
    CategoryRetryConfig,
    RetryExecutor,
    RetryPolicy,
)

attempts = {"count": 0}

def flaky():
    attempts["count"] += 1
    if attempts["count"] < 3:
        raise ConnectionError("transient network blip")
    return "ok"

executor = RetryExecutor(
    policy=RetryPolicy(
        connection=CategoryRetryConfig(max_attempts=5, backoff_min=0.0, backoff_max=0.0),
    ),
    registry=ErrorRegistry(),
)

result = executor.execute(flaky)
result, attempts["count"]
```

Two retries fired (third call succeeded), the executor returned the success value, and the budget tracker shows that two of the five available connection retries were spent.

```python
# Permanent errors short-circuit immediately
def bad_input():
    raise ValueError("not even a number")

try:
    executor.execute(bad_input)
except ValueError as exc:
    print(f"raised on first call: {exc}")
```

### Where Adapters Use It

Every LLM and parser adapter holds its own `RetryExecutor` constructed from the pipeline-level `RetryPolicy`.

- **LangChain** LLM and parser route `invoke`, `ainvoke`, `stream_invoke`, and parser calls through the executor. A per-attempt wall-clock guard runs alongside the SDK-level timeout to make sure runaway calls cannot exceed `request_timeout`.
- **Claude Tool** and **Claude Agent SDK** configure their SDK-level `max_retries` from `RetryPolicy.derive_sdk_max_retries()` and additionally wrap streaming calls with `RetryExecutor` so the error-classification path stays consistent.
- **Deep agents** (LangChain DeepAgents) do the same.
- **Manual** never retries; the executor is bypassed because the adapter is interactive.

The legacy `tenacity` decorators and the standalone `karenina.utils.retry` module have been removed. Any code that imported them must move to `RetryExecutor`.

## Observability: track_retries and retry_counts

`RetryExecutor` writes every retry decision into a context-local tracker installed by `track_retries()`. The pipeline opens one tracker per `run_single_model_verification` call, so the counts cover everything that happened inside that one pipeline run, across both sync and async paths and across worker threads (each tracker is local to its `contextvars` context).

```python
from karenina.utils.retry_policy import (
    CategoryRetryConfig,
    RetryExecutor,
    RetryPolicy,
    track_retries,
)
from karenina.utils.errors import ErrorRegistry

# Reuse the flaky callable from above
attempts["count"] = 0
executor = RetryExecutor(
    policy=RetryPolicy(
        connection=CategoryRetryConfig(max_attempts=5, backoff_min=0.0, backoff_max=0.0),
    ),
    registry=ErrorRegistry(),
)

with track_retries(RetryPolicy()) as tracker:
    executor.execute(flaky)

tracker
```

The tracker is a dict keyed by `ErrorCategory.value`. Each entry is `{"used": int, "budget": int}`, where `budget` reflects the `max_attempts` of the policy passed to `track_retries` and `used` counts how many of those retries actually fired. Categories that never triggered show up with `used=0`, so consumers can always see "we had N attempts available, used K of them".

The pipeline copies the final tracker into `VerificationResultMetadata.retry_counts`. The shape matches the dict above. A value of `None` means the pipeline ran without an active tracker (for example, in legacy code paths that bypass the executor entirely). Reading `retry_counts` is the supported way to answer questions like:

- Did this question succeed only because we retried? (any `used > 0`)
- Are we hitting the timeout budget on a particular model? (`timeout.used == timeout.budget`)
- Should the rate-limit budget be larger for this provider? (`rate_limit.used` consistently near `rate_limit.budget`)

## StreamingTimeoutError

LLM streaming has its own typed exception, `StreamingTimeoutError(KareninaError, TimeoutError)`. It inherits from the standard `TimeoutError` so the registry classifies it as `TIMEOUT` automatically through the MRO check, and from `KareninaError` so callers can catch it as part of the karenina hierarchy. It also carries a `partial_content: str` attribute holding any tokens accumulated before the timeout fired.

```python
from karenina.exceptions import StreamingTimeoutError
from karenina.utils.errors import ErrorRegistry

err = StreamingTimeoutError("stream stalled after 30s", partial_content="Once upon a time, ")
ErrorRegistry().classify(err).value, err.partial_content
```

The `generate_answer` stage salvages this partial content when present, so a streaming timeout still produces a usable answer text on the result, marked with the `response_timeout_partial` guard. Zero-content streaming timeouts (no tokens received at all) are reclassified as `RATE_LIMIT` so they get the larger rate-limit retry budget instead of the smaller timeout budget; in practice these almost always indicate a queue or throughput issue at the model server rather than a slow generation.

## What Was Removed

The harmonization deletes several previously-supported retry surfaces.

- **`max_transient_retries` on `VerificationConfig` and `ModelConfig`.** Replaced by `RetryPolicy`. Configuration files written against the old field must be updated.
- **`transient: bool` on context and result metadata.** Replaced by a structured `failure: Failure | None` on `VerificationResultMetadata`. The `failure` holds a `FailureCategory`, a derived `FailureGroup`, the originating stage, and a reason.
- **`tenacity` retry decorators on LangChain LLM and parser, deep agents, and the streaming-timeout retry decorator.** All replaced by `RetryExecutor`.
- **Evaluator retry decorators (abstention, sufficiency).** Removed entirely; the executor is the only retry surface.
- **Scenario turn retry.** A failing turn now terminates the scenario instead of retrying. Per-call retry still happens inside the executor.
- **`karenina.utils.retry` module.** Re-exports point at the new modules; importing the old paths raises.

## Related

- [Stages in Detail](stages.md): which stages call the executor and which guards record streaming-timeout partial content.
- [Adapters Overview](../advanced-adapters/index.md): where individual adapters install their `RetryExecutor`.
- [Writing Adapters](../advanced-adapters/writing-adapters.md): how to wire `RetryExecutor` into a new adapter.
