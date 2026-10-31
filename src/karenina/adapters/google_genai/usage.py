"""Usage extraction for the google_genai adapter.

Maps Gemini ``usage_metadata`` onto ``UsageMetadata`` (design Section 12):

- ``input_tokens`` = ``prompt_token_count`` + ``tool_use_prompt_token_count``
  (the prompt count already includes cached tokens),
- ``output_tokens`` = ``candidates_token_count`` + ``thoughts_token_count``
  (thinking is billed at the output rate),
- ``cache_read_tokens`` = ``cached_content_token_count``,
- ``reasoning_tokens`` = ``thoughts_token_count``,
- ``web_search_queries`` = the number of Google Search queries of the call.

``cost_usd`` stays None: karenina has no price table.
"""

from __future__ import annotations

import logging
from typing import Any

from karenina.ports import UsageMetadata

logger = logging.getLogger(__name__)


def _count(usage_metadata: Any, name: str) -> int | None:
    """Read one optional token count."""
    value = getattr(usage_metadata, name, None)
    return int(value) if value is not None else None


def extract_usage(
    usage_metadata: Any,
    *,
    model: str | None,
    web_search_queries: int | None = None,
) -> UsageMetadata:
    """Convert one call's ``usage_metadata`` into UsageMetadata.

    Args:
        usage_metadata: ``response.usage_metadata`` (None when the call
            reported none).
        model: Model name recorded on the usage.
        web_search_queries: Google Search queries made by the call, or None
            when Google Search was not available to it.

    Returns:
        The call's usage. All token counts are zero when no metadata came.
    """
    if usage_metadata is None:
        return UsageMetadata(model=model, web_search_queries=web_search_queries)

    prompt = _count(usage_metadata, "prompt_token_count") or 0
    tool_prompt = _count(usage_metadata, "tool_use_prompt_token_count") or 0
    candidates = _count(usage_metadata, "candidates_token_count") or 0
    thoughts = _count(usage_metadata, "thoughts_token_count")
    input_tokens = prompt + tool_prompt
    output_tokens = candidates + (thoughts or 0)
    total = _count(usage_metadata, "total_token_count")

    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total if total is not None else input_tokens + output_tokens,
        cache_read_tokens=_count(usage_metadata, "cached_content_token_count"),
        model=model,
        reasoning_tokens=thoughts,
        web_search_queries=web_search_queries,
    )


def _add_optional(a: int | None, b: int | None) -> int | None:
    """Sum two optional counts, keeping None only when both are None."""
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def aggregate_usage(base: UsageMetadata, new: UsageMetadata) -> UsageMetadata:
    """Sum two usage records field by field.

    Optional fields stay None only when neither record reported them, so a
    real zero is kept distinct from "not reported".

    Args:
        base: Usage accumulated so far.
        new: Usage of the next call.

    Returns:
        A new UsageMetadata with the summed values.
    """
    cost: float | None = None
    if base.cost_usd is not None or new.cost_usd is not None:
        cost = (base.cost_usd or 0.0) + (new.cost_usd or 0.0)
    return UsageMetadata(
        input_tokens=base.input_tokens + new.input_tokens,
        output_tokens=base.output_tokens + new.output_tokens,
        total_tokens=base.total_tokens + new.total_tokens,
        cost_usd=cost,
        cache_read_tokens=_add_optional(base.cache_read_tokens, new.cache_read_tokens),
        cache_creation_tokens=_add_optional(base.cache_creation_tokens, new.cache_creation_tokens),
        model=new.model or base.model,
        reasoning_tokens=_add_optional(base.reasoning_tokens, new.reasoning_tokens),
        web_search_queries=_add_optional(base.web_search_queries, new.web_search_queries),
    )


def per_call_usage(usage: UsageMetadata) -> dict[str, int]:
    """Per-turn usage for ``Message.usage_metadata`` on trace messages.

    Uses the key names of the other adapters' per-message usage
    (``cache_read_input_tokens``), so per-turn accounting survives into
    ``template.trace_messages``.

    Args:
        usage: One call's usage.

    Returns:
        Dict with input and output tokens plus each optional count that was
        reported.
    """
    result = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
    if usage.cache_read_tokens is not None:
        result["cache_read_input_tokens"] = usage.cache_read_tokens
    if usage.reasoning_tokens is not None:
        result["reasoning_tokens"] = usage.reasoning_tokens
    if usage.web_search_queries is not None:
        result["web_search_queries"] = usage.web_search_queries
    return result
