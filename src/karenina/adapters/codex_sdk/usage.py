"""Usage extraction for Codex SDK turn results.

Codex reports token usage through ``ThreadTokenUsage`` objects carried on
``TurnResult.usage`` and on ``thread/tokenUsage/updated`` notifications.
Each holds two ``TokenUsageBreakdown`` records (``last`` for the most
recent model call, ``total`` for the cumulative turn) with fields
``cached_input_tokens``, ``input_tokens``, ``output_tokens``,
``reasoning_output_tokens``, and ``total_tokens``.

Karenina maps the cumulative ``total`` breakdown. Cost is always None:
custom endpoints have no price table and codex does not report cost.
"""

from __future__ import annotations

import logging
from typing import Any

from karenina.ports import UsageMetadata

logger = logging.getLogger(__name__)


def _breakdown_int(breakdown: Any, field: str) -> int:
    """Read an int field from a TokenUsageBreakdown defensively."""
    value = getattr(breakdown, field, None)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_codex_usage(
    usage: Any,
    model: str | None = None,
) -> UsageMetadata:
    """Extract UsageMetadata from a Codex ThreadTokenUsage object.

    Uses the cumulative ``total`` breakdown when present, falling back to
    ``last``. Returns zeroed usage when nothing was reported (for example
    on a timeout before the first ``thread/tokenUsage/updated``
    notification arrived).

    Args:
        usage: A ThreadTokenUsage (from ``TurnResult.usage`` or a token
            usage notification), or None.
        model: Optional model name recorded on the result.

    Returns:
        UsageMetadata with token counts. ``cached_input_tokens`` maps to
        ``cache_read_tokens``. ``cost_usd`` is always None.
    """
    if usage is None:
        return UsageMetadata(model=model)

    breakdown = getattr(usage, "total", None) or getattr(usage, "last", None)
    if breakdown is None:
        # Tolerate being handed a bare breakdown instead of the wrapper.
        breakdown = usage

    input_tokens = _breakdown_int(breakdown, "input_tokens")
    output_tokens = _breakdown_int(breakdown, "output_tokens")
    total_tokens = _breakdown_int(breakdown, "total_tokens") or (input_tokens + output_tokens)
    cached_input = getattr(breakdown, "cached_input_tokens", None)

    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cost_usd=None,
        cache_read_tokens=int(cached_input) if cached_input is not None else None,
        cache_creation_tokens=None,
        model=model,
    )
