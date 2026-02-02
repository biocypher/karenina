"""Usage metadata extraction from Anthropic SDK responses.

This module provides utilities for extracting token usage and cost information
from Anthropic Python SDK response objects.
"""

from __future__ import annotations

import logging
from typing import Any

from karenina.ports.usage import UsageMetadata

logger = logging.getLogger(__name__)


def extract_usage(usage: Any, model: str | None = None) -> UsageMetadata:
    """Extract usage metadata from an Anthropic SDK usage object.

    Args:
        usage: The usage object from an Anthropic API response.
            Expected attributes: input_tokens, output_tokens.
            Optional: cache_read_input_tokens, cache_creation_input_tokens.
        model: Optional model name to include in metadata.

    Returns:
        UsageMetadata with token counts and optional cache information.
    """
    if usage is None:
        return UsageMetadata(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            model=model,
        )

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    total_tokens = input_tokens + output_tokens

    # Extract cache tokens if available (Anthropic prompt caching)
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    cache_creation = getattr(usage, "cache_creation_input_tokens", None)

    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        model=model,
    )


def extract_usage_from_response(response: Any, model: str | None = None) -> UsageMetadata:
    """Extract usage metadata from an Anthropic SDK response object.

    Args:
        response: An Anthropic API response object (e.g., from messages.create).
            Expected to have a usage attribute.
        model: Optional model name to include in metadata.

    Returns:
        UsageMetadata with token counts.
    """
    usage = getattr(response, "usage", None)
    return extract_usage(usage, model=model)


def aggregate_usage(base: UsageMetadata, new: UsageMetadata) -> UsageMetadata:
    """Aggregate multiple usage metadata objects.

    Useful for accumulating usage across multiple API calls in an agent loop.

    Args:
        base: The base usage metadata to add to.
        new: The new usage metadata to add.

    Returns:
        A new UsageMetadata with accumulated values.
    """
    return UsageMetadata(
        input_tokens=base.input_tokens + new.input_tokens,
        output_tokens=base.output_tokens + new.output_tokens,
        total_tokens=base.total_tokens + new.total_tokens,
        cost_usd=(base.cost_usd or 0) + (new.cost_usd or 0) if base.cost_usd or new.cost_usd else None,
        cache_read_tokens=(base.cache_read_tokens or 0) + (new.cache_read_tokens or 0)
        if base.cache_read_tokens or new.cache_read_tokens
        else None,
        cache_creation_tokens=(base.cache_creation_tokens or 0) + (new.cache_creation_tokens or 0)
        if base.cache_creation_tokens or new.cache_creation_tokens
        else None,
        model=new.model or base.model,
    )


def aggregate_usage_from_response(base: UsageMetadata, response: Any) -> UsageMetadata:
    """Aggregate usage metadata from a response into a base.

    Convenience function for accumulating usage in an agent loop.

    Args:
        base: The base usage metadata to add to.
        response: An Anthropic API response object with usage attribute.

    Returns:
        A new UsageMetadata with accumulated values.
    """
    new_usage = extract_usage_from_response(response, model=base.model)
    return aggregate_usage(base, new_usage)
