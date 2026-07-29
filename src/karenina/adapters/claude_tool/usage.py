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


def merge_stream_usage(current: UsageMetadata, usage: Any, model: str | None = None) -> UsageMetadata:
    """Merge a streaming usage payload into already-captured usage.

    Anthropic streaming spreads usage across events: ``message_start``
    carries ``input_tokens`` (and cache fields), while ``message_delta``
    carries the cumulative ``output_tokens``. This helper overlays
    whichever fields the event reports onto the current snapshot, so a
    mid-stream interruption still leaves the usage that arrived so far.

    Args:
        current: The usage captured so far (fields not reported by this
            event are preserved).
        usage: The usage object from a streaming event. Attributes that
            are absent or None leave the current value untouched.
        model: Optional model name to include in metadata.

    Returns:
        A new UsageMetadata with the merged values.
    """
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    cache_creation = getattr(usage, "cache_creation_input_tokens", None)

    merged_input = input_tokens if input_tokens is not None else current.input_tokens
    merged_output = output_tokens if output_tokens is not None else current.output_tokens

    return UsageMetadata(
        input_tokens=merged_input,
        output_tokens=merged_output,
        total_tokens=(merged_input or 0) + (merged_output or 0),
        cache_read_tokens=cache_read if cache_read is not None else current.cache_read_tokens,
        cache_creation_tokens=cache_creation if cache_creation is not None else current.cache_creation_tokens,
        model=model if model is not None else current.model,
    )


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
