"""Usage extraction for Claude Agent SDK responses.

This module provides utilities for extracting UsageMetadata from SDK
ResultMessage objects. This consolidates the usage extraction logic
that was previously duplicated in llm.py and agent.py.

Usage:
    >>> from claude_agent_sdk import ResultMessage
    >>> from karenina.adapters.claude_agent_sdk.usage import extract_sdk_usage
    >>>
    >>> # After receiving a ResultMessage from SDK
    >>> usage = extract_sdk_usage(result_message)
    >>> print(f"Total tokens: {usage.total_tokens}")
    >>> print(f"Cost: ${usage.cost_usd:.4f}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from karenina.ports import UsageMetadata

if TYPE_CHECKING:
    from claude_agent_sdk import ResultMessage


def extract_sdk_usage(
    result: ResultMessage,
    model: str | None = None,
) -> UsageMetadata:
    """Extract usage metadata from a Claude SDK ResultMessage.

    This function extracts token counts, cost information, and cache metadata
    from the SDK's ResultMessage into a unified UsageMetadata dataclass.

    The SDK provides usage data in result.usage as a dict with keys:
    - input_tokens: Number of input tokens
    - output_tokens: Number of output tokens
    - cache_read_input_tokens: Tokens read from Anthropic's prompt cache
    - cache_creation_input_tokens: Tokens used to create prompt cache

    Cost is provided separately in result.total_cost_usd.

    Args:
        result: SDK ResultMessage containing usage data.
        model: Optional model name to include in metadata. If not provided,
            the UsageMetadata.model field will be None.

    Returns:
        UsageMetadata with token counts, cost, and cache information.

    Example:
        >>> result = await get_sdk_result()  # Returns ResultMessage
        >>> usage = extract_sdk_usage(result, model="claude-sonnet-4-20250514")
        >>> print(f"Input: {usage.input_tokens}, Output: {usage.output_tokens}")
        >>> if usage.cost_usd:
        ...     print(f"Cost: ${usage.cost_usd:.4f}")
        >>> if usage.cache_read_tokens:
        ...     print(f"Cache hits: {usage.cache_read_tokens} tokens")
    """
    # Extract usage dict - may be None if no usage data
    usage_data: dict[str, Any] = getattr(result, "usage", None) or {}

    # Extract token counts with defaults of 0
    input_tokens = usage_data.get("input_tokens", 0)
    output_tokens = usage_data.get("output_tokens", 0)
    total_tokens = input_tokens + output_tokens

    # Extract cache tokens if available (Anthropic-specific)
    cache_read = usage_data.get("cache_read_input_tokens")
    cache_creation = usage_data.get("cache_creation_input_tokens")

    # Extract cost - SDK provides this as total_cost_usd
    cost_usd = getattr(result, "total_cost_usd", None)

    return UsageMetadata(
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        total_tokens=int(total_tokens),
        cost_usd=cost_usd,
        cache_read_tokens=int(cache_read) if cache_read is not None else None,
        cache_creation_tokens=int(cache_creation) if cache_creation is not None else None,
        model=model,
    )
