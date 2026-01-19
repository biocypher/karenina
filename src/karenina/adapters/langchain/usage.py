"""LangChain usage extraction utilities.

This module provides functions to extract and convert usage metadata from
LangChain responses and callback handlers into the unified UsageMetadata format.

The extraction handles multiple LangChain metadata formats:
- response_metadata (newer LangChain versions)
- usage_metadata attribute
- Callback handler usage_metadata dict (from get_usage_metadata_callback)

Usage:
    >>> from karenina.adapters.langchain.usage import extract_langchain_usage
    >>> from langchain_core.callbacks import get_usage_metadata_callback
    >>>
    >>> with get_usage_metadata_callback() as cb:
    ...     response = model.invoke(messages)
    ...     usage = extract_langchain_usage(cb.usage_metadata, model_name="claude-sonnet")
    >>> print(usage.total_tokens)
    150
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from karenina.ports import UsageMetadata

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def extract_langchain_usage(
    source: dict[str, Any] | Any | None,
    *,
    model_name: str | None = None,
) -> UsageMetadata:
    """Extract usage metadata from LangChain sources.

    This function handles multiple formats of LangChain usage data:

    1. **Callback metadata** (from get_usage_metadata_callback):
       {"model-name": {"input_tokens": 100, "output_tokens": 50, ...}}

    2. **Response object** (AIMessage or similar):
       response.response_metadata["token_usage"] or response.usage_metadata

    3. **Direct usage dict**:
       {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

    Args:
        source: One of:
            - Dict from get_usage_metadata_callback().usage_metadata
            - LangChain response object (AIMessage, etc.)
            - Direct usage dict with token counts
            - None (returns zero-usage metadata)
        model_name: Optional model name to include in metadata.
            If not provided and source is callback metadata, extracts from dict key.

    Returns:
        UsageMetadata with extracted token counts. Fields not present
        in source will default to 0 or None.

    Example:
        >>> # From callback handler
        >>> with get_usage_metadata_callback() as cb:
        ...     response = model.invoke(messages)
        ...     usage = extract_langchain_usage(cb.usage_metadata)

        >>> # From response object
        >>> response = await model.ainvoke(messages)
        >>> usage = extract_langchain_usage(response, model_name="gpt-4")

        >>> # From direct dict
        >>> data = {"input_tokens": 100, "output_tokens": 50}
        >>> usage = extract_langchain_usage(data)
    """
    if source is None:
        return UsageMetadata(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            model=model_name,
        )

    # Try to extract usage data based on source type
    usage_data: dict[str, Any] = {}
    extracted_model: str | None = model_name

    if isinstance(source, dict):
        # Check if this is callback metadata format {"model-name": {...}}
        # by looking for nested dicts with token keys
        for key, value in source.items():
            if isinstance(value, dict) and ("input_tokens" in value or "prompt_tokens" in value):
                # Callback metadata format - key is model name
                if extracted_model is None:
                    extracted_model = key
                usage_data = value
                break

        # If no nested format found, assume direct usage dict
        if not usage_data:
            usage_data = source

    else:
        # Response object - try various attributes
        usage_data = _extract_from_response(source)

    return _build_usage_metadata(usage_data, extracted_model)


def extract_usage_from_response(response: Any, *, model_name: str | None = None) -> UsageMetadata:
    """Extract usage metadata from a LangChain response object.

    This is a specialized version of extract_langchain_usage for response objects.
    It extracts usage from response_metadata or usage_metadata attributes.

    Args:
        response: LangChain response object (AIMessage, etc.)
        model_name: Optional model name to include in metadata.

    Returns:
        UsageMetadata with extracted token counts.

    Example:
        >>> response = await model.ainvoke(messages)
        >>> usage = extract_usage_from_response(response, model_name="claude-sonnet")
    """
    usage_data = _extract_from_response(response)
    return _build_usage_metadata(usage_data, model_name)


def extract_usage_cumulative(
    messages: list[BaseMessage] | list[Any],
    *,
    model_name: str | None = None,
) -> UsageMetadata:
    """Extract cumulative usage metadata from a list of messages.

    This is useful for agent executions where usage accumulates across
    multiple LLM calls. It sums usage from all AIMessage objects in the list.

    Args:
        messages: List of LangChain messages (typically from agent execution).
        model_name: Optional model name to include in metadata.

    Returns:
        UsageMetadata with cumulative token counts across all messages.

    Example:
        >>> result = await agent.ainvoke({"messages": [...]})
        >>> messages = result["messages"]
        >>> usage = extract_usage_cumulative(messages, model_name="claude-sonnet")
    """
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_creation = 0

    try:
        from langchain_core.messages import AIMessage
    except ImportError:
        # LangChain not available - return empty usage
        return UsageMetadata(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            model=model_name,
        )

    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue

        usage_data = _extract_from_response(msg)
        if usage_data:
            total_input += int(usage_data.get("input_tokens") or usage_data.get("prompt_tokens") or 0)
            total_output += int(usage_data.get("output_tokens") or usage_data.get("completion_tokens") or 0)

            cache_read = usage_data.get("cache_read_input_tokens") or usage_data.get("cache_read_tokens")
            cache_creation = usage_data.get("cache_creation_input_tokens") or usage_data.get("cache_creation_tokens")

            if cache_read:
                total_cache_read += int(cache_read)
            if cache_creation:
                total_cache_creation += int(cache_creation)

    return UsageMetadata(
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_input + total_output,
        cache_read_tokens=total_cache_read if total_cache_read else None,
        cache_creation_tokens=total_cache_creation if total_cache_creation else None,
        model=model_name,
    )


def _extract_from_response(response: Any) -> dict[str, Any]:
    """Extract usage dict from a response object.

    Handles both response_metadata (newer) and usage_metadata attributes.
    """
    usage_data: dict[str, Any] = {}

    # Try response_metadata first (newer LangChain versions)
    if hasattr(response, "response_metadata") and response.response_metadata:
        metadata = response.response_metadata
        # Anthropic/OpenAI style: token_usage or usage
        usage_data = metadata.get("token_usage") or metadata.get("usage") or {}

    # Fallback to usage_metadata attribute
    if not usage_data and hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        if isinstance(um, dict):
            usage_data = um
        else:
            # UsageMetadata object - convert to dict
            usage_data = {
                "input_tokens": getattr(um, "input_tokens", 0),
                "output_tokens": getattr(um, "output_tokens", 0),
                "cache_read_input_tokens": getattr(um, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(um, "cache_creation_input_tokens", None),
            }

    return usage_data


def _build_usage_metadata(usage_data: dict[str, Any], model_name: str | None) -> UsageMetadata:
    """Build UsageMetadata from extracted usage dict."""
    if not usage_data:
        return UsageMetadata(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            model=model_name,
        )

    input_tokens = int(usage_data.get("input_tokens") or usage_data.get("prompt_tokens") or 0)
    output_tokens = int(usage_data.get("output_tokens") or usage_data.get("completion_tokens") or 0)
    total_tokens = int(usage_data.get("total_tokens") or (input_tokens + output_tokens))

    # Extract cache tokens if available (Anthropic)
    cache_read = usage_data.get("cache_read_input_tokens") or usage_data.get("cache_read_tokens")
    cache_creation = usage_data.get("cache_creation_input_tokens") or usage_data.get("cache_creation_tokens")

    # Note: cost_usd is not provided by LangChain directly
    # It would need to be calculated separately based on model pricing
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_read_tokens=int(cache_read) if cache_read else None,
        cache_creation_tokens=int(cache_creation) if cache_creation else None,
        cost_usd=None,  # LangChain doesn't provide cost directly
        model=model_name,
    )
