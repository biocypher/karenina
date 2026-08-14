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


def collapse_partial_assistant_messages(messages: list[Any]) -> list[Any]:
    """Merge multiple partial ``AssistantMessage`` events into one per turn.

    The CLI subprocess emits one ``AssistantMessage`` per ``content_block_stop``
    event, not per LLM turn. A single turn that produces a thinking block plus
    a text block plus a tool-use block surfaces as three ``AssistantMessage``
    objects: each carrying only the single block that just finished, all
    sharing the same ``message_id`` and the same ``message_start``-state usage
    dict. The blocks do NOT accumulate across emissions — every emission
    contains exactly one block.

    Without merging:
      * ``trace_messages`` (derived from the messages) has 2-3× the count
        of real LLM turns; ``agent_metrics.iterations`` is inflated.
      * ``extract_sdk_usage_from_messages`` double- or triple-counts input
        tokens on a wall-clock timeout where ``ResultMessage`` is absent
        (all emissions for one turn share the same input-token snapshot).

    Naive de-duplication (keeping only the last per ``message_id``) would
    drop the earlier blocks — typically losing thinking and text content
    while preserving only the final tool-use block. To avoid that, this
    helper merges all blocks for a given ``message_id`` into the LAST
    emission's ``content`` list (preserving source order), then drops the
    earlier emissions. The surviving emission keeps its own ``usage``,
    ``model``, ``stop_reason``, etc. — these are stable across partials
    sharing a ``message_id``.

    AssistantMessages without a ``message_id`` and all non-``AssistantMessage``
    entries are passed through unchanged in their original positions. The
    relative order of distinct turns is preserved: each merged turn appears
    at the position of its LAST partial emission, so the resulting sequence
    still interleaves correctly with surrounding ``StreamEvent`` /
    ``UserMessage`` / ``ResultMessage`` records.

    Args:
        messages: Mixed collection from ``ClaudeSDKClient.receive_response()``.

    Returns:
        New list with at most one ``AssistantMessage`` per ``message_id``,
        each carrying the concatenated content of all its partial emissions.
    """
    try:
        from claude_agent_sdk import AssistantMessage
    except ImportError:
        return list(messages)

    # Pass 1: bucket content blocks by message_id and find the last
    # emission's index for each id.
    blocks_by_msgid: dict[str, list[Any]] = {}
    last_index_by_msgid: dict[str, int] = {}
    for i, msg in enumerate(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        mid = getattr(msg, "message_id", None)
        if mid is None:
            continue
        blocks_by_msgid.setdefault(mid, []).extend(msg.content or [])
        last_index_by_msgid[mid] = i

    if not last_index_by_msgid:
        return list(messages)

    # Pass 2: rewrite the last emission for each group with the merged
    # block list, then emit each entry, dropping earlier partials.
    result: list[Any] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, AssistantMessage):
            mid = getattr(msg, "message_id", None)
            if mid is not None:
                if last_index_by_msgid[mid] != i:
                    continue
                msg.content = blocks_by_msgid[mid]
        result.append(msg)
    return result


def backfill_assistant_output_tokens(messages: list[Any]) -> None:
    """Backfill ``output_tokens`` on ``AssistantMessage`` records in-place.

    On backends where ``output_tokens`` is only finalised in the streaming
    ``message_delta`` event (e.g. vLLM, sglang, and any non-canonical Anthropic
    shim), ``AssistantMessage.usage`` reflects only ``message_start`` state and
    reports ``output_tokens=0``. When the agent run is cancelled mid-stream
    (wall-clock timeout) no ``ResultMessage`` is emitted, so per-message
    aggregation is the only signal left — and it under-reports output tokens
    to zero.

    With ``ClaudeAgentOptions(include_partial_messages=True)`` the SDK exposes
    the raw stream events as ``StreamEvent`` objects. This helper extracts the
    final ``output_tokens`` from each ``message_delta`` event, correlates it
    with the parent ``AssistantMessage`` by ``message_id``, and patches the
    usage dict in place. Downstream consumers (trace conversion, aggregate
    summation) see corrected per-call usage without further changes.

    The patch is idempotent and never decreases a non-zero value: if the SDK
    already reported a real output count (e.g. canonical Anthropic), the
    delta value is ignored.

    Args:
        messages: Mixed collection of SDK message objects as collected from
            ``ClaudeSDKClient.receive_response()``. ``StreamEvent`` objects
            must be present for backfill to happen — that requires the
            ``include_partial_messages=True`` option.
    """
    try:
        from claude_agent_sdk import AssistantMessage, StreamEvent
    except ImportError:
        return

    # Walk the event stream once to map message_id -> final output_tokens.
    # In Anthropic's streaming protocol every message_delta belongs to the
    # most recent message_start, so a single rolling pointer is enough.
    delta_output_by_msgid: dict[str, int] = {}
    current_msgid: str | None = None
    for msg in messages:
        if not isinstance(msg, StreamEvent):
            continue
        event = getattr(msg, "event", None) or {}
        event_type = event.get("type")
        if event_type == "message_start":
            current_msgid = (event.get("message") or {}).get("id")
        elif event_type == "message_delta" and current_msgid is not None:
            usage = event.get("usage") or {}
            output_tokens = usage.get("output_tokens")
            if output_tokens is not None:
                delta_output_by_msgid[current_msgid] = int(output_tokens)

    if not delta_output_by_msgid:
        return

    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        msg_id = getattr(msg, "message_id", None)
        if msg_id is None or msg_id not in delta_output_by_msgid:
            continue
        current_usage = msg.usage or {}
        existing_output = int(current_usage.get("output_tokens", 0) or 0)
        if existing_output != 0:
            # Don't clobber a real non-zero value reported by canonical
            # Anthropic endpoints.
            continue
        patched = dict(current_usage)
        patched["output_tokens"] = delta_output_by_msgid[msg_id]
        msg.usage = patched


def extract_sdk_usage_from_messages(
    messages: list[Any],
    model: str | None = None,
) -> UsageMetadata:
    """Aggregate per-message usage from SDK AssistantMessage objects.

    The SDK's ``ResultMessage`` is only emitted on a clean loop exit, so when
    an agent run is cancelled mid-stream (e.g. ``asyncio.wait_for`` timeout)
    no aggregate usage is available. Individual ``AssistantMessage`` objects
    collected before the cancellation still carry their per-call ``usage``
    dict though, so we can reconstruct the run-level totals by summing them.

    This mirrors the per-message aggregation that
    ``extract_deep_agents_usage`` performs for LangChain AIMessage objects.

    Args:
        messages: List of SDK message objects collected during the run.
            Only ``AssistantMessage`` instances are considered; other types
            (UserMessage, ResultMessage, etc.) are skipped.
        model: Optional model name to record on the result.

    Returns:
        ``UsageMetadata`` with summed tokens. Cache fields are summed only
        when at least one message reports them, otherwise left ``None``.

    Notes:
        Cost is intentionally NOT estimated here. SDK cost figures live on
        the ``ResultMessage`` (``total_cost_usd``) which is absent in the
        partial-trace scenarios this helper exists to handle.
    """
    try:
        from claude_agent_sdk import AssistantMessage
    except ImportError:
        return UsageMetadata(model=model)

    total_input = 0
    total_output = 0
    cache_read_total = 0
    cache_creation_total = 0
    saw_cache_read = False
    saw_cache_creation = False

    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        usage_data: dict[str, Any] | None = getattr(msg, "usage", None)
        if not usage_data or not isinstance(usage_data, dict):
            continue

        total_input += int(usage_data.get("input_tokens", 0) or 0)
        total_output += int(usage_data.get("output_tokens", 0) or 0)

        cache_read = usage_data.get("cache_read_input_tokens")
        if cache_read is not None:
            cache_read_total += int(cache_read)
            saw_cache_read = True

        cache_creation = usage_data.get("cache_creation_input_tokens")
        if cache_creation is not None:
            cache_creation_total += int(cache_creation)
            saw_cache_creation = True

    return UsageMetadata(
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_input + total_output,
        cache_read_tokens=cache_read_total if saw_cache_read else None,
        cache_creation_tokens=cache_creation_total if saw_cache_creation else None,
        model=model,
    )
