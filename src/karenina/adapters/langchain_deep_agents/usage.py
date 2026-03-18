"""Usage metadata extraction from LangGraph agent results.

Extracts token counts from AIMessage.response_metadata, which is populated
by LangChain's LLM integrations for most providers.
"""

from __future__ import annotations

import logging
from typing import Any

from karenina.ports.usage import UsageMetadata

logger = logging.getLogger(__name__)


def extract_deep_agents_usage(
    messages: list[Any],
    model: str | None = None,
) -> UsageMetadata:
    """Extract aggregated usage metadata from LangGraph messages.

    Sums token counts across all AIMessage instances in the conversation.
    Token counts come from AIMessage.usage_metadata (preferred) or
    AIMessage.response_metadata.token_usage (fallback).

    Args:
        messages: List of LangGraph BaseMessage objects.
        model: Model name to include in usage metadata.

    Returns:
        Aggregated UsageMetadata for the entire agent run.
    """
    from langchain_core.messages import AIMessage

    total_input = 0
    total_output = 0

    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue

        usage_meta = getattr(msg, "usage_metadata", None)
        if usage_meta and isinstance(usage_meta, dict):
            total_input += usage_meta.get("input_tokens", 0)
            total_output += usage_meta.get("output_tokens", 0)
            continue

        resp_meta = getattr(msg, "response_metadata", None)
        if resp_meta and isinstance(resp_meta, dict):
            token_usage = resp_meta.get("token_usage", {})
            if isinstance(token_usage, dict):
                total_input += token_usage.get("prompt_tokens", 0)
                total_output += token_usage.get("completion_tokens", 0)

    return UsageMetadata(
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_input + total_output,
        model=model,
    )


def extract_actual_model(messages: list[Any]) -> str | None:
    """Extract actual model name from the last AIMessage.

    Checks response_metadata for model_name or model fields,
    searching from the end of the message list (most recent first).

    Args:
        messages: List of LangGraph BaseMessage objects.

    Returns:
        Model name string, or None if not found.
    """
    from langchain_core.messages import AIMessage

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            resp_meta = getattr(msg, "response_metadata", None)
            if resp_meta and isinstance(resp_meta, dict):
                model_name = resp_meta.get("model_name") or resp_meta.get("model")
                if model_name:
                    return str(model_name)
    return None
