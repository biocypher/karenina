"""Token usage tracking for native tool-calling agents.

This module provides the NativeUsageTracker class that extracts and accumulates
token usage from native OpenAI and Anthropic SDK responses, normalizing them
to a LangChain-compatible format.
"""

from __future__ import annotations

from typing import Any


class NativeUsageTracker:
    """Tracks token usage from native SDK responses.

    This class extracts token usage information from OpenAI and Anthropic
    responses and accumulates them across multiple API calls during an
    agent loop. The accumulated usage is normalized to a format compatible
    with LangChain's usage metadata callback.

    Attributes:
        model_name: The model name/identifier for the usage metadata key
        accumulated_usage: Dict tracking accumulated token counts
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the usage tracker.

        Args:
            model_name: Model name to use as key in usage metadata
        """
        self.model_name = model_name
        self.accumulated_usage: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_token_details": {"audio": 0, "cache_read": 0},
            "output_token_details": {"audio": 0, "reasoning": 0},
        }

    def track_openai_response(self, response: Any) -> None:
        """Extract and accumulate usage from an OpenAI response.

        Handles the OpenAI ChatCompletion response format:
        - response.usage.prompt_tokens -> input_tokens
        - response.usage.completion_tokens -> output_tokens
        - response.usage.total_tokens -> total_tokens
        - response.usage.prompt_tokens_details.cached_tokens -> cache_read (if available)

        Args:
            response: OpenAI ChatCompletion response object
        """
        if not hasattr(response, "usage") or response.usage is None:
            return

        usage = response.usage

        # Accumulate basic token counts
        self.accumulated_usage["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        self.accumulated_usage["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        self.accumulated_usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0

        # Handle prompt token details (cached tokens)
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            details = usage.prompt_tokens_details
            if hasattr(details, "cached_tokens"):
                self.accumulated_usage["input_token_details"]["cache_read"] += details.cached_tokens or 0

        # Handle completion token details (reasoning tokens for o1/o3 models)
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            details = usage.completion_tokens_details
            if hasattr(details, "reasoning_tokens"):
                self.accumulated_usage["output_token_details"]["reasoning"] += details.reasoning_tokens or 0

    def track_anthropic_response(self, response: Any) -> None:
        """Extract and accumulate usage from an Anthropic response.

        Handles the Anthropic Message response format:
        - response.usage.input_tokens -> input_tokens
        - response.usage.output_tokens -> output_tokens
        - response.usage.cache_read_input_tokens -> cache_read (if available)

        Args:
            response: Anthropic Message response object
        """
        if not hasattr(response, "usage") or response.usage is None:
            return

        usage = response.usage

        # Accumulate basic token counts
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0

        self.accumulated_usage["input_tokens"] += input_tokens
        self.accumulated_usage["output_tokens"] += output_tokens
        self.accumulated_usage["total_tokens"] += input_tokens + output_tokens

        # Handle cache read tokens (Anthropic prompt caching)
        if hasattr(usage, "cache_read_input_tokens"):
            self.accumulated_usage["input_token_details"]["cache_read"] += usage.cache_read_input_tokens or 0

        # Handle cache creation tokens (if we want to track them)
        # Anthropic also has cache_creation_input_tokens but we don't track it currently

    def get_usage_metadata(self) -> dict[str, Any]:
        """Return accumulated usage in LangChain callback format.

        This format is compatible with karenina's UsageTracker.track_call() method
        and the existing token tracking infrastructure.

        Returns:
            Dict with model_name as key and accumulated usage as value
        """
        return {self.model_name: self.accumulated_usage.copy()}

    def get_usage_dict(self) -> dict[str, int]:
        """Return simple usage dict for NativeAgentResponse.

        Returns a simplified dict with just the core token counts,
        suitable for the NativeAgentResponse.usage field.

        Returns:
            Dict with input_tokens, output_tokens, total_tokens
        """
        return {
            "input_tokens": self.accumulated_usage["input_tokens"],
            "output_tokens": self.accumulated_usage["output_tokens"],
            "total_tokens": self.accumulated_usage["total_tokens"],
        }

    def reset(self) -> None:
        """Reset accumulated usage to zero.

        Useful for starting a fresh tracking session without creating
        a new tracker instance.
        """
        self.accumulated_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_token_details": {"audio": 0, "cache_read": 0},
            "output_token_details": {"audio": 0, "reasoning": 0},
        }
