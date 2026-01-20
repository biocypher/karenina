"""Usage metadata for tracking token consumption and costs.

This module defines the UsageMetadata dataclass used to track
token usage and cost information across LLM invocations.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class UsageMetadata:
    """Token usage and cost tracking for LLM invocations.

    This dataclass captures usage metrics from LLM calls in a unified format
    that works across different backends (LangChain, Claude Agent SDK, etc.).

    Attributes:
        input_tokens: Number of tokens in the input/prompt.
        output_tokens: Number of tokens in the generated output.
        total_tokens: Total tokens (input + output).
        cost_usd: Cost in USD for this invocation (if available from provider).
        cache_read_tokens: Tokens read from Anthropic's prompt cache (optional).
        cache_creation_tokens: Tokens used to create Anthropic's prompt cache (optional).
        model: The model that generated this usage (optional).

    Example:
        >>> usage = UsageMetadata(
        ...     input_tokens=100,
        ...     output_tokens=50,
        ...     total_tokens=150,
        ...     cost_usd=0.003,
        ...     model="claude-sonnet-4-20250514"
        ... )
        >>> usage.total_tokens
        150
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert usage metadata to a dictionary.

        Returns:
            Dict with all usage fields. Core token counts are always included.
            Optional fields (cost_usd, cache tokens, model) are included only
            when they have non-None values.

        Example:
            >>> usage = UsageMetadata(input_tokens=100, output_tokens=50, total_tokens=150)
            >>> usage.to_dict()
            {'input_tokens': 100, 'output_tokens': 50, 'total_tokens': 150}
        """
        result: dict[str, Any] = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cost_usd is not None:
            result["cost_usd"] = self.cost_usd
        if self.cache_read_tokens is not None:
            result["cache_read_tokens"] = self.cache_read_tokens
        if self.cache_creation_tokens is not None:
            result["cache_creation_tokens"] = self.cache_creation_tokens
        if self.model is not None:
            result["model"] = self.model
        return result
