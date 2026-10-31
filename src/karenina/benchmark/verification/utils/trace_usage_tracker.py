"""
Usage tracking utilities for LLM calls.

Provides classes to track and aggregate token usage and agent metrics across
verification stages. Works with any adapter that returns usage metadata in
the expected format.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from karenina.ports.usage import UsageMetadata as PortUsageMetadata

# Optional per-call counts that are summed per stage and reported only when
# at least one call in the stage set them, so existing summaries keep their keys.
_OPTIONAL_COUNT_KEYS: tuple[str, ...] = ("reasoning_tokens", "web_search_queries")


@dataclass
class UsageMetadata:
    """
    Metadata for a single LLM call.

    Attributes:
        model_name: Model identifier (e.g., "gpt-4.1-mini-2025-04-14")
        input_tokens: Number of tokens in the input/prompt
        output_tokens: Number of tokens in the output/completion
        total_tokens: Total tokens (input + output)
        input_token_details: Additional input token details (audio, cache_read)
        output_token_details: Additional output token details (audio, reasoning)
        reasoning_tokens: Thinking tokens reported separately by the adapter, or None
        web_search_queries: Provider-side web searches made by the call, or None
    """

    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_token_details: dict[str, int] = field(default_factory=dict)
    output_token_details: dict[str, int] = field(default_factory=dict)
    reasoning_tokens: int | None = None
    web_search_queries: int | None = None

    @classmethod
    def from_callback_metadata(
        cls, callback_metadata: dict[str, Any], fallback_model_name: str = "unknown"
    ) -> "UsageMetadata":
        """
        Create UsageMetadata from callback metadata.

        Accepts two formats:

        1. **Model-keyed** (from LangChain callbacks):
           ``{"model-name": {"input_tokens": 11, "output_tokens": 20, ...}}``

        2. **Flat** (from ``dataclasses.asdict(ports.UsageMetadata)``):
           ``{"input_tokens": 11, "output_tokens": 20, "total_tokens": 31, ...}``

        Args:
            callback_metadata: Dict with usage data in either format.
            fallback_model_name: Model name to use when not present in metadata.

        Returns:
            UsageMetadata instance
        """
        if not callback_metadata:
            return cls(model_name=fallback_model_name)

        if not isinstance(callback_metadata, dict):
            return cls(model_name=fallback_model_name)

        # Detect flat format: top-level "input_tokens" key whose value is an int
        if "input_tokens" in callback_metadata and isinstance(callback_metadata.get("input_tokens"), int):
            # Prefer fallback_model_name (caller-provided ModelIdentity.display_string)
            # over bare model name from adapter, which lacks the interface prefix.
            if fallback_model_name != "unknown":
                model_name = fallback_model_name
            else:
                model_name = callback_metadata.get("model") or fallback_model_name
            return cls(
                model_name=model_name,
                input_tokens=callback_metadata.get("input_tokens", 0),
                output_tokens=callback_metadata.get("output_tokens", 0),
                total_tokens=callback_metadata.get("total_tokens", 0),
                input_token_details=callback_metadata.get("input_token_details", {}),
                output_token_details=callback_metadata.get("output_token_details", {}),
                reasoning_tokens=callback_metadata.get("reasoning_tokens"),
                web_search_queries=callback_metadata.get("web_search_queries"),
            )

        # Model-keyed format: iterate to find first nested dict entry
        for model_name, metadata in callback_metadata.items():
            if isinstance(metadata, dict):
                return cls(
                    model_name=model_name,
                    input_tokens=metadata.get("input_tokens", 0),
                    output_tokens=metadata.get("output_tokens", 0),
                    total_tokens=metadata.get("total_tokens", 0),
                    input_token_details=metadata.get("input_token_details", {}),
                    output_token_details=metadata.get("output_token_details", {}),
                    reasoning_tokens=metadata.get("reasoning_tokens"),
                    web_search_queries=metadata.get("web_search_queries"),
                )

        # Fallback for unexpected format
        return cls(model_name=fallback_model_name)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        ``reasoning_tokens`` and ``web_search_queries`` are included only when set.
        """
        result: dict[str, Any] = {
            "model": self.model_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_token_details": self.input_token_details,
            "output_token_details": self.output_token_details,
        }
        for key in _OPTIONAL_COUNT_KEYS:
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


def usage_to_callback_metadata(model_name: str, usage: "PortUsageMetadata") -> dict[str, dict[str, Any]]:
    """Convert a port UsageMetadata into the model-keyed dict UsageTracker accepts.

    Builds the same ``inner_usage`` dict the generate_answer stage builds for
    the answering model (token counts, cost and cache keys), plus
    ``reasoning_tokens`` and ``web_search_queries`` when the adapter set them.

    Args:
        model_name: Model display string used as the dict key.
        usage: Usage reported by an adapter for one call or agent run.

    Returns:
        ``{model_name: inner_usage}`` suitable for :meth:`UsageTracker.track_call`.
    """
    inner_usage: dict[str, Any] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }
    if usage.cost_usd is not None:
        inner_usage["cost_usd"] = usage.cost_usd
    if usage.cache_read_tokens is not None:
        inner_usage["cache_read_input_tokens"] = usage.cache_read_tokens
    if usage.cache_creation_tokens is not None:
        inner_usage["cache_creation_input_tokens"] = usage.cache_creation_tokens
    if usage.reasoning_tokens is not None:
        inner_usage["reasoning_tokens"] = usage.reasoning_tokens
    if usage.web_search_queries is not None:
        inner_usage["web_search_queries"] = usage.web_search_queries
    return {model_name: inner_usage}


class UsageTracker:
    """
    Tracks and aggregates LLM usage metadata across multiple calls and stages.

    Usage:
        tracker = UsageTracker()

        # Track calls
        tracker.track_call("answer_generation", "gpt-4.1-mini", callback_metadata)
        tracker.track_call("parsing", "gpt-4.1-mini", callback_metadata)

        # Set agent metrics (if agent used)
        tracker.set_agent_metrics(extracted_metrics)

        # Get summaries
        answer_summary = tracker.get_stage_summary("answer_generation")
        total_summary = tracker.get_total_summary()
        agent_metrics = tracker.get_agent_metrics()
    """

    def __init__(self) -> None:
        """Initialize empty tracking structures."""
        # Stage name -> list of UsageMetadata
        self._stage_calls: dict[str, list[UsageMetadata]] = {}
        # Agent metrics (only populated if agent used)
        self._agent_metrics: dict[str, Any] | None = None

    def track_call(
        self,
        stage_name: str,
        model_name: str,
        callback_metadata: dict[str, Any],
    ) -> None:
        """
        Record a single LLM call.

        Args:
            stage_name: Name of verification stage (e.g., "answer_generation")
            model_name: Model identifier for fallback if not in metadata
            callback_metadata: Dict keyed by model name with usage data.
                See UsageMetadata.from_callback_metadata() for expected format.
        """
        if not callback_metadata:
            # Create minimal metadata if callback returned nothing
            metadata = UsageMetadata(model_name=model_name)
        else:
            metadata = UsageMetadata.from_callback_metadata(callback_metadata, fallback_model_name=model_name)

        if stage_name not in self._stage_calls:
            self._stage_calls[stage_name] = []

        self._stage_calls[stage_name].append(metadata)

    def get_stage_summary(self, stage_name: str) -> dict[str, Any] | None:
        """
        Get aggregated metadata for a single stage.

        Args:
            stage_name: Name of verification stage

        Returns:
            Aggregated metadata dict or None if stage not tracked
        """
        if stage_name not in self._stage_calls:
            return None

        calls = self._stage_calls[stage_name]
        if not calls:
            return None

        # Aggregate across all calls in this stage
        total_input = sum(c.input_tokens for c in calls)
        total_output = sum(c.output_tokens for c in calls)
        total_tokens = sum(c.total_tokens for c in calls)

        # Use model name from first call (should be consistent within stage)
        model_name = calls[0].model_name

        # Aggregate token details (sum across calls)
        input_details: dict[str, int] = {}
        output_details: dict[str, int] = {}

        for call in calls:
            for key, val in call.input_token_details.items():
                input_details[key] = input_details.get(key, 0) + val
            for key, val in call.output_token_details.items():
                output_details[key] = output_details.get(key, 0) + val

        summary: dict[str, Any] = {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
            "model": model_name,
            "input_token_details": input_details if input_details else {"audio": 0, "cache_read": 0},
            "output_token_details": output_details if output_details else {"audio": 0, "reasoning": 0},
        }
        for key in _OPTIONAL_COUNT_KEYS:
            values = [getattr(c, key) for c in calls if getattr(c, key) is not None]
            if values:
                summary[key] = sum(values)
        return summary

    def get_total_summary(self) -> dict[str, dict[str, Any]]:
        """
        Get aggregated metadata for all stages plus overall total.

        Returns:
            Dict with per-stage summaries and a "total" entry aggregating all stages
        """
        result = {}

        # Per-stage summaries
        for stage_name in self._stage_calls:
            summary = self.get_stage_summary(stage_name)
            if summary:
                result[stage_name] = summary

        # Overall total
        if result:
            total_input = sum(s["input_tokens"] for s in result.values())
            total_output = sum(s["output_tokens"] for s in result.values())
            total_tokens = sum(s["total_tokens"] for s in result.values())

            total: dict[str, Any] = {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_tokens,
            }
            for key in _OPTIONAL_COUNT_KEYS:
                values = [s[key] for s in result.values() if key in s]
                if values:
                    total[key] = sum(values)
            result["total"] = total

        return result

    def set_agent_metrics(self, agent_metrics: dict[str, Any]) -> None:
        """
        Directly set agent metrics (when already extracted).

        Args:
            agent_metrics: Pre-extracted agent metrics dict with keys:
                - iterations: Number of agent think-act cycles
                - tool_calls: Total tool invocations
                - tools_used: List of unique tool names
        """
        self._agent_metrics = agent_metrics

    def get_agent_metrics(self) -> dict[str, Any] | None:
        """
        Get agent execution metrics if available.

        Returns:
            Agent metrics dict or None if no agent was used
        """
        return self._agent_metrics
