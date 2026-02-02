"""
Usage tracking utilities for LLM calls.

Provides classes to track and aggregate token usage and agent metrics across
verification stages. Works with any adapter that returns usage metadata in
the expected format.
"""

from dataclasses import dataclass, field
from typing import Any


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
    """

    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_token_details: dict[str, int] = field(default_factory=dict)
    output_token_details: dict[str, int] = field(default_factory=dict)

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
            return cls(
                model_name=callback_metadata.get("model") or fallback_model_name,
                input_tokens=callback_metadata.get("input_tokens", 0),
                output_tokens=callback_metadata.get("output_tokens", 0),
                total_tokens=callback_metadata.get("total_tokens", 0),
                input_token_details=callback_metadata.get("input_token_details", {}),
                output_token_details=callback_metadata.get("output_token_details", {}),
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
                )

        # Fallback for unexpected format
        return cls(model_name=fallback_model_name)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_token_details": self.input_token_details,
            "output_token_details": self.output_token_details,
        }


class UsageTracker:
    """
    Tracks and aggregates LLM usage metadata across multiple calls and stages.

    Usage:
        tracker = UsageTracker()

        # Track calls
        tracker.track_call("answer_generation", "gpt-4.1-mini", callback_metadata)
        tracker.track_call("parsing", "gpt-4.1-mini", callback_metadata)

        # Track agent metrics (if agent used)
        tracker.track_agent_metrics(agent_response)

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

    def track_agent_metrics(self, response: Any) -> None:
        """
        Extract and store agent execution metrics from response.

        Note: This method is used by stages that need to extract metrics from
        raw response objects. For pre-extracted metrics, use set_agent_metrics().

        Args:
            response: Agent response dict with "messages" key containing message objects.
        """
        if not response or not isinstance(response, dict):
            return

        messages = response.get("messages", [])
        if not messages:
            return

        # Count iterations (AI message cycles)
        iterations = 0
        tool_calls = 0
        tools_used = set()

        for msg in messages:
            # Check message type
            msg_type = getattr(msg, "__class__", None)
            if msg_type:
                type_name = msg_type.__name__

                # Count AI messages as iterations
                if type_name == "AIMessage":
                    iterations += 1

                # Count tool messages and extract tool names
                elif type_name == "ToolMessage":
                    tool_calls += 1
                    # Extract tool name from ToolMessage
                    tool_name = getattr(msg, "name", None)
                    if tool_name:
                        tools_used.add(tool_name)

        self._agent_metrics = {
            "iterations": iterations,
            "tool_calls": tool_calls,
            "tools_used": sorted(tools_used),  # Sort for deterministic output
        }

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

        return {
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
            "model": model_name,
            "input_token_details": input_details if input_details else {"audio": 0, "cache_read": 0},
            "output_token_details": output_details if output_details else {"audio": 0, "reasoning": 0},
        }

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

            result["total"] = {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_tokens,
            }

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
