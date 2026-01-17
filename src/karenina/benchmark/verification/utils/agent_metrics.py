"""Agent execution metrics extraction utilities.

This module provides functions for extracting metrics from LangGraph agent responses,
including iteration counts, tool usage, suspected failures, and middleware metrics.

Functions provided:
- extract_agent_metrics: Extract metrics from agent response
- extract_middleware_metrics: Extract middleware-specific metrics
- TOOL_FAILURE_PATTERNS: Pre-compiled patterns for detecting tool failures
"""

import re
from collections import defaultdict
from typing import Any

# ============================================================================
# Tool Call Failure Detection Patterns
# ============================================================================

# Pre-compiled regex patterns for detecting suspected tool failures in agent traces.
# Compiled at module load time to avoid recompilation on every extract call.
TOOL_FAILURE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Error indicators
    re.compile(r"\berror\b", re.IGNORECASE),
    re.compile(r"\bfailed\b", re.IGNORECASE),
    re.compile(r"\bexception\b", re.IGNORECASE),
    re.compile(r"\btraceback\b", re.IGNORECASE),
    re.compile(r"\bstack\s+trace\b", re.IGNORECASE),
    # HTTP errors
    re.compile(r"\b404\b", re.IGNORECASE),
    re.compile(r"\b500\b", re.IGNORECASE),
    re.compile(r"\b502\b", re.IGNORECASE),
    re.compile(r"\b503\b", re.IGNORECASE),
    re.compile(r"\btimeout\b", re.IGNORECASE),
    # API failures
    re.compile(r"\binvalid\b", re.IGNORECASE),
    re.compile(r"\bunauthorized\b", re.IGNORECASE),
    re.compile(r"\bforbidden\b", re.IGNORECASE),
    re.compile(r"\bnot\s+found\b", re.IGNORECASE),
    re.compile(r"\bcannot\b", re.IGNORECASE),
    re.compile(r"\bunable\s+to\b", re.IGNORECASE),
)


def extract_agent_metrics(response: Any) -> dict[str, Any] | None:
    """
    Extract agent execution metrics from LangGraph agent response.

    This function analyzes agent messages to track:
    - Iterations (AI message cycles)
    - Tool calls (successful tool invocations)
    - Tools used (unique tool names)
    - Per-tool call counts
    - Suspected failed tool calls (tools with error-like output patterns)
    - Middleware-related metrics (LangChain 1.1+)

    Args:
        response: Agent response object from LangGraph (dict with "messages" key)

    Returns:
        Dict with agent metrics:
        - iterations: Number of AI message cycles
        - tool_calls: Total tool invocations
        - tools_used: Sorted list of unique tool names
        - tool_call_counts: Dict mapping tool name to call count
        - suspect_failed_tool_calls: Count of tool calls with error-like patterns
        - suspect_failed_tools: Sorted list of tools with suspected failures
        - model_call_limit_reached: Whether model call limit was hit (middleware)
        - tool_call_limit_reached: Whether tool call limit was hit (middleware)
        - summarization_triggered: Whether summarization middleware ran
        - model_retries: Number of model retry attempts (middleware)
        - tool_retries: Number of tool retry attempts (middleware)
        Returns None if extraction fails
    """
    if not response or not isinstance(response, dict):
        return None

    messages = response.get("messages", [])
    if not messages:
        return None

    # Count iterations (AI message cycles)
    iterations = 0
    tool_calls = 0
    tools_used: set[str] = set()
    tool_call_counts: dict[str, int] = defaultdict(int)  # Per-tool call counts
    suspect_failed_tool_calls = 0
    suspect_failed_tools: set[str] = set()

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
                    tool_call_counts[tool_name] += 1

                # Check for suspected failures in tool output
                is_suspect_failure = False

                # Check content field for error patterns
                content = getattr(msg, "content", None)
                if content and isinstance(content, str):
                    # Test against all failure patterns
                    for pattern in TOOL_FAILURE_PATTERNS:
                        if pattern.search(content):
                            is_suspect_failure = True
                            break

                # Check status field if available (some tool messages have status)
                if not is_suspect_failure:
                    status = getattr(msg, "status", None)
                    if status and isinstance(status, str) and status.lower() in ["error", "failed", "failure"]:
                        is_suspect_failure = True

                # Track suspected failure
                if is_suspect_failure:
                    suspect_failed_tool_calls += 1
                    if tool_name:
                        suspect_failed_tools.add(tool_name)

    # Extract middleware metrics from response metadata if available
    middleware_metrics = extract_middleware_metrics(response)

    return {
        "iterations": iterations,
        "tool_calls": tool_calls,
        "tools_used": sorted(tools_used),  # Sort for deterministic output
        "tool_call_counts": dict(tool_call_counts),  # Per-tool call counts
        "suspect_failed_tool_calls": suspect_failed_tool_calls,
        "suspect_failed_tools": sorted(suspect_failed_tools),  # Sort for deterministic output
        # Middleware metrics (LangChain 1.1+)
        "model_call_limit_reached": middleware_metrics.get("model_call_limit_reached", False),
        "tool_call_limit_reached": middleware_metrics.get("tool_call_limit_reached", False),
        "summarization_triggered": middleware_metrics.get("summarization_triggered", False),
        "model_retries": middleware_metrics.get("model_retries", 0),
        "tool_retries": middleware_metrics.get("tool_retries", 0),
    }


def extract_middleware_metrics(response: Any) -> dict[str, Any]:
    """
    Extract middleware-related metrics from agent response.

    LangChain 1.1 middleware may include metrics in response metadata.
    This function attempts to extract them from various possible locations.

    Args:
        response: Agent response object from LangGraph

    Returns:
        Dict with middleware metrics (defaults to False/0 if not found)
    """
    metrics: dict[str, Any] = {
        "model_call_limit_reached": False,
        "tool_call_limit_reached": False,
        "summarization_triggered": False,
        "model_retries": 0,
        "tool_retries": 0,
    }

    if not response or not isinstance(response, dict):
        return metrics

    # Try to extract from response metadata
    metadata = response.get("metadata", {})
    if isinstance(metadata, dict):
        # Check for limit-related flags
        if metadata.get("model_call_limit_reached"):
            metrics["model_call_limit_reached"] = True
        if metadata.get("tool_call_limit_reached"):
            metrics["tool_call_limit_reached"] = True
        if metadata.get("summarization_triggered"):
            metrics["summarization_triggered"] = True

        # Check for retry counts
        if "model_retries" in metadata:
            metrics["model_retries"] = int(metadata.get("model_retries", 0))
        if "tool_retries" in metadata:
            metrics["tool_retries"] = int(metadata.get("tool_retries", 0))

    # Also check for middleware_stats if present
    middleware_stats = response.get("middleware_stats", {})
    if isinstance(middleware_stats, dict):
        metrics["model_call_limit_reached"] = middleware_stats.get(
            "model_call_limit_reached", metrics["model_call_limit_reached"]
        )
        metrics["tool_call_limit_reached"] = middleware_stats.get(
            "tool_call_limit_reached", metrics["tool_call_limit_reached"]
        )
        metrics["summarization_triggered"] = middleware_stats.get(
            "summarization_triggered", metrics["summarization_triggered"]
        )
        metrics["model_retries"] = middleware_stats.get("model_retries", metrics["model_retries"])
        metrics["tool_retries"] = middleware_stats.get("tool_retries", metrics["tool_retries"])

    return metrics
