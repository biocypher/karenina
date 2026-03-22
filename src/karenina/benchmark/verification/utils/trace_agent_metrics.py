"""Agent execution metrics extraction utilities.

This module provides functions for extracting metrics from port Message lists,
including iteration counts, tool usage, and suspected failures.

Functions provided:
- extract_agent_metrics_from_messages: Extract metrics from port Message list (canonical)
- TOOL_FAILURE_PATTERNS: Pre-compiled patterns for detecting tool failures
"""

import re
from collections import defaultdict
from typing import Any

from karenina.ports.messages import (
    Message,
    Role,
    ToolResultContent,
)

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


def extract_agent_metrics_from_messages(messages: list[Message]) -> dict[str, Any]:
    """
    Extract agent execution metrics from a list of port Message objects.

    This is the canonical function for extracting tool metrics from any adapter
    that provides trace_messages on AgentResult. It works with the unified
    Message type rather than LangChain-specific or raw dict formats.

    Tracks:
    - Iterations (assistant message cycles)
    - Tool calls (tool result messages)
    - Tools used (unique tool names from assistant tool_calls)
    - Per-tool call counts
    - Suspected failed tool calls (is_error flag or error-like content patterns)
    - Suspected failed tool names (resolved via tool_use_id -> tool_name mapping)

    Args:
        messages: List of port Message objects (from AgentResult.trace_messages)

    Returns:
        Dict with agent metrics matching the documented schema in result_components.
    """
    iterations = 0
    tool_calls = 0
    tools_used: set[str] = set()
    tool_call_counts: dict[str, int] = defaultdict(int)
    suspect_failed_tool_calls = 0
    suspect_failed_tools: set[str] = set()

    # Build a tool_use_id -> tool_name mapping from assistant messages
    # so we can resolve tool names for failed tool results
    tool_id_to_name: dict[str, str] = {}

    for msg in messages:
        if msg.role == Role.ASSISTANT:
            iterations += 1
            for tc in msg.tool_calls:
                tools_used.add(tc.name)
                tool_call_counts[tc.name] += 1
                tool_id_to_name[tc.id] = tc.name

        elif msg.role == Role.TOOL:
            for content_block in msg.content:
                if isinstance(content_block, ToolResultContent):
                    tool_calls += 1

                    # Check for suspected failures
                    is_suspect = content_block.is_error
                    if not is_suspect and content_block.content:
                        for pattern in TOOL_FAILURE_PATTERNS:
                            if pattern.search(content_block.content):
                                is_suspect = True
                                break

                    if is_suspect:
                        suspect_failed_tool_calls += 1
                        # Resolve tool name from the id mapping
                        tool_name = tool_id_to_name.get(content_block.tool_use_id)
                        if tool_name:
                            suspect_failed_tools.add(tool_name)

    return {
        "iterations": iterations,
        "tool_calls": tool_calls,
        "tools_used": sorted(tools_used),
        "tool_call_counts": dict(tool_call_counts),
        "suspect_failed_tool_calls": suspect_failed_tool_calls,
        "suspect_failed_tools": sorted(suspect_failed_tools),
    }
