"""Availability check for LangChain Deep Agents adapter.

Verifies that the deepagents package is installed. No fallback interface
is provided because Deep Agents' natively agentic behavior cannot be
meaningfully approximated by the scaffolded LangChain adapter.
"""

from __future__ import annotations

from karenina.adapters.registry import AdapterAvailability


def check_deep_agents_available() -> AdapterAvailability:
    """Check if the deepagents package is installed.

    Returns:
        AdapterAvailability with status and installation instructions.
    """
    try:
        import deepagents  # noqa: F401

        return AdapterAvailability(
            available=True,
            reason="deepagents package is installed",
        )
    except ImportError:
        return AdapterAvailability(
            available=False,
            reason=("deepagents package not installed. Install with: pip install deepagents or: uv add deepagents"),
            fallback_interface=None,
        )
