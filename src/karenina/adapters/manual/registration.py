"""Registration module for the manual interface adapter.

This module registers the manual interface with the AdapterRegistry.
The manual interface uses pre-recorded traces, not live LLM calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from karenina.adapters.registry import AdapterAvailability, AdapterRegistry, AdapterSpec

from . import ManualAgentAdapter, ManualLLMAdapter, ManualParserAdapter

if TYPE_CHECKING:
    from karenina.schemas.workflow.models import ModelConfig


def _check_availability() -> AdapterAvailability:
    """Manual interface is always available - uses pre-recorded traces."""
    return AdapterAvailability(
        available=True,
        reason="Manual interface uses pre-recorded traces",
    )


def _format_model_string(config: ModelConfig) -> str:
    """Format model string for manual interface."""
    return config.model_name or "manual"


# Register the manual adapter
_manual_spec = AdapterSpec(
    interface="manual",
    description="Manual interface for pre-recorded traces (no live LLM calls)",
    agent_factory=ManualAgentAdapter,
    llm_factory=ManualLLMAdapter,
    parser_factory=ManualParserAdapter,
    availability_checker=_check_availability,
    fallback_interface=None,  # No fallback - manual is intentional
    model_string_formatter=_format_model_string,
    routes_to=None,
    supports_mcp=False,
    supports_tools=False,
)

AdapterRegistry.register(_manual_spec)
