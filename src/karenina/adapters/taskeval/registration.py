"""Registration module for the taskeval interface.

Registers a no-op 'taskeval' interface with the AdapterRegistry. This interface
exists solely as a valid sentinel for ModelConfig when TaskEval evaluates
pre-collected outputs. No LLM invocation occurs; the model is never called.
"""

from karenina.adapters.registry import AdapterAvailability, AdapterRegistry, AdapterSpec


def _check_availability() -> AdapterAvailability:
    """TaskEval interface is always available."""
    return AdapterAvailability(
        available=True,
        reason="TaskEval interface for pre-collected output evaluation",
    )


_taskeval_spec = AdapterSpec(
    interface="taskeval",
    description="No-op interface for TaskEval pre-collected output evaluation",
    llm_factory=None,
    parser_factory=None,
    agent_factory=None,
    availability_checker=_check_availability,
    fallback_interface=None,
    routes_to=None,
    supports_mcp=False,
    supports_tools=False,
    requires_provider=False,
)

AdapterRegistry.register(_taskeval_spec)
