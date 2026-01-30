"""Adapters module - implementations of port interfaces for LLM backends.

This module contains adapter implementations that connect the port abstractions
(karenina.ports) to concrete LLM backends like LangChain and Claude Agent SDK.

The adapters follow the Ports and Adapters (hexagonal) architecture pattern,
providing pluggable implementations that can be swapped without changing
application code.

Available adapters:
    - LangChain: Wraps existing LangChain/LangGraph infrastructure
    - Claude SDK: (Planned) Native Anthropic Agent SDK support

Factory functions:
    - get_llm: Create an LLMPort implementation for a given model config
    - get_agent: Create an AgentPort implementation for a given model config
    - get_parser: Create a ParserPort implementation for a given model config
    - check_adapter_available: Check if an adapter is available

Parallel invokers:
    - LLMParallelInvoker: Batch LLMPort invocations (plain text and structured)

Example:
    >>> from karenina.adapters import get_agent, get_llm
    >>> from karenina.schemas.workflow.models import ModelConfig
    >>>
    >>> config = ModelConfig(model="claude-sonnet-4-20250514", provider="anthropic")
    >>> agent = get_agent(config)
    >>> result = await agent.run(messages=[Message.user("Hello!")])
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Import types for type checking only
    from karenina.adapters.factory import (
        build_llm_kwargs,
        check_adapter_available,
        format_model_string,
        get_agent,
        get_llm,
        get_parser,
        validate_model_config,
    )
    from karenina.adapters.llm_parallel import LLMParallelInvoker
    from karenina.adapters.registry import AdapterAvailability

# Note: Factory functions will be exported once lc-008 is implemented.
# For now, this module establishes the package structure.

__all__ = [
    # Factory functions
    "get_llm",
    "get_agent",
    "get_parser",
    "check_adapter_available",
    "format_model_string",
    "build_llm_kwargs",
    "validate_model_config",
    # Availability checking
    "AdapterAvailability",
    # Parallel invocation
    "LLMParallelInvoker",
]


def __getattr__(name: str) -> Any:
    """Lazy import factory functions to avoid circular imports.

    This allows importing from karenina.adapters before all submodules
    are implemented.
    """
    if name in (
        "get_llm",
        "get_agent",
        "get_parser",
        "check_adapter_available",
        "format_model_string",
        "build_llm_kwargs",
        "validate_model_config",
    ):
        try:
            from karenina.adapters.factory import (
                build_llm_kwargs,
                check_adapter_available,
                format_model_string,
                get_agent,
                get_llm,
                get_parser,
                validate_model_config,
            )

            return {
                "get_llm": get_llm,
                "get_agent": get_agent,
                "get_parser": get_parser,
                "check_adapter_available": check_adapter_available,
                "format_model_string": format_model_string,
                "build_llm_kwargs": build_llm_kwargs,
                "validate_model_config": validate_model_config,
            }[name]
        except ImportError as e:
            raise ImportError(
                f"Cannot import '{name}' - factory module not yet implemented. See PR2 task lc-008. Original error: {e}"
            ) from e

    if name == "AdapterAvailability":
        from karenina.adapters.registry import AdapterAvailability

        return AdapterAvailability

    if name == "LLMParallelInvoker":
        from karenina.adapters.llm_parallel import LLMParallelInvoker

        return LLMParallelInvoker

    raise AttributeError(f"module 'karenina.adapters' has no attribute '{name}'")
