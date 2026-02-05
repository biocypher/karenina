"""Centralized adapter registry for LLM backends.

This module provides a registry pattern for adapter discovery and creation.
Adapters self-register via AdapterSpec, enabling:

1. Centralized interface definitions (single source of truth)
2. Factory functions that always return port implementations (never None)
3. Clean fallback behavior without scattered if-else chains

Additionally, this module provides adapter instance tracking for cleanup:
- register_adapter(): Track an adapter instance for later cleanup
- unregister_adapter(): Remove an adapter from tracking
- cleanup_all_adapters(): Close all tracked adapters

Example:
    >>> from karenina.adapters.registry import AdapterRegistry
    >>>
    >>> # Get all registered interfaces
    >>> interfaces = AdapterRegistry.get_interfaces()
    >>> # {'langchain', 'openrouter', 'openai_endpoint', 'claude_agent_sdk', 'manual'}
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from karenina.ports import AgentPort, LLMPort, ParserPort
    from karenina.schemas.workflow.models import ModelConfig

logger = logging.getLogger(__name__)

# ============================================================================
# Adapter Instance Tracking for Resource Cleanup
# ============================================================================

# List of active adapter instances that need cleanup
_active_adapters: list[Any] = []

# Lock for thread-safe access to _active_adapters
_adapters_lock = threading.Lock()

# Lock for thread-safe lazy initialization of the AdapterRegistry
_registry_lock = threading.RLock()


def register_adapter(adapter: Any) -> None:
    """Register an adapter instance for cleanup tracking.

    Call this after creating an adapter to ensure it gets cleaned up
    when cleanup_all_adapters() is called.

    Args:
        adapter: An adapter instance with an optional aclose() method.
    """
    with _adapters_lock:
        _active_adapters.append(adapter)


def unregister_adapter(adapter: Any) -> None:
    """Remove an adapter from cleanup tracking.

    Call this if you manually close an adapter and want to remove it
    from the tracked list.

    Args:
        adapter: The adapter instance to remove.
    """
    with _adapters_lock:
        if adapter in _active_adapters:
            _active_adapters.remove(adapter)


async def cleanup_all_adapters() -> None:
    """Close all tracked adapter instances.

    This function iterates through all registered adapters and calls
    their aclose() method if available. Safe to call multiple times.

    Should be called during application shutdown or after verification
    completes to release HTTP connections and other resources.
    """
    with _adapters_lock:
        adapters = list(_active_adapters)
        _active_adapters.clear()

    closed_count = 0
    for adapter in adapters:
        if hasattr(adapter, "aclose"):
            try:
                await adapter.aclose()
                closed_count += 1
            except Exception as e:
                logger.debug(f"Error closing adapter {type(adapter).__name__}: {e}")

    if closed_count > 0:
        logger.debug(f"Closed {closed_count} adapter(s)")


@dataclass
class AdapterAvailability:
    """Result of checking adapter availability.

    Attributes:
        available: Whether the adapter can be used.
        reason: Human-readable explanation of availability status.
        fallback_interface: Suggested alternative interface if unavailable.

    Example:
        >>> availability = AdapterRegistry.check_availability("claude_agent_sdk")
        >>> if not availability.available:
        ...     print(f"Claude SDK unavailable: {availability.reason}")
        ...     print(f"Suggested fallback: {availability.fallback_interface}")
    """

    available: bool
    reason: str
    fallback_interface: str | None = None


@dataclass
class AdapterSpec:
    """Specification for an adapter that handles a specific interface.

    Each adapter registers itself with the registry by providing an AdapterSpec.
    The spec defines how to create adapters for that interface.

    Attributes:
        interface: The interface identifier (e.g., "langchain", "claude_agent_sdk").
        description: Human-readable description of the adapter.
        agent_factory: Factory function to create AgentPort for this interface.
        llm_factory: Factory function to create LLMPort for this interface.
        parser_factory: Factory function to create ParserPort for this interface.
        availability_checker: Function to check if this adapter is available.
        fallback_interface: Interface to fall back to if this one is unavailable.
        routes_to: If set, this interface routes to another (e.g., "openrouter" -> "langchain").

    Example:
        >>> spec = AdapterSpec(
        ...     interface="langchain",
        ...     description="LangChain adapter for standard LLM providers",
        ...     agent_factory=lambda cfg: LangChainAgentAdapter(cfg),
        ...     llm_factory=lambda cfg: LangChainLLMAdapter(cfg),
        ...     parser_factory=lambda cfg: LangChainParserAdapter(cfg),
        ... )
        >>> AdapterRegistry.register(spec)
    """

    interface: str
    description: str

    # Factory functions (can be None if adapter doesn't support that port type)
    agent_factory: Callable[[ModelConfig], AgentPort] | None = None
    llm_factory: Callable[[ModelConfig], LLMPort] | None = None
    parser_factory: Callable[[ModelConfig], ParserPort] | None = None

    # Availability checking
    availability_checker: Callable[[], AdapterAvailability] | None = None
    fallback_interface: str | None = None

    # Routing (for interfaces that delegate to another)
    routes_to: str | None = None

    # Additional metadata
    supports_mcp: bool = False
    supports_tools: bool = False


class AdapterRegistry:
    """Central registry for adapter specifications.

    This class maintains a registry of all available adapters and provides
    methods to query and create adapters based on interface type.

    The registry is populated lazily - adapters register themselves when
    their registration modules are imported.

    Class Methods:
        register: Register an adapter specification.
        get_spec: Get the spec for an interface.
        get_interfaces: Get all registered interface names.
        check_availability: Check if an interface is available.
        resolve_interface: Resolve routing (e.g., openrouter -> langchain).

    Example:
        >>> # Adapters self-register on import
        >>> from karenina.adapters.langchain import registration  # noqa: F401
        >>>
        >>> # Now the registry knows about langchain
        >>> spec = AdapterRegistry.get_spec("langchain")
        >>> spec.interface
        'langchain'
    """

    _specs: dict[str, AdapterSpec] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, spec: AdapterSpec, *, force: bool = False) -> AdapterSpec:
        """Register an adapter specification.

        Args:
            spec: The adapter specification to register.
            force: If True, allow overwriting an existing registration.
                   Use only for testing or intentional replacement.

        Returns:
            The registered spec (for chaining).

        Raises:
            ValueError: If an adapter for this interface is already registered
                        and force=False.
        """
        if spec.interface in cls._specs:
            if not force:
                raise ValueError(
                    f"Adapter for interface '{spec.interface}' is already registered. "
                    f"Use force=True to intentionally overwrite, or call _reset() first in tests."
                )
            logger.warning(f"Overwriting existing adapter spec for interface: {spec.interface}")

        cls._specs[spec.interface] = spec
        logger.debug(f"Registered adapter: {spec.interface} ({spec.description})")
        return spec

    @classmethod
    def get_spec(cls, interface: str) -> AdapterSpec | None:
        """Get the adapter specification for an interface.

        Args:
            interface: The interface identifier.

        Returns:
            The AdapterSpec if found, None otherwise.
        """
        cls._ensure_initialized()
        return cls._specs.get(interface)

    @classmethod
    def get_interfaces(cls) -> frozenset[str]:
        """Get all registered interface names.

        Returns:
            Frozen set of interface identifiers.
        """
        cls._ensure_initialized()
        return frozenset(cls._specs.keys())

    @classmethod
    def check_availability(cls, interface: str) -> AdapterAvailability:
        """Check if an adapter is available for the given interface.

        Args:
            interface: The interface to check.

        Returns:
            AdapterAvailability with status and details.
        """
        cls._ensure_initialized()
        spec = cls._specs.get(interface)

        if spec is None:
            return AdapterAvailability(
                available=False,
                reason=f"Unknown interface: {interface}. Registered: {list(cls._specs.keys())}",
                fallback_interface="langchain",
            )

        if spec.availability_checker is not None:
            return spec.availability_checker()

        # No checker means always available
        return AdapterAvailability(
            available=True,
            reason=f"{spec.description} is available",
        )

    @classmethod
    def resolve_interface(cls, interface: str) -> str:
        """Resolve an interface to its target (for routing).

        Some interfaces (like openrouter) route to another interface (langchain).
        This method resolves the chain.

        Args:
            interface: The interface to resolve.

        Returns:
            The resolved interface (may be the same if no routing).
        """
        cls._ensure_initialized()
        spec = cls._specs.get(interface)

        if spec is not None and spec.routes_to is not None:
            return cls.resolve_interface(spec.routes_to)

        return interface

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure all registration modules have been loaded.

        Uses double-checked locking to be thread-safe without paying the lock
        cost on every call after initialization is complete.
        """
        if cls._initialized:
            return

        with _registry_lock:
            # Double-check after acquiring the lock
            if cls._initialized:
                return

            # Import registration modules to trigger self-registration
            # This happens lazily on first registry access
            try:
                from karenina.adapters.langchain import registration as _lc  # noqa: F401
            except ImportError:
                logger.debug("LangChain registration module not available")

            try:
                from karenina.adapters.claude_agent_sdk import registration as _cas  # noqa: F401
            except ImportError:
                logger.debug("Claude Agent SDK registration module not available")

            try:
                from karenina.adapters.manual import registration as _manual  # noqa: F401
            except ImportError:
                logger.debug("Manual registration module not available")

            try:
                from karenina.adapters.claude_tool import registration as _ct  # noqa: F401
            except ImportError:
                logger.debug("Claude Tool registration module not available")

            cls._initialized = True

    @classmethod
    def _reset(cls) -> None:
        """Reset the registry (for testing only)."""
        with _registry_lock:
            cls._specs.clear()
            cls._initialized = False
