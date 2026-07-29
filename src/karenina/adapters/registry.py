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
import weakref
from collections.abc import Callable
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anyio.from_thread import BlockingPortal

    from karenina.ports import AgentPort, LLMPort, ParserPort
    from karenina.schemas.config import ModelConfig

logger = logging.getLogger(__name__)

ADAPTER_CLOSE_TIMEOUT = 10.0

# ============================================================================
# Adapter Instance Tracking for Resource Cleanup
# ============================================================================

# List of active adapter instances that need cleanup
_active_adapters: list[Any] = []

# Lock for thread-safe access to _active_adapters
_adapters_lock = threading.Lock()

# Lock for thread-safe lazy initialization of the AdapterRegistry
_registry_lock = threading.RLock()

# Per-portal adapter tracking for loop-affine teardown. When an adapter is
# created inside a worker thread that has an active BlockingPortal, we record
# a (weakref, portal) pair here so the executor can call adapter.aclose() on
# the portal's own event loop before the portal is torn down. This prevents
# "Event loop is closed" errors from httpx.AsyncClient.aclose() running on a
# different loop than the one that opened its transports.
_adapter_portal_refs: list[tuple[weakref.ref[Any], BlockingPortal]] = []
_adapter_portal_lock = threading.Lock()


def register_adapter(adapter: Any) -> None:
    """Register an adapter instance for cleanup tracking.

    Call this after creating an adapter to ensure it gets cleaned up
    when cleanup_all_adapters() is called.

    Args:
        adapter: An adapter instance with an optional aclose() method.
    """
    with _adapters_lock:
        _active_adapters.append(adapter)

    # Record loop affinity when the creating thread has an active portal.
    # Late import avoids a circular import with benchmark.verification.executor.
    try:
        from karenina.benchmark.verification.executor import get_async_portal
    except ImportError:
        return

    portal = get_async_portal()
    if portal is None:
        return
    with _adapter_portal_lock:
        _adapter_portal_refs.append((weakref.ref(adapter), portal))


def snapshot_adapters_for_portal(portal: BlockingPortal) -> list[Any]:
    """Return live adapters registered while this portal was active.

    The copy is taken under the portal-adapter lock, but the lock is released
    before the caller invokes aclose. This prevents a stray register_adapter
    call on another thread from deadlocking against a stuck portal aclose.

    Args:
        portal: The BlockingPortal to look up.

    Returns:
        List of adapter instances (may be empty) that are still alive.
    """
    result: list[Any] = []
    with _adapter_portal_lock:
        for adapter_ref, portal_ref in _adapter_portal_refs:
            if portal_ref is portal:
                adapter = adapter_ref()
                if adapter is not None:
                    result.append(adapter)
    return result


def clear_portal_adapter_refs(portal: BlockingPortal) -> None:
    """Drop all tracked (adapter, portal) pairs for the given portal.

    Call this after a portal has been torn down, or on a sequential-mode
    finally, to prevent the module-global tracking list from leaking entries
    across runs.

    Args:
        portal: The BlockingPortal whose entries should be removed.
    """
    with _adapter_portal_lock:
        _adapter_portal_refs[:] = [pair for pair in _adapter_portal_refs if pair[1] is not portal]


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
    with _adapter_portal_lock:
        _adapter_portal_refs[:] = [
            pair for pair in _adapter_portal_refs if (tracked := pair[0]()) is not None and tracked is not adapter
        ]


async def aclose_adapter(adapter: Any, *, unregister: bool = True) -> None:
    """Close one adapter immediately and optionally remove it from tracking.

    This is intended for adapters with a short, local lifetime. Batch-level
    cleanup remains as a fallback for adapters that are intentionally shared or
    whose close fails here.
    """
    aclose = getattr(adapter, "aclose", None)
    if not callable(aclose):
        if unregister:
            unregister_adapter(adapter)
        return

    await aclose()
    if unregister:
        unregister_adapter(adapter)


def close_adapter(adapter: Any, *, timeout: float = ADAPTER_CLOSE_TIMEOUT, unregister: bool = True) -> None:
    """Synchronously close one adapter using the active worker portal when available."""
    aclose = getattr(adapter, "aclose", None)
    if not callable(aclose):
        if unregister:
            unregister_adapter(adapter)
        return

    try:
        from karenina.benchmark.verification.executor import get_async_portal
    except ImportError:
        portal = None
    else:
        portal = get_async_portal()

    try:
        if portal is not None:
            future = portal.start_task_soon(aclose)
            future.result(timeout=timeout)
        else:
            import asyncio

            asyncio.run(aclose())
    except FutureTimeoutError:
        logger.warning(
            "Immediate adapter close timed out on %s (>%ss); leaving it registered for batch cleanup",
            type(adapter).__name__,
            timeout,
        )
        return
    except Exception as exc:
        logger.debug(
            "Immediate adapter close failed on %s; leaving it registered for batch cleanup: %s",
            type(adapter).__name__,
            exc,
        )
        return

    if unregister:
        unregister_adapter(adapter)


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
    failed_count = 0
    for adapter in adapters:
        if hasattr(adapter, "aclose"):
            try:
                await adapter.aclose()
                closed_count += 1
            except Exception:
                failed_count += 1
                logger.warning(
                    "Error closing adapter %s",
                    type(adapter).__name__,
                    exc_info=True,
                )

    if failed_count > 0:
        logger.warning(
            "Adapter cleanup finished with %d failure(s) out of %d adapter(s)",
            failed_count,
            closed_count + failed_count,
        )
    if closed_count > 0:
        logger.debug("Closed %d adapter(s)", closed_count)


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
        runtime_profile: Optional runtime behavior profile for shared agent
            prompt/capability helpers. External adapters can provide this to
            integrate with sandbox path mapping without changing core code.

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

    # Optional runtime behavior extension hook. Kept as Any to avoid coupling
    # the adapter registry to agent-only helper types at import time.
    runtime_profile: Any | None = None

    # Additional metadata
    supports_mcp: bool = False
    supports_tools: bool = False

    # Agent capability tier:
    # - "tool_loop": Basic tool-calling loop (e.g. LangChain ReAct). The adapter
    #   orchestrates each tool call turn explicitly.
    # - "deep_agent": Full agent runtime with built-in tools (e.g. Claude Code,
    #   LangChain Deep Agents). The runtime handles tool loops internally;
    #   GenerateAnswer should prefer the AgentPort path to capture the full trace.
    agent_tier: str = "tool_loop"

    # If False, model_provider is not required for this interface.
    # Used by validate_model_config() to replace the hardcoded
    # INTERFACES_NO_PROVIDER_REQUIRED list.
    requires_provider: bool = True


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
    _initializing: bool = False

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
        if spec.runtime_profile is not None:
            from karenina.adapters.agent_runtime import register_agent_runtime_profile

            register_agent_runtime_profile(spec.interface, spec.runtime_profile, force=force)
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
        if cls._initialized or cls._initializing:
            return

        with _registry_lock:
            if cls._initialized or cls._initializing:
                return

            cls._initializing = True
            try:
                cls._load_builtins()
                cls._discover_entry_points()
                cls._initialized = True
            finally:
                cls._initializing = False

    @classmethod
    def _load_builtins(cls) -> None:
        """Import built-in registration modules.

        Each module calls AdapterRegistry.register() at import time.
        ImportError is caught per-adapter so missing optional dependencies
        do not block other adapters.
        """
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

        try:
            from karenina.adapters.langchain_deep_agents import registration as _da  # noqa: F401
        except ImportError:
            logger.debug("LangChain Deep Agents registration module not available")

        try:
            from karenina.adapters.taskeval import registration as _te  # noqa: F401
        except ImportError:
            logger.debug("TaskEval registration module not available")

    @classmethod
    def _discover_entry_points(cls) -> None:
        """Discover and load external adapters via ``karenina.adapters`` entry points.

        Entry points whose name conflicts with an already-registered
        (built-in) interface are skipped with a warning.
        """
        for ep in entry_points(group="karenina.adapters"):
            if ep.name in cls._specs:
                logger.warning(
                    "External adapter '%s' conflicts with built-in interface, skipping",
                    ep.name,
                )
                continue
            try:
                ep.load()
                logger.debug("Loaded external adapter: %s", ep.name)
            except ValueError:
                logger.warning(
                    "External adapter '%s' conflicts with another external adapter, skipping",
                    ep.name,
                )
            except Exception:
                logger.warning(
                    "Failed to load external adapter '%s'",
                    ep.name,
                    exc_info=True,
                )

    @classmethod
    def _reset(cls) -> None:
        """Reset the registry (for testing only)."""
        with _registry_lock:
            cls._specs.clear()
            cls._initialized = False
            cls._initializing = False
