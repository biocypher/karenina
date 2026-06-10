"""Resource cleanup helper functions for verification.

This module provides cleanup functionality for lingering resources
(adapters, database engines) that may prevent clean process exit.
"""

import logging

logger = logging.getLogger(__name__)


def cleanup_resources() -> None:
    """Clean up lingering resources that may prevent process exit.

    This function closes:
    1. All tracked adapter instances (AsyncAnthropic clients, etc.)
    2. All cached SQLAlchemy database engines

    These resources create non-daemon threads (httpx connection pools)
    that prevent clean process exit if not properly closed.
    """
    import asyncio
    import gc

    from karenina.adapters.registry import cleanup_all_adapters
    from karenina.benchmark.verification.async_lifecycle import get_async_portal

    # Close all tracked adapters (AsyncAnthropic clients, etc.)
    cleanup_timeout = 10
    try:
        running_loop: asyncio.AbstractEventLoop | None
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None:
            # We're in a sync function but an event loop is running in this
            # thread (e.g. nested inside an async caller). The shared portal
            # must NOT be used here: portal dispatch would block this thread
            # and could nest on the portal's own loop. Schedule the coroutine
            # on the running loop and block until it finishes or times out.
            try:
                future = asyncio.run_coroutine_threadsafe(cleanup_all_adapters(), running_loop)
                future.result(timeout=cleanup_timeout)
            except TimeoutError:
                logger.warning("Adapter cleanup timed out after %d seconds", cleanup_timeout)
        else:
            portal = get_async_portal()
            if portal is not None:
                # Loop affinity: adapters created on the shared portal's loop
                # own httpx transports pinned to it, so close them on that
                # loop instead of a fresh asyncio.run() loop. The dispatch is
                # bounded (start_task_soon + result(timeout) + cancel) so a
                # wedged aclose cannot stall shutdown.
                portal_future = portal.start_task_soon(cleanup_all_adapters)
                try:
                    portal_future.result(timeout=cleanup_timeout)
                except TimeoutError:
                    portal_future.cancel()
                    logger.warning(
                        "Adapter cleanup timed out after %d seconds, cancelled portal dispatch",
                        cleanup_timeout,
                    )
            else:
                # No event loop running and no portal: create one to run cleanup.
                asyncio.run(cleanup_all_adapters())
    except Exception:
        logger.exception("Adapter cleanup failed")

    # Dispose all cached SQLAlchemy engines
    try:
        from karenina.storage.engine import close_all_engines

        close_all_engines()
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Engine cleanup encountered error: {e}")

    # Force garbage collection to clean up any remaining objects
    gc.collect()

    logger.debug("Resource cleanup completed")
