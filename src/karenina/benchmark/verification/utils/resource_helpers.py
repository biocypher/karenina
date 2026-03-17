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

    # Close all tracked adapters (AsyncAnthropic clients, etc.)
    cleanup_timeout = 10
    try:
        try:
            loop = asyncio.get_running_loop()
            # We're in a sync function but an event loop is running (e.g. nested
            # inside an async caller).  Schedule the coroutine on that loop and
            # block the current thread until it finishes or times out.
            future = asyncio.run_coroutine_threadsafe(cleanup_all_adapters(), loop)
            future.result(timeout=cleanup_timeout)
        except RuntimeError:
            # No event loop running — create one to run cleanup
            asyncio.run(cleanup_all_adapters())
        except TimeoutError:
            logger.warning("Adapter cleanup timed out after %d seconds", cleanup_timeout)
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
