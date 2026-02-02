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
    try:
        # Try to run in existing event loop if available
        try:
            loop = asyncio.get_running_loop()
            # Schedule cleanup but don't wait (we're in sync context)
            loop.create_task(cleanup_all_adapters())
        except RuntimeError:
            # No event loop running - create one to run cleanup
            asyncio.run(cleanup_all_adapters())
    except Exception as e:
        logger.debug(f"Adapter cleanup encountered error: {e}")

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
