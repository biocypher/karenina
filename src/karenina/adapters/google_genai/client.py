"""google-genai client construction for the google_genai adapter.

Builds ``genai.Client`` for either backend (design D3):

- ``vertex``: ``Client(vertexai=True, project, location)`` with application
  default credentials.
- ``gemini_api``: ``Client(api_key, http_options=HttpOptions(base_url))``,
  for example the EBI POC gateway.

The SDK's own HTTP retries are switched off (``HttpRetryOptions(attempts=1)``)
because ``RetryExecutor`` is the only retry layer (design D6). One SDK retry
cannot be switched off: on its aiohttp transport, a request that fails to
connect or whose connection drops (``ClientConnectorError``, ``ClientOSError``,
``ServerDisconnectedError``) is sent once more after a 1 to 10 second sleep,
inside the same attempt and its time limit. The request
timeout is not set here: ``calls.py`` passes it per attempt, so a timeout
escalated by the retry policy can lengthen a call.

The SDK's async transport (aiohttp when installed) is bound to the event
loop that first used it, while karenina may call one adapter from several
worker loops. ``GenAIClientHolder`` therefore keeps one client per event
loop. A sync call made outside karenina's portal runs on a short-lived loop
(``asyncio.run``), so it goes through ``run_then_release``, which closes
that loop's client when the call ends.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import weakref
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from karenina.ports import AdapterUnavailableError

from .errors import translate_auth_error

if TYPE_CHECKING:
    from .settings import GoogleGenAISettings

logger = logging.getLogger(__name__)

_INSTALL_HINT = "google-genai not installed. Install with: pip install 'karenina[google-genai]'"

T = TypeVar("T")


def build_client(settings: GoogleGenAISettings) -> Any:
    """Build a ``genai.Client`` for the configured backend.

    Args:
        settings: The adapter settings.

    Returns:
        A new ``google.genai.Client``.

    Raises:
        AdapterUnavailableError: If the SDK is missing, no API key is
            available for the ``gemini_api`` backend, or the SDK rejects the
            client arguments.
        GeminiAuthError: If Google application default credentials are
            missing or cannot be loaded.
        GeminiConnectionError: If loading the credentials failed for a
            network reason.
    """
    try:
        from google import genai
        from google.auth.exceptions import GoogleAuthError
        from google.genai import types
    except ImportError as e:
        raise AdapterUnavailableError(_INSTALL_HINT, reason="google-genai not installed") from e

    http_kwargs: dict[str, Any] = {"retry_options": types.HttpRetryOptions(attempts=1)}
    if settings.base_url:
        http_kwargs["base_url"] = settings.base_url
    http_options = types.HttpOptions(**http_kwargs)

    try:
        if settings.backend == "vertex":
            return genai.Client(
                vertexai=True,
                project=settings.project,
                location=settings.location,
                http_options=http_options,
            )
        api_key = settings.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise AdapterUnavailableError(
                "The gemini_api backend needs an API key: set ModelConfig.endpoint_api_key, "
                "GEMINI_API_KEY or GOOGLE_API_KEY",
                reason="missing_api_key",
            )
        return genai.Client(api_key=api_key, http_options=http_options)
    except GoogleAuthError as e:
        raise translate_auth_error(e) from e
    except ValueError as e:
        raise AdapterUnavailableError(f"Could not create the google-genai client: {e}", reason=str(e)) from e


def running_loop() -> asyncio.AbstractEventLoop | None:
    """Return the running event loop, or None outside one."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class GenAIClientHolder:
    """Lazily builds one ``genai.Client`` per event loop and closes them.

    Several adapter instances may share one holder (for example an LLM
    adapter and the structured-output adapter derived from it), so they
    share connection pools.
    """

    def __init__(self, settings: GoogleGenAISettings) -> None:
        """Create an empty holder.

        Args:
            settings: Settings used to build each client.
        """
        self._settings = settings
        self._lock = threading.Lock()
        self._clients: dict[int, tuple[weakref.ref[asyncio.AbstractEventLoop] | None, Any]] = {}

    def get(self) -> Any:
        """Return the client for the running event loop, building it on first use.

        Returns:
            A ``google.genai.Client``.

        Raises:
            AdapterUnavailableError: See ``build_client``.
            GeminiAuthError: See ``build_client``.
        """
        loop = running_loop()
        key = id(loop) if loop is not None else 0
        with self._lock:
            entry = self._clients.get(key)
            if entry is not None:
                loop_ref, client = entry
                if loop_ref is None or loop_ref() is loop:
                    return client
            # Drop entries whose loop is gone or closed (an id may be reused).
            stale = [k for k, (ref, _) in self._clients.items() if ref is not None and _loop_finished(ref)]
            for stale_key in stale:
                _close_sync(self._clients.pop(stale_key)[1])
            client = build_client(self._settings)
            self._clients[key] = (weakref.ref(loop) if loop is not None else None, client)
            return client

    async def run_then_release(self, func: Callable[..., Awaitable[T]], *args: Any) -> T:
        """Await ``func(*args)``, then close the running loop's client.

        For sync wrappers that run a call on a loop that ends with it
        (``asyncio.run`` or a fresh thread). Without this, each such call
        would leave a client and its open sockets behind until ``aclose()``.

        Args:
            func: The async function to call.
            *args: Positional arguments for ``func``.

        Returns:
            What ``func`` returned.
        """
        try:
            return await func(*args)
        finally:
            await self.release_running_loop()

    async def release_running_loop(self) -> None:
        """Close and forget the client of the running loop, if one was built."""
        loop = running_loop()
        if loop is None:
            return
        with self._lock:
            entry = self._clients.pop(id(loop), None)
        if entry is None:
            return
        loop_ref, client = entry
        if loop_ref is not None and loop_ref() is loop:
            try:
                await client.aio.aclose()
            except Exception:
                logger.debug("Closing the google-genai async client failed", exc_info=True)
        _close_sync(client)

    async def aclose(self) -> None:
        """Close every client. Safe to call several times.

        The async transport is closed only for the client of the running
        loop, because an aiohttp session cannot be closed from another loop.
        Clients of other loops close their sync transport and are dropped.
        """
        loop = running_loop()
        with self._lock:
            entries = list(self._clients.values())
            self._clients = {}
        for loop_ref, client in entries:
            owner = loop_ref() if loop_ref is not None else None
            if owner is not None and owner is loop:
                try:
                    await client.aio.aclose()
                except Exception:
                    logger.debug("Closing the google-genai async client failed", exc_info=True)
            _close_sync(client)


def _loop_finished(loop_ref: weakref.ref[asyncio.AbstractEventLoop]) -> bool:
    """True when the loop was garbage collected or closed."""
    loop = loop_ref()
    return loop is None or loop.is_closed()


def _close_sync(client: Any) -> None:
    """Close a client's sync transport, ignoring failures."""
    try:
        client.close()
    except Exception:
        logger.debug("Closing the google-genai sync client failed", exc_info=True)
