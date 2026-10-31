"""google-genai client construction for the google_genai adapter.

Builds ``genai.Client`` for either backend (design D3):

- ``vertex``: ``Client(vertexai=True, project, location)`` with application
  default credentials.
- ``gemini_api``: ``Client(api_key, http_options=HttpOptions(base_url))``,
  for example the EBI POC gateway.

The SDK's own HTTP retries are switched off (``HttpRetryOptions(attempts=1)``)
because ``RetryExecutor`` is the only retry layer (design D6). The request
timeout is not set here: ``calls.py`` passes it per attempt, so a timeout
escalated by the retry policy can lengthen a call.

The SDK's async transport (aiohttp when installed) is bound to the event
loop that first used it, while karenina may call one adapter from several
worker loops. ``GenAIClientHolder`` therefore keeps one client per event
loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import weakref
from typing import TYPE_CHECKING, Any

from karenina.ports import AdapterUnavailableError

from .errors import translate_auth_error

if TYPE_CHECKING:
    from .settings import GoogleGenAISettings

logger = logging.getLogger(__name__)

_INSTALL_HINT = "google-genai not installed. Install with: pip install 'karenina[google-genai]'"


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
            # Drop entries whose loop is gone (their id may be reused).
            self._clients = {k: v for k, v in self._clients.items() if v[0] is None or v[0]() is not None}
            client = build_client(self._settings)
            self._clients[key] = (weakref.ref(loop) if loop is not None else None, client)
            return client

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
            try:
                client.close()
            except Exception:
                logger.debug("Closing the google-genai sync client failed", exc_info=True)
