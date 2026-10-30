"""Error types and retry classification for the google_genai adapter.

The SDK's own exceptions are translated into port errors inside each call
attempt, so ``RetryExecutor`` classifies them by type rather than by
message text (a 400 whose message mentions "timeout" must stay permanent).
``build_error_registry`` returns the adapter's own ``ErrorRegistry``
instance with these types registered (design Section 14).
"""

from __future__ import annotations

import logging
from typing import Any

from karenina.ports import AdapterUnavailableError, PortError
from karenina.utils.errors import ErrorCategory, ErrorRegistry

logger = logging.getLogger(__name__)

AUTH_HINT = "Google login missing or expired: run gcloud auth application-default login"


class GeminiAPIError(PortError):
    """An HTTP error returned by the Gemini or Vertex API.

    Attributes:
        code: The HTTP status code, when known.
        status: The API status string (for example ``INVALID_ARGUMENT``).
    """

    def __init__(self, message: str, *, code: int | None = None, status: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class GeminiClientError(GeminiAPIError):
    """A 4xx API error other than rate limiting. Permanent, never retried."""


class GeminiRateLimitError(GeminiAPIError):
    """A 429 / ``RESOURCE_EXHAUSTED`` API error. Retried with the rate-limit budget."""


class GeminiServerError(GeminiAPIError):
    """A 5xx API error, including the EBI gateway's HTML 500. Retried."""


class GeminiAuthError(PortError):
    """Google credentials are missing or were refused. Permanent."""


class GeminiConnectionError(PortError):
    """A transient network failure outside the API call itself.

    Raised when refreshing Google credentials fails for a network reason
    (``google.auth`` TransportError, or a RefreshError marked retryable).
    Retried with the connection budget.
    """


class GeminiIncompleteResponseError(PortError):
    """A call returned no usable answer.

    Raised when a call ends without a finish reason (for example a stream
    cut short), when it stops with STOP but holds no text and no function
    call, or when a response body is not JSON. Treated as a server error and
    retried, never returned as an empty answer.
    """


class GeminiFinishReasonError(PortError):
    """The model stopped for a reason that leaves no usable answer.

    Raised for safety and policy stops, and in the LLM port for tool-related
    stops. Permanent, never retried.

    Attributes:
        finish_reason: The finish reason name, for example ``SAFETY``.
        finish_message: The API's explanation, when it sent one.
    """

    def __init__(self, message: str, *, finish_reason: str, finish_message: str | None = None) -> None:
        super().__init__(message)
        self.finish_reason = finish_reason
        self.finish_message = finish_message


def translate_api_error(exc: Any) -> GeminiAPIError:
    """Translate a ``google.genai.errors.APIError`` into a port error.

    Args:
        exc: The SDK API error (``ClientError`` or ``ServerError``).

    Returns:
        The matching GeminiAPIError subclass. The caller raises it with
        ``from exc``.
    """
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    detail = getattr(exc, "message", None) or str(exc)
    message = f"Gemini API error {code} {status}: {detail}"
    if isinstance(code, int) and code >= 500:
        return GeminiServerError(message, code=code, status=status)
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        return GeminiRateLimitError(message, code=code, status=status)
    if code is None and type(exc).__name__ == "ServerError":
        return GeminiServerError(message, code=code, status=status)
    return GeminiClientError(message, code=code, status=status)


def translate_auth_error(exc: Exception) -> PortError:
    """Translate a ``google.auth`` exception into a port error.

    Network failures while refreshing credentials are transient. Every
    other auth failure means the login is missing, expired or refused.

    Args:
        exc: A ``google.auth.exceptions.GoogleAuthError``.

    Returns:
        GeminiConnectionError for a TransportError or a retryable error,
        GeminiAuthError (with the login hint) otherwise. The caller raises
        it with ``from exc``.
    """
    from google.auth.exceptions import TransportError

    detail = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, TransportError) or getattr(exc, "retryable", False):
        return GeminiConnectionError(f"Network failure while refreshing Google credentials ({detail})")
    return GeminiAuthError(f"{AUTH_HINT} ({detail})")


def build_error_registry() -> ErrorRegistry:
    """Build the adapter's ErrorRegistry with the google_genai patterns.

    Returns:
        A new ErrorRegistry. User-registered rules take priority over the
        built-in type and message rules.
    """
    registry = ErrorRegistry()
    registry.register(AdapterUnavailableError, ErrorCategory.PERMANENT)
    registry.register(GeminiAuthError, ErrorCategory.PERMANENT)
    registry.register(GeminiFinishReasonError, ErrorCategory.PERMANENT)
    registry.register(GeminiClientError, ErrorCategory.PERMANENT)
    registry.register(GeminiRateLimitError, ErrorCategory.RATE_LIMIT)
    registry.register(GeminiServerError, ErrorCategory.SERVER_ERROR)
    registry.register(GeminiIncompleteResponseError, ErrorCategory.SERVER_ERROR)
    registry.register(GeminiConnectionError, ErrorCategory.CONNECTION)
    # Raw SDK errors, in case one escapes translation.
    registry.register("ServerError", ErrorCategory.SERVER_ERROR)
    registry.register_pattern("resource_exhausted", ErrorCategory.RATE_LIMIT)
    _register_transport_errors(registry)
    return registry


def _register_transport_errors(registry: ErrorRegistry) -> None:
    """Register the SDK transports' network errors.

    The SDK uses aiohttp for async calls when it is installed, httpx
    otherwise. Their connection errors carry names and messages the
    built-in rules do not recognise (for example "Server disconnected"), so
    a dropped stream would otherwise be classified as permanent.
    """
    try:
        import aiohttp
    except ImportError:
        pass
    else:
        # ServerTimeoutError subclasses ClientConnectionError, so it goes first.
        registry.register(aiohttp.ServerTimeoutError, ErrorCategory.TIMEOUT)
        registry.register(aiohttp.ClientConnectionError, ErrorCategory.CONNECTION)
        registry.register(aiohttp.ClientPayloadError, ErrorCategory.CONNECTION)
    try:
        import httpx
    except ImportError:
        pass
    else:
        registry.register(httpx.TimeoutException, ErrorCategory.TIMEOUT)
        registry.register(httpx.TransportError, ErrorCategory.CONNECTION)
