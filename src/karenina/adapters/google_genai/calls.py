"""One Gemini model call, shared by the agent, LLM and parser ports.

``generate`` makes one ``generate_content`` (or ``generate_content_stream``)
call through ``RetryExecutor``, so only a failed call is repeated, never a
whole agent run. Each attempt:

- holds one ``GlobalLLMLimiter`` permit for the call only (design D13),
- runs under the ``request_timeout`` wall-clock guard,
- translates SDK errors into port errors (``errors.py``),
- passes ``request_timeout`` as the request's own HTTP timeout, so a
  timeout escalated by the retry policy really lengthens the call,
- for a stream, accumulates parts, usage and grounding across chunks,
- fails as ``GeminiIncompleteResponseError`` (retried) when the call ends
  without a finish reason, or with STOP but no text and no function call,
  so a cut stream or an empty turn is never an empty success.

Finish reasons are compared by name string (``classify_finish_reason``), so
reasons the installed SDK does not know map to a permanent stop instead of
crashing.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from google.auth.exceptions import GoogleAuthError
from google.genai import errors as genai_errors

from karenina.benchmark.verification.async_lifecycle import get_global_llm_limiter

from .errors import (
    GeminiClientError,
    GeminiFinishReasonError,
    GeminiIncompleteResponseError,
    translate_api_error,
    translate_auth_error,
)
from .grounding import GroundingInfo, merge_candidate_metadata

if TYPE_CHECKING:
    from karenina.utils.retry_policy import RetryExecutor

    from .client import GenAIClientHolder
    from .settings import GoogleGenAISettings

logger = logging.getLogger(__name__)

FinishKind = Literal["ok", "max_tokens", "tool_limit", "bad_tool_call", "blocked"]

_INCOMPLETE_REASONS = frozenset({"FINISH_REASON_UNSPECIFIED"})
_BAD_TOOL_CALL_REASONS = frozenset({"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"})


@dataclass
class CallOutcome:
    """The result of one model call.

    Attributes:
        parts: Every part the model returned (all stream chunks together).
        content: The model ``Content`` to append to the history unchanged.
            For a non-streamed call this is the API's own object. For a
            stream it holds every chunk's parts in order. None when the
            model returned no parts.
        finish_reason: Finish reason name, or None when none arrived.
        finish_message: The API's finish message, when sent.
        usage_metadata: The call's ``usage_metadata`` (last chunk for a stream).
        grounding: Server-side search and URL-reading activity.
        model_version: ``response.model_version``.
        response: The raw response (last chunk for a stream).
        block_reason: ``prompt_feedback.block_reason`` when the prompt was blocked.
    """

    parts: list[Any]
    content: Any | None
    finish_reason: str | None
    finish_message: str | None
    usage_metadata: Any | None
    grounding: GroundingInfo | None
    model_version: str | None
    response: Any | None
    block_reason: str | None = None

    @property
    def text(self) -> str:
        """The answer text: non-thought text parts joined without separator."""
        return "".join(p.text for p in self.parts if getattr(p, "text", None) and not getattr(p, "thought", None))

    @property
    def has_function_calls(self) -> bool:
        """True when the model asked for at least one function call."""
        return any(getattr(p, "function_call", None) is not None for p in self.parts)

    @property
    def search_query_count(self) -> int:
        """Google Search queries made during the call."""
        return self.grounding.query_count if self.grounding is not None else 0


def _enum_name(value: Any) -> str | None:
    """Name of an SDK enum value (or its string form), None for None."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value)
    return text.rsplit(".", 1)[-1] if "." in text else text


def classify_finish_reason(name: str | None) -> FinishKind:
    """Map a finish reason name to how the ports handle it (design Section 14).

    Args:
        name: Finish reason name, for example ``STOP``.

    Returns:
        ``ok`` (STOP), ``max_tokens``, ``tool_limit`` (TOO_MANY_TOOL_CALLS),
        ``bad_tool_call`` (MALFORMED_FUNCTION_CALL, UNEXPECTED_TOOL_CALL) or
        ``blocked`` for every other reason, including unknown ones.
    """
    if name == "STOP":
        return "ok"
    if name == "MAX_TOKENS":
        return "max_tokens"
    if name == "TOO_MANY_TOOL_CALLS":
        return "tool_limit"
    if name in _BAD_TOOL_CALL_REASONS:
        return "bad_tool_call"
    return "blocked"


def ensure_complete(outcome: CallOutcome, *, stream: bool) -> None:
    """Fail a call that returned no usable answer, so it is retried.

    Raises:
        GeminiFinishReasonError: When the prompt itself was blocked
            (permanent).
        GeminiIncompleteResponseError: When the call ended without a finish
            reason, or stopped with STOP but returned no text (thoughts
            excluded) and no function call. Gemini sometimes ends a turn
            this way after function responses.
    """
    if outcome.block_reason is not None and not outcome.parts:
        raise GeminiFinishReasonError(
            f"Gemini blocked the prompt: {outcome.block_reason}", finish_reason=outcome.block_reason
        )
    kind = "stream" if stream else "single call"
    if outcome.finish_reason is None or outcome.finish_reason in _INCOMPLETE_REASONS:
        raise GeminiIncompleteResponseError(
            f"Gemini call ended without a finish reason ({kind}, {len(outcome.parts)} parts received)"
        )
    if outcome.finish_reason == "STOP" and not outcome.text and not outcome.has_function_calls:
        raise GeminiIncompleteResponseError(
            f"Gemini stopped with no text and no function call ({kind}, {len(outcome.parts)} parts received)"
        )


def finish_error(outcome: CallOutcome) -> GeminiFinishReasonError:
    """Build the permanent error for a finish reason that leaves no answer."""
    reason = outcome.finish_reason or "UNKNOWN"
    message = f"Gemini stopped with finish reason {reason}"
    if outcome.finish_message:
        message += f": {outcome.finish_message}"
    return GeminiFinishReasonError(message, finish_reason=reason, finish_message=outcome.finish_message)


def raise_for_llm_finish(outcome: CallOutcome) -> None:
    """Apply the LLM port's finish-reason rules.

    STOP passes. MAX_TOKENS passes with a warning (the content is kept).
    Every other reason raises, because a single call has no loop to recover
    in.

    Raises:
        GeminiFinishReasonError: For tool-related, safety, policy and
            unknown stops.
    """
    kind = classify_finish_reason(outcome.finish_reason)
    if kind == "ok":
        return
    if kind == "max_tokens":
        logger.warning("Gemini stopped at MAX_TOKENS: the response may be truncated (%d chars)", len(outcome.text))
        return
    raise finish_error(outcome)


class StreamAccumulator:
    """Collects stream chunks into one ``CallOutcome``."""

    def __init__(self) -> None:
        self.parts: list[Any] = []
        self.role = "model"
        self.finish_reason: str | None = None
        self.finish_message: str | None = None
        self.usage_metadata: Any | None = None
        self.grounding: GroundingInfo | None = None
        self.model_version: str | None = None
        self.block_reason: str | None = None
        self.last_chunk: Any | None = None

    def add(self, chunk: Any) -> list[Any]:
        """Fold one chunk in and return the parts it added."""
        self.last_chunk = chunk
        if getattr(chunk, "usage_metadata", None) is not None:
            self.usage_metadata = chunk.usage_metadata
        if getattr(chunk, "model_version", None):
            self.model_version = chunk.model_version
        feedback = getattr(chunk, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None) is not None:
            self.block_reason = _enum_name(feedback.block_reason)
        candidates = getattr(chunk, "candidates", None) or []
        if not candidates:
            return []
        candidate = candidates[0]
        new_parts: list[Any] = []
        content = getattr(candidate, "content", None)
        if content is not None:
            if getattr(content, "role", None):
                self.role = content.role
            new_parts = list(getattr(content, "parts", None) or [])
            self.parts.extend(new_parts)
        if getattr(candidate, "finish_reason", None) is not None:
            self.finish_reason = _enum_name(candidate.finish_reason)
        if getattr(candidate, "finish_message", None):
            self.finish_message = candidate.finish_message
        self.grounding = merge_candidate_metadata(
            self.grounding,
            getattr(candidate, "grounding_metadata", None),
            getattr(candidate, "url_context_metadata", None),
        )
        return new_parts

    def outcome(self) -> CallOutcome:
        """Build the call outcome from the chunks seen so far."""
        content = None
        if self.parts:
            from google.genai import types

            content = types.Content(role=self.role, parts=list(self.parts))
        return CallOutcome(
            parts=list(self.parts),
            content=content,
            finish_reason=self.finish_reason,
            finish_message=self.finish_message,
            usage_metadata=self.usage_metadata,
            grounding=self.grounding,
            model_version=self.model_version,
            response=self.last_chunk,
            block_reason=self.block_reason,
        )


def outcome_from_response(response: Any) -> CallOutcome:
    """Build the outcome of a non-streamed call, keeping its Content unchanged."""
    accumulator = StreamAccumulator()
    accumulator.add(response)
    outcome = accumulator.outcome()
    candidates = getattr(response, "candidates", None) or []
    original = getattr(candidates[0], "content", None) if candidates else None
    if outcome.parts and original is not None:
        outcome.content = original
    return outcome


def build_generate_config(
    settings: GoogleGenAISettings,
    *,
    system_instruction: str | None,
    tools: Iterable[Any] | None = None,
    response_json_schema: dict[str, Any] | None = None,
    response_schema: Any | None = None,
) -> Any:
    """Build the ``GenerateContentConfig`` of one call.

    Automatic function calling is always disabled: the adapter runs tools
    itself. ``tool_config`` stays at its AUTO default.

    Args:
        settings: The adapter settings.
        system_instruction: The system instruction, or None.
        tools: ``types.Tool`` objects (function declarations and built-ins).
        response_json_schema: JSON Schema for structured output.
        response_schema: A Pydantic class for structured output, used when
            the JSON Schema form is rejected.

    Returns:
        A ``types.GenerateContentConfig``.
    """
    from google.genai import types

    kwargs: dict[str, Any] = dict(settings.generate_content_config)
    kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    if settings.max_output_tokens:
        kwargs["max_output_tokens"] = settings.max_output_tokens
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    tool_list = list(tools or [])
    if tool_list:
        kwargs["tools"] = tool_list
    thinking = settings.thinking_config_kwargs()
    if thinking is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(**thinking)
    if response_json_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_json_schema"] = response_json_schema
    elif response_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = response_schema
    return types.GenerateContentConfig(**kwargs)


def with_request_timeout(config: Any, timeout: float | None) -> Any:
    """Return ``config`` with the attempt's timeout as its own HTTP timeout.

    The SDK applies ``HttpOptions.timeout`` per request (an aiohttp total
    timeout), so setting it per attempt, rather than once on the client,
    lets an escalated retry timeout really lengthen the call. The client's
    other HTTP options (base URL, retries off) still apply.

    Args:
        config: The ``GenerateContentConfig``.
        timeout: The attempt's timeout in seconds, or None to leave the
            config unchanged.

    Returns:
        A copy of the config carrying the timeout, or the config itself.
    """
    if timeout is None:
        return config
    from google.genai import types

    return config.model_copy(update={"http_options": types.HttpOptions(timeout=int(timeout * 1000))})


_MAX_RAW_RESPONSE_CHARS = 300


@contextlib.contextmanager
def translated_sdk_errors() -> Iterator[None]:
    """Translate SDK and auth exceptions raised in the block into port errors.

    Raises:
        GeminiAPIError: For ``google.genai.errors.APIError``.
        GeminiIncompleteResponseError: For a response body or stream chunk
            that is not JSON (``UnknownApiResponseError``). Retried.
        GeminiConnectionError: For a network failure while refreshing
            Google credentials. Retried.
        GeminiAuthError: For other ``google.auth`` errors (login missing,
            expired or refused). Never retried.
        GeminiClientError: For other ``ValueError``s, which the SDK raises
            for requests it cannot send (for example a field the backend
            does not support). Never retried.
    """
    try:
        yield
    except genai_errors.APIError as e:
        raise translate_api_error(e) from e
    except genai_errors.UnknownApiResponseError as e:
        detail = str(e)[:_MAX_RAW_RESPONSE_CHARS]
        raise GeminiIncompleteResponseError(f"Gemini returned a response that is not JSON: {detail}") from e
    except GoogleAuthError as e:
        raise translate_auth_error(e) from e
    except ValueError as e:
        raise GeminiClientError(f"Gemini request rejected by the SDK: {e}") from e


async def generate(
    client: GenAIClientHolder,
    *,
    model: str,
    contents: list[Any],
    config: Any,
    stream: bool,
    request_timeout: float | None,
    retry_executor: RetryExecutor,
) -> CallOutcome:
    """Make one model call with transient retries.

    Args:
        client: Holder of the per-loop ``genai.Client``.
        model: The model name.
        contents: The conversation ``Content`` list.
        config: The ``GenerateContentConfig``.
        stream: Use ``generate_content_stream`` when True.
        request_timeout: Per-attempt wall-clock guard in seconds (None
            disables it). Escalated on TIMEOUT retries by the policy.
        retry_executor: The adapter's RetryExecutor.

    Returns:
        The call outcome. Its finish reason is set (never None).

    Raises:
        GeminiAPIError: An API error, after retries for transient ones.
        GeminiAuthError: Missing or expired Google credentials.
        GeminiIncompleteResponseError: No finish reason, an empty STOP, or
            a non-JSON response, after retries.
        GeminiFinishReasonError: The prompt itself was blocked.
        TimeoutError: The per-attempt guard fired on every attempt.
    """
    outcome: CallOutcome = await retry_executor.aexecute_with_timeout(
        _attempt, client, model, contents, config, stream, timeout=request_timeout
    )
    return outcome


async def _attempt(
    client: GenAIClientHolder,
    model: str,
    contents: list[Any],
    config: Any,
    stream: bool,
    *,
    timeout: float | None,
) -> CallOutcome:
    """One call attempt under a limiter permit and the wall-clock guard."""
    attempt_config = with_request_timeout(config, timeout)
    async with get_global_llm_limiter().borrow():
        with translated_sdk_errors():
            genai_client = client.get()
            call = (_call_streamed if stream else _call_once)(genai_client, model, contents, attempt_config)
            outcome = await (call if timeout is None else asyncio.wait_for(call, timeout=timeout))
    ensure_complete(outcome, stream=stream)
    return outcome


async def _call_once(genai_client: Any, model: str, contents: list[Any], config: Any) -> CallOutcome:
    """A non-streamed call."""
    response = await genai_client.aio.models.generate_content(model=model, contents=contents, config=config)
    return outcome_from_response(response)


async def _call_streamed(genai_client: Any, model: str, contents: list[Any], config: Any) -> CallOutcome:
    """A streamed call, accumulated into one outcome."""
    accumulator = StreamAccumulator()
    stream = await genai_client.aio.models.generate_content_stream(model=model, contents=contents, config=config)
    try:
        async for chunk in stream:
            accumulator.add(chunk)
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()
    return accumulator.outcome()
