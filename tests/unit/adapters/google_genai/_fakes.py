"""Shared fakes and builders for the google_genai cold tests.

A ``FakeModels`` object stands in for ``client.aio.models``: it replays
scripted responses (or raises scripted errors) for ``generate_content`` and
``generate_content_stream`` and records every request. ``install`` swaps a
fake client into an adapter's client holder, so no network call is made.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from google.genai import types

from karenina.schemas.config import ModelConfig
from karenina.utils.retry_policy import CategoryRetryConfig, RetryExecutor, RetryPolicy

MODEL_VERSION = "gemini-3.8-flash-001"


def make_config(**overrides: Any) -> ModelConfig:
    """ModelConfig for the google_genai interface (Vertex backend by default)."""
    values: dict[str, Any] = {
        "id": "gemini-test",
        "model_name": "gemini-3.8-flash",
        "interface": "google_genai",
        "max_tokens": 1024,
        "retry_policy": fast_policy(),
    }
    values.update(overrides)
    return ModelConfig(**values)


def fast_policy(*, server_error: int = 2, rate_limit: int = 2, timeout: int = 1, connection: int = 2) -> RetryPolicy:
    """Retry policy with zero backoff, so retry tests never sleep."""

    def cfg(attempts: int) -> CategoryRetryConfig:
        return CategoryRetryConfig(max_attempts=attempts, backoff_min=0.0, backoff_max=0.0)

    return RetryPolicy(
        connection=cfg(connection),
        timeout=cfg(timeout),
        rate_limit=cfg(rate_limit),
        server_error=cfg(server_error),
    )


def usage(
    prompt: int = 10,
    candidates: int = 5,
    *,
    thoughts: int | None = None,
    tool_prompt: int | None = None,
    cached: int | None = None,
    total: int | None = None,
) -> types.GenerateContentResponseUsageMetadata:
    """A usage_metadata object."""
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt,
        candidates_token_count=candidates,
        thoughts_token_count=thoughts,
        tool_use_prompt_token_count=tool_prompt,
        cached_content_token_count=cached,
        total_token_count=total,
    )


def text_part(text: str, *, signature: bytes | None = None) -> types.Part:
    return types.Part(text=text, thought_signature=signature)


def thought_part(text: str, *, signature: bytes | None = None) -> types.Part:
    return types.Part(text=text, thought=True, thought_signature=signature)


def call_part(name: str, args: dict[str, Any] | None = None, *, call_id: str | None = None) -> types.Part:
    return types.Part(function_call=types.FunctionCall(id=call_id, name=name, args=args or {}))


def search_grounding(queries: list[str], sources: list[tuple[str, str]] = ()) -> types.GroundingMetadata:
    """Grounding metadata with search queries and (title, uri) sources."""
    return types.GroundingMetadata(
        web_search_queries=queries,
        grounding_chunks=[types.GroundingChunk(web=types.GroundingChunkWeb(title=t, uri=u)) for t, u in sources],
    )


def url_context(*items: tuple[str, str]) -> types.UrlContextMetadata:
    """URL context metadata from (url, status) pairs."""
    return types.UrlContextMetadata(
        url_metadata=[types.UrlMetadata(retrieved_url=url, url_retrieval_status=status) for url, status in items]
    )


def response(
    parts: list[types.Part] | None,
    *,
    finish: str | None = "STOP",
    usage_metadata: Any = "default",
    grounding: types.GroundingMetadata | None = None,
    url_metadata: types.UrlContextMetadata | None = None,
    block_reason: str | None = None,
    finish_message: str | None = None,
) -> types.GenerateContentResponse:
    """A GenerateContentResponse (also used as a stream chunk)."""
    candidates = None
    if block_reason is None:
        candidates = [
            types.Candidate(
                content=types.Content(role="model", parts=parts) if parts is not None else None,
                finish_reason=finish,
                finish_message=finish_message,
                grounding_metadata=grounding,
                url_context_metadata=url_metadata,
            )
        ]
    return types.GenerateContentResponse(
        candidates=candidates,
        usage_metadata=usage() if usage_metadata == "default" else usage_metadata,
        model_version=MODEL_VERSION,
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=block_reason)
        if block_reason
        else None,
    )


class Delay:
    """Scripted pause inside a response list or a stream."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds


class FakeModels:
    """Stand-in for ``client.aio.models`` replaying scripted results.

    ``responses`` feeds ``generate_content``. ``streams`` feeds
    ``generate_content_stream`` (each entry is a list of chunks). Entries may
    be exceptions (raised) or lists starting with a ``Delay``.
    """

    def __init__(self, responses: list[Any] | None = None, streams: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.streams = list(streams or [])
        self.calls: list[dict[str, Any]] = []

    def _record(self, model: str, contents: list[Any], config: Any, kind: str) -> None:
        self.calls.append({"model": model, "contents": list(contents), "config": config, "kind": kind})

    async def generate_content(self, *, model: str, contents: list[Any], config: Any) -> Any:
        self._record(model, contents, config, "generate")
        item = self.responses.pop(0)
        while isinstance(item, Delay):
            await asyncio.sleep(item.seconds)
            item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def generate_content_stream(self, *, model: str, contents: list[Any], config: Any) -> AsyncIterator[Any]:
        self._record(model, contents, config, "stream")
        item = self.streams.pop(0)
        if isinstance(item, BaseException):
            raise item

        async def chunks() -> AsyncIterator[Any]:
            for chunk in item:
                if isinstance(chunk, Delay):
                    await asyncio.sleep(chunk.seconds)
                    continue
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk

        return chunks()


class FakeClient:
    """Stand-in for ``genai.Client`` exposing ``aio.models`` and close methods."""

    def __init__(self, models: FakeModels) -> None:
        self.models = models
        self.aio = SimpleNamespace(models=models, aclose=AsyncMock())
        self.close = MagicMock()


def install(holder_owner: Any, models: FakeModels) -> FakeClient:
    """Swap a fake client into an adapter's (or a holder's) ``get``."""
    holder = getattr(holder_owner, "_client", holder_owner)
    client = FakeClient(models)
    holder.get = lambda: client
    return client


async def run_generate(
    models: FakeModels,
    *,
    stream: bool = False,
    request_timeout: float | None = None,
    retry_policy: RetryPolicy | None = None,
) -> Any:
    """Run ``calls.generate`` once against fake models with a zero-backoff policy."""
    from karenina.adapters.google_genai.calls import build_generate_config, generate
    from karenina.adapters.google_genai.client import GenAIClientHolder
    from karenina.adapters.google_genai.errors import build_error_registry
    from karenina.adapters.google_genai.settings import GoogleGenAISettings

    settings = GoogleGenAISettings.from_model_config(make_config())
    holder = GenAIClientHolder(settings)
    install(holder, models)
    return await generate(
        holder,
        model="gemini-3.8-flash",
        contents=[types.Content(role="user", parts=[types.Part(text="Q")])],
        config=build_generate_config(settings, system_instruction=None),
        stream=stream,
        request_timeout=request_timeout,
        retry_executor=RetryExecutor(retry_policy or fast_policy(), build_error_registry()),
    )
