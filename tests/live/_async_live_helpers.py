"""Shared helpers for the live async behavior acceptance suite.

Model builders, retry policies, the in-flight counting wrapper, and the
docker preflight gate used by ``test_async_behavior_live.py``. Everything
here targets the vLLM server on codon-gpu-001 (OpenAI side at
``/v1/chat/completions``, Anthropic side at ``/v1/messages``) or, for the
retry telemetry tests, a dead local port that fails fast without a GPU.

Environment overrides:
    KARENINA_LIVE_VLLM_URL: base URL of the vLLM server
        (default ``http://codon-gpu-001:8000``).
    KARENINA_LIVE_VLLM_MODEL: served model name
        (default ``qwen3.5-122b-a10b``).
    KARENINA_LIVE_CLAUDE_CLI_IMAGE: pinned Claude CLI container image
        (default ``karenina-live-claude-cli:2.1.146``).
"""

from __future__ import annotations

import asyncio
import functools
import os
import shutil
import subprocess
import threading
from collections.abc import Callable, Coroutine, Iterable
from typing import Any

import pytest

from karenina.adapters import check_adapter_available
from karenina.schemas.config import ModelConfig
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy

LIVE_VLLM_URL = os.getenv("KARENINA_LIVE_VLLM_URL", "http://codon-gpu-001:8000")
LIVE_VLLM_MODEL = os.getenv("KARENINA_LIVE_VLLM_MODEL", "qwen3.5-122b-a10b")
CLAUDE_CLI_IMAGE = os.getenv("KARENINA_LIVE_CLAUDE_CLI_IMAGE", "karenina-live-claude-cli:2.1.146")

# Dead endpoint for the no-GPU retry telemetry tests (B9). Port 1 is
# reserved and unbound, so connections are refused immediately.
DEAD_PORT_URL = "http://127.0.0.1:1"

# vLLM honors chat_template_kwargs on the OpenAI side. Disabling thinking
# keeps short factual answers fast and cheap. SDK retries are zeroed so the
# RetryExecutor inside the adapters is the only retry layer.
OPENAI_EXTRA_KWARGS: dict[str, Any] = {
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    "max_retries": 0,
}

# Substrings that indicate broken async teardown. Checked against captured
# warnings and log records by B7/B8/B10.
TEARDOWN_ERROR_MARKERS: tuple[str, ...] = (
    "Event loop is closed",
    "was never awaited",
    "Task was destroyed but it is pending",
)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def openai_model(
    role_id: str = "qwen-openai",
    *,
    stream_usage: bool = False,
    **overrides: Any,
) -> ModelConfig:
    """OpenAI-side vLLM model (interface ``openai_endpoint``, langchain adapter).

    Args:
        role_id: ModelConfig.id, useful to distinguish roles in multi-model runs.
        stream_usage: When True, ask ChatOpenAI to request usage in streaming
            responses (``stream_options.include_usage``), needed by B3.
        **overrides: Extra ModelConfig fields applied last.
    """
    extra_kwargs = {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "max_retries": 0,
    }
    if stream_usage:
        extra_kwargs["stream_usage"] = True
    fields: dict[str, Any] = {
        "id": role_id,
        "model_name": LIVE_VLLM_MODEL,
        "interface": "openai_endpoint",
        "endpoint_base_url": LIVE_VLLM_URL,
        "endpoint_api_key": "EMPTY",
        "temperature": 0.0,
        "max_tokens": 512,
        "request_timeout": 180.0,
        "extra_kwargs": extra_kwargs,
    }
    fields.update(overrides)
    return ModelConfig(**fields)


def claude_tool_model(role_id: str = "qwen-claude-tool", **overrides: Any) -> ModelConfig:
    """Anthropic-side vLLM model (interface ``claude_tool``, raw Anthropic SDK).

    vLLM's ``/v1/messages`` returns thinking blocks by default and thinking
    can burn most of ``max_tokens``, so the budget is generous and tests must
    extract text blocks tolerantly (the adapter already joins text blocks only).
    """
    fields: dict[str, Any] = {
        "id": role_id,
        "model_name": LIVE_VLLM_MODEL,
        "interface": "claude_tool",
        "anthropic_base_url": LIVE_VLLM_URL,
        "anthropic_api_key": "EMPTY",
        "temperature": 0.0,
        "max_tokens": 1024,
        "request_timeout": 180.0,
    }
    fields.update(overrides)
    return ModelConfig(**fields)


def deep_agents_model(role_id: str = "qwen-deep-agents", **overrides: Any) -> ModelConfig:
    """Deep Agents model targeting vLLM directly.

    ``create_chat_model`` routes ``endpoint_base_url`` + ``endpoint_api_key``
    through ``init_chat_model`` with ``model_provider="openai"``
    (langchain_deep_agents/initialization.py). The default FilesystemBackend
    needs no container.
    """
    fields: dict[str, Any] = {
        "id": role_id,
        "model_name": LIVE_VLLM_MODEL,
        "model_provider": "openai",
        "interface": "langchain_deep_agents",
        "endpoint_base_url": LIVE_VLLM_URL,
        "endpoint_api_key": "EMPTY",
        "temperature": 0.0,
        "request_timeout": 180.0,
    }
    fields.update(overrides)
    return ModelConfig(**fields)


def claude_sdk_container_model(role_id: str = "qwen-claude-sdk-container", **overrides: Any) -> ModelConfig:
    """claude_agent_sdk model running the pinned CLI inside a docker container.

    The ``agent_runtime`` block activates ``docker_cli_wrapper.py`` as
    ``cli_path`` (claude_agent_sdk/agent.py). The wrapper forwards
    ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY``, auto-adds ``--add-host``
    for the base-url hostname, and mounts the workspace at ``/workspace``.
    The image must contain Claude Code CLI pinned to 2.1.146 (newer CLIs
    inject system-role skill messages that vLLM rejects with HTTP 400).
    """
    fields: dict[str, Any] = {
        "id": role_id,
        "model_name": LIVE_VLLM_MODEL,
        "interface": "claude_agent_sdk",
        "anthropic_base_url": LIVE_VLLM_URL,
        "anthropic_api_key": "EMPTY",
        "temperature": 0.0,
        "extra_kwargs": {
            "agent_runtime": {
                "backend": "docker",
                "container_runtime": "docker",
                "container_image": CLAUDE_CLI_IMAGE,
                "container_network": "bridge",
            }
        },
    }
    fields.update(overrides)
    return ModelConfig(**fields)


# ---------------------------------------------------------------------------
# Retry policies and dead-port models (B9, B3 timeout sub-case)
# ---------------------------------------------------------------------------


def tight_retry_policy() -> RetryPolicy:
    """Small retry budgets with zero backoff so dead-port tests fail fast.

    Connection gets 2 attempts so a recorded retry count of >= 1 is
    unambiguous. The other categories are minimal. ``derive_sdk_max_retries``
    over this policy is 2, which bounds the internal SDK retries of the
    adapters that still delegate retrying to their SDK on the baseline.
    """
    return RetryPolicy(
        connection=CategoryRetryConfig(max_attempts=2, backoff_min=0.0, backoff_max=0.0),
        timeout=CategoryRetryConfig(max_attempts=1, backoff_min=0.0, backoff_max=0.0),
        rate_limit=CategoryRetryConfig(max_attempts=0, backoff_min=0.0, backoff_max=0.0),
        server_error=CategoryRetryConfig(max_attempts=0, backoff_min=0.0, backoff_max=0.0),
    )


def zero_retry_policy() -> RetryPolicy:
    """All retry budgets at zero, for tests that need a single attempt."""
    zero = CategoryRetryConfig(max_attempts=0, backoff_min=0.0, backoff_max=0.0)
    return RetryPolicy(connection=zero, timeout=zero, rate_limit=zero, server_error=zero)


def dead_port_model(interface: str) -> ModelConfig:
    """Model pointing at a dead local port, with a tight RetryPolicy.

    Used by B9 to exercise retry plus telemetry without the GPU. The
    connection is refused immediately, so each attempt costs milliseconds.

    Args:
        interface: One of ``openai_endpoint``, ``claude_tool``,
            ``langchain_deep_agents``, ``claude_agent_sdk``.
    """
    fields: dict[str, Any] = {
        "id": f"dead-{interface}",
        "model_name": LIVE_VLLM_MODEL,
        "interface": interface,
        "temperature": 0.0,
        "max_tokens": 64,
        "request_timeout": 2.0,
        "retry_policy": tight_retry_policy(),
    }
    if interface in ("openai_endpoint", "langchain_deep_agents"):
        fields["endpoint_base_url"] = DEAD_PORT_URL
        fields["endpoint_api_key"] = "EMPTY"
        if interface == "langchain_deep_agents":
            fields["model_provider"] = "openai"
        else:
            fields["extra_kwargs"] = {"max_retries": 0}
    elif interface in ("claude_tool", "claude_agent_sdk"):
        fields["anthropic_base_url"] = DEAD_PORT_URL
        fields["anthropic_api_key"] = "EMPTY"
    else:
        raise ValueError(f"unsupported interface for dead_port_model: {interface}")
    return ModelConfig(**fields)


# ---------------------------------------------------------------------------
# In-flight counting wrapper (B6)
# ---------------------------------------------------------------------------


class InFlightCounter:
    """Thread-safe in-flight call counter for concurrency-cap assertions.

    ``enter``/``exit`` are called from coroutines running on the worker
    portal's event loop and from plain threads, so a ``threading.Lock``
    (never held across an await) guards all state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = 0
        self.max_observed = 0
        self.total_calls = 0

    def enter(self) -> None:
        """Record one call entering flight."""
        with self._lock:
            self._current += 1
            self.total_calls += 1
            self.max_observed = max(self.max_observed, self._current)

    def exit(self) -> None:
        """Record one call leaving flight."""
        with self._lock:
            self._current -= 1


def counted_async(
    fn: Callable[..., Coroutine[Any, Any, Any]],
    counter: InFlightCounter,
    settle_seconds: float = 0.05,
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Wrap an async adapter method with in-flight counting.

    The small sleep before delegating widens the overlap window so that
    truly concurrent callers are observed as concurrent instead of racing
    past each other between increment and the actual network call.

    Args:
        fn: The original unbound async method (e.g.
            ``LangChainLLMAdapter.ainvoke``).
        counter: Shared InFlightCounter instance.
        settle_seconds: Sleep inserted before delegation to force overlap.

    Returns:
        Wrapped coroutine function suitable for ``monkeypatch.setattr`` on
        the adapter class.
    """

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        counter.enter()
        try:
            await asyncio.sleep(settle_seconds)
            return await fn(*args, **kwargs)
        finally:
            counter.exit()

    return wrapper


# ---------------------------------------------------------------------------
# Baseline workaround: langchain-openai global client cache (deep_agents)
# ---------------------------------------------------------------------------


def reset_langchain_openai_client_cache() -> None:
    """Clear langchain-openai's module-global cached httpx clients.

    Baseline bug (documented in the daily note, surfaced while authoring
    this suite): langchain-openai caches one async httpx client per
    (base_url, timeout) in ``chat_models/_client_utils.py``. The
    deep_agents adapter does not inject per-model clients (the langchain
    adapter does), so two deep_agents calls on different event loops within
    httpx's keepalive window (about 5 seconds) reuse a pooled TCP
    connection bound to the first, already-closed loop and fail with
    ``RuntimeError: Event loop is closed`` surfacing as APIConnectionError.

    Clearing the cache before entering a fresh event loop makes the
    deep_agents tests deterministic on the unmodified baseline. T1 (retry
    routing through RetryExecutor) makes production robust to this because
    the connection error becomes retryable, after which this workaround can
    be dropped from the flipped tests.
    """
    from langchain_openai.chat_models import _client_utils

    _client_utils._cached_sync_httpx_client.cache_clear()
    _client_utils._cached_async_httpx_client.cache_clear()


# ---------------------------------------------------------------------------
# Teardown hygiene (B7, B8, B10)
# ---------------------------------------------------------------------------


def find_teardown_problems(texts: Iterable[str]) -> list[str]:
    """Return the subset of texts that look like async teardown failures."""
    return [text for text in texts if any(marker in text for marker in TEARDOWN_ERROR_MARKERS)]


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def require_adapter(interface: str) -> None:
    """Skip the calling test when the adapter for ``interface`` is unavailable."""
    availability = check_adapter_available(interface)
    if not availability.available:
        pytest.skip(f"adapter '{interface}' unavailable: {availability.reason}")


@functools.cache
def docker_gate_reason() -> str | None:
    """Cheap docker preflight for container-backed tests (B10).

    Mirrors what ``preflight_container_runtime`` checks (docker binary,
    reachable daemon, pinned image present) but returns a skip reason
    instead of raising, so the suite stays runnable on machines without
    Colima. Returns None when the gate is open.
    """
    if os.getenv("KARENINA_LIVE_DOCKER_TESTS") != "1":
        return "set KARENINA_LIVE_DOCKER_TESTS=1 to run container-backed tests"
    if shutil.which("docker") is None:
        return "docker binary not found in PATH"
    try:
        info = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "docker info timed out (daemon unreachable)"
    if info.returncode != 0:
        return "docker daemon not reachable (start Colima: colima start --runtime docker)"
    inspect = subprocess.run(
        ["docker", "image", "inspect", CLAUDE_CLI_IMAGE],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if inspect.returncode != 0:
        return f"container image {CLAUDE_CLI_IMAGE} not present (build via tests/live/docker/)"
    return None
