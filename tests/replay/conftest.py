"""Shared fixtures for replay tests.

Provides a live vLLM-backed ModelConfig for end-to-end tests that want
to exercise the real pipeline against a real model. Gated on the
KARENINA_LIVE_TESTS environment variable AND on TCP reachability of
the endpoint, so the default (offline) test run is never affected.

Live endpoint:
  codon-gpu-001.ebi.ac.uk:8000 (internal), http://codon-gpu-001:8000
  vLLM serving Qwen 3.5 122B (MoE, 10B active), 4x H200 GPUs
  OpenAI-compatible API, api_key="EMPTY"

To run the live-marked tests:
  KARENINA_LIVE_TESTS=1 uv run pytest tests/replay/ -m live -q
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest

from karenina.schemas.config import ModelConfig

LIVE_ENDPOINT_URL = "http://codon-gpu-001:8000"
LIVE_MODEL_NAME = "qwen3.5-122b-a10b"
LIVE_MODEL_ID = "qwen3.5-122b-a10b"
LIVE_REACHABILITY_TIMEOUT = 3.0  # seconds


def _live_tests_enabled() -> bool:
    """Return True if the KARENINA_LIVE_TESTS env var is truthy."""
    value = os.environ.get("KARENINA_LIVE_TESTS", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _endpoint_reachable(url: str, timeout: float = LIVE_REACHABILITY_TIMEOUT) -> bool:
    """Return True if we can open a TCP connection to the endpoint host:port."""
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Skip @pytest.mark.live tests when the gate is off or endpoint is unreachable."""
    enabled = _live_tests_enabled()
    reachable = _endpoint_reachable(LIVE_ENDPOINT_URL) if enabled else False

    if enabled and not reachable:
        skip = pytest.mark.skip(reason=f"KARENINA_LIVE_TESTS=1 but {LIVE_ENDPOINT_URL} is unreachable")
    elif not enabled:
        skip = pytest.mark.skip(reason="KARENINA_LIVE_TESTS not set (live tests are opt-in)")
    else:
        skip = None

    if skip is None:
        return

    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def live_primary_model() -> ModelConfig:
    """Answering-side ModelConfig pointing at the Qwen 122B vLLM endpoint."""
    return ModelConfig(
        id=LIVE_MODEL_ID,
        model_name=LIVE_MODEL_NAME,
        interface="openai_endpoint",
        endpoint_base_url=LIVE_ENDPOINT_URL,
        endpoint_api_key="EMPTY",
        temperature=0.0,
        extra_kwargs={
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "max_retries": 0,
        },
    )


@pytest.fixture
def live_parsing_model() -> ModelConfig:
    """Parsing-side ModelConfig pointing at the same Qwen 122B vLLM endpoint.

    Kept as a separate fixture so callers can easily swap in a different
    judge without touching the primary model.
    """
    return ModelConfig(
        id=LIVE_MODEL_ID,
        model_name=LIVE_MODEL_NAME,
        interface="openai_endpoint",
        endpoint_base_url=LIVE_ENDPOINT_URL,
        endpoint_api_key="EMPTY",
        temperature=0.0,
        extra_kwargs={
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "max_retries": 0,
        },
    )
