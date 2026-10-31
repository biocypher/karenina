"""Shared fixtures for Codex SDK adapter tests.

The adapter discriminates codex thread items by their ``type`` string, so
tests use lightweight stand-in objects instead of the real pydantic
models. The agent tests install a fake ``openai_codex`` module in
sys.modules so no app-server subprocess or network is involved.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from karenina.schemas.config import ModelConfig


@pytest.fixture
def endpoint_model_config() -> ModelConfig:
    """ModelConfig for the custom-endpoint (vLLM-style) path."""
    return ModelConfig(
        id="qwen-codex",
        model_name="qwen3.5-122b-a10b",
        model_provider="openai",
        interface="codex_sdk",
        endpoint_base_url="http://example-endpoint:8000/v1",
    )


@pytest.fixture
def native_model_config() -> ModelConfig:
    """ModelConfig for the native OpenAI path (no custom endpoint)."""
    return ModelConfig(
        id="gpt-codex",
        model_name="gpt-5.2-codex",
        interface="codex_sdk",
    )


def make_item(item_type: str, **fields: Any) -> SimpleNamespace:
    """Build a fake codex thread item with a type discriminator."""
    return SimpleNamespace(type=item_type, **fields)


def make_reasoning(item_id: str = "item_r", content: list[str] | None = None) -> SimpleNamespace:
    return make_item("reasoning", id=item_id, content=content or ["thinking about it"], summary=[])


def make_agent_message(text: str, item_id: str = "item_a", phase: str | None = None) -> SimpleNamespace:
    return make_item("agentMessage", id=item_id, text=text, phase=phase)


def make_command_execution(
    item_id: str = "item_c",
    command: str = "ls -la",
    output: str = "total 0",
    exit_code: int | None = 0,
    status: str = "completed",
) -> SimpleNamespace:
    return make_item(
        "commandExecution",
        id=item_id,
        command=command,
        aggregated_output=output,
        exit_code=exit_code,
        status=status,
        cwd="/tmp/work",
        duration_ms=12,
    )


def make_usage(
    input_tokens: int = 100,
    output_tokens: int = 20,
    cached: int = 5,
    reasoning: int = 3,
) -> SimpleNamespace:
    """Build a fake ThreadTokenUsage with last/total breakdowns."""
    breakdown = SimpleNamespace(
        cached_input_tokens=cached,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_output_tokens=reasoning,
        total_tokens=input_tokens + output_tokens,
    )
    half = SimpleNamespace(
        cached_input_tokens=0,
        input_tokens=input_tokens // 2,
        output_tokens=output_tokens // 2,
        reasoning_output_tokens=0,
        total_tokens=(input_tokens + output_tokens) // 2,
    )
    return SimpleNamespace(last=half, total=breakdown, model_context_window=258400)
