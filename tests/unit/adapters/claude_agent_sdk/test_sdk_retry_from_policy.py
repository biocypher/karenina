"""Tests for Claude Agent SDK parser SDK retry suppression (design decision D1).

Since T2, transient retries are owned by RetryExecutor at the adapter layer
and the Anthropic/OpenAI SDK clients created by the parser are constructed
with max_retries=0 regardless of the configured RetryPolicy. See
test_parser_retry_routing.py for the RetryExecutor routing itself.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from karenina.adapters.claude_agent_sdk.parser import ClaudeSDKParserAdapter
from karenina.schemas.config import ModelConfig
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy


def _make_config(
    *,
    retry_policy: RetryPolicy | None = None,
    anthropic_base_url: str | None = None,
    extra_kwargs: dict | None = None,
) -> ModelConfig:
    """Create a ModelConfig for claude_agent_sdk parser tests."""
    return ModelConfig(
        id="test-claude-sdk",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
        interface="claude_agent_sdk",
        max_tokens=1024,
        retry_policy=retry_policy,
        anthropic_base_url=anthropic_base_url,
        extra_kwargs=extra_kwargs,
    )


@pytest.mark.unit
class TestClaudeSDKParserAnthropicRetrySuppression:
    """The Anthropic client always receives max_retries=0."""

    def test_anthropic_client_receives_zero_max_retries(self) -> None:
        """Default RetryPolicy still yields SDK max_retries=0."""
        config = _make_config()
        parser = ClaudeSDKParserAdapter(config)

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            parser._get_anthropic_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 0

    def test_anthropic_client_custom_policy_not_leaked(self) -> None:
        """A generous RetryPolicy budget never re-enables SDK retries."""
        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=8),
        )
        config = _make_config(retry_policy=policy)
        parser = ClaudeSDKParserAdapter(config)

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            parser._get_anthropic_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 0


@pytest.mark.unit
class TestClaudeSDKParserOpenAIRetrySuppression:
    """The OpenAI client always receives max_retries=0."""

    def test_openai_client_receives_zero_max_retries(self) -> None:
        """Default RetryPolicy still yields SDK max_retries=0."""
        config = _make_config(anthropic_base_url="http://localhost:8000/v1")
        parser = ClaudeSDKParserAdapter(config)

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            parser._get_openai_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 0

    def test_openai_client_custom_policy_not_leaked(self) -> None:
        """A generous RetryPolicy budget never re-enables SDK retries."""
        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=6),
        )
        config = _make_config(
            retry_policy=policy,
            anthropic_base_url="http://localhost:8000/v1",
        )
        parser = ClaudeSDKParserAdapter(config)

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            parser._get_openai_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 0

    def test_openai_client_uses_explicit_parser_base_url_override(self) -> None:
        """Z.ai-style endpoint layouts can override the OpenAI parser base URL."""
        config = _make_config(
            anthropic_base_url="https://api.z.ai/api/anthropic",
            extra_kwargs={
                "claude_sdk_parser_openai_base_url": "https://api.z.ai/api/coding/paas/v4",
            },
        )
        parser = ClaudeSDKParserAdapter(config)

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            parser._get_openai_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["base_url"] == "https://api.z.ai/api/coding/paas/v4"
