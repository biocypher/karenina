"""Tests for Claude Agent SDK parser retry configuration from RetryPolicy.

Verifies that ClaudeSDKParserAdapter derives max_retries from RetryPolicy
and passes it to the Anthropic/OpenAI SDK constructors it creates internally.
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
    )


@pytest.mark.unit
class TestClaudeSDKParserAnthropicRetry:
    """Verify max_retries is passed to the Anthropic client in the parser."""

    def test_anthropic_client_receives_default_max_retries(self) -> None:
        """Default RetryPolicy yields max_retries=5."""
        config = _make_config()
        parser = ClaudeSDKParserAdapter(config)

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            parser._get_anthropic_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 5

    def test_anthropic_client_custom_policy(self) -> None:
        """Custom policy propagates to Anthropic client."""
        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=8),
        )
        config = _make_config(retry_policy=policy)
        parser = ClaudeSDKParserAdapter(config)

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            parser._get_anthropic_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 8

    def test_anthropic_client_zero_policy(self) -> None:
        """All-zero policy passes max_retries=0."""
        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=0),
            timeout=CategoryRetryConfig(max_attempts=0),
            rate_limit=CategoryRetryConfig(max_attempts=0),
            server_error=CategoryRetryConfig(max_attempts=0),
        )
        config = _make_config(retry_policy=policy)
        parser = ClaudeSDKParserAdapter(config)

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            parser._get_anthropic_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 0


@pytest.mark.unit
class TestClaudeSDKParserOpenAIRetry:
    """Verify max_retries is passed to the OpenAI client in the parser."""

    def test_openai_client_receives_default_max_retries(self) -> None:
        """Default RetryPolicy yields max_retries=5 for OpenAI client."""
        config = _make_config(anthropic_base_url="http://localhost:8000/v1")
        parser = ClaudeSDKParserAdapter(config)

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            parser._get_openai_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 5

    def test_openai_client_custom_policy(self) -> None:
        """Custom policy propagates to OpenAI client."""
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
            assert kwargs["max_retries"] == 6

    def test_openai_client_zero_policy(self) -> None:
        """All-zero policy passes max_retries=0 to OpenAI client."""
        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=0),
            timeout=CategoryRetryConfig(max_attempts=0),
            rate_limit=CategoryRetryConfig(max_attempts=0),
            server_error=CategoryRetryConfig(max_attempts=0),
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
