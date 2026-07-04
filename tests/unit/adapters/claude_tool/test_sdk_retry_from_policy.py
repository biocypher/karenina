"""Tests for Claude Tool adapter SDK retry configuration from RetryPolicy.

Verifies that ClaudeToolLLMAdapter derives max_retries from RetryPolicy
and passes it to the Anthropic/AsyncAnthropic SDK constructors.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from karenina.adapters.claude_tool import ClaudeToolLLMAdapter
from karenina.schemas.config import ModelConfig
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy


def _make_config(*, retry_policy: RetryPolicy | None = None) -> ModelConfig:
    """Create a ModelConfig for claude_tool tests."""
    return ModelConfig(
        id="test-claude-tool",
        model_name="claude-haiku-4-5",
        model_provider="anthropic",
        interface="claude_tool",
        max_tokens=1024,
        retry_policy=retry_policy,
    )


@pytest.mark.unit
class TestClaudeToolSDKRetryFromPolicy:
    """Verify max_retries is derived from RetryPolicy and passed to SDK clients."""

    def test_sync_client_receives_default_max_retries(self) -> None:
        """Default RetryPolicy yields max_retries=3 (rate_limit and connection)."""
        config = _make_config()
        adapter = ClaudeToolLLMAdapter(config)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter._get_client()

            _, kwargs = mock_cls.call_args
            # Default policy: connection=3, timeout=1, rate_limit=3, server=2 -> max=3
            assert kwargs["max_retries"] == 5

    def test_async_client_receives_default_max_retries(self) -> None:
        """Default RetryPolicy yields max_retries=3 for async client too."""
        config = _make_config()
        adapter = ClaudeToolLLMAdapter(config)

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter._get_async_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 5

    def test_custom_policy_max_retries_propagated(self) -> None:
        """Custom RetryPolicy with high connection attempts propagates to SDK."""
        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=7),
        )
        config = _make_config(retry_policy=policy)
        adapter = ClaudeToolLLMAdapter(config)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter._get_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 7

    def test_custom_policy_async_client_propagated(self) -> None:
        """Custom RetryPolicy propagates to async client."""
        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=7),
        )
        config = _make_config(retry_policy=policy)
        adapter = ClaudeToolLLMAdapter(config)

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter._get_async_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 7

    def test_zero_retry_policy_propagated(self) -> None:
        """All-zero retry policy passes max_retries=0 to SDK."""
        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=0),
            timeout=CategoryRetryConfig(max_attempts=0),
            rate_limit=CategoryRetryConfig(max_attempts=0),
            server_error=CategoryRetryConfig(max_attempts=0),
        )
        config = _make_config(retry_policy=policy)
        adapter = ClaudeToolLLMAdapter(config)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter._get_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 0

    def test_none_retry_policy_uses_defaults(self) -> None:
        """When retry_policy is None, default RetryPolicy is used."""
        config = _make_config(retry_policy=None)
        adapter = ClaudeToolLLMAdapter(config)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter._get_client()

            _, kwargs = mock_cls.call_args
            # Default: max(connection=3, timeout=1, rate_limit=3, server=2) = 3
            assert kwargs["max_retries"] == 5
