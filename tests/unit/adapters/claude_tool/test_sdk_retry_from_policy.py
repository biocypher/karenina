"""Tests for Claude Tool adapter SDK retry suppression (design decision D1).

Since T2, transient retries are owned by RetryExecutor at the adapter layer
and the Anthropic/AsyncAnthropic SDK clients are constructed with
max_retries=0 regardless of the configured RetryPolicy. See
test_retry_routing.py for the RetryExecutor routing itself.
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
class TestClaudeToolSDKRetrySuppression:
    """SDK clients always receive max_retries=0 (RetryExecutor owns retries)."""

    def test_sync_client_receives_zero_max_retries(self) -> None:
        """Default RetryPolicy still yields SDK max_retries=0."""
        config = _make_config()
        adapter = ClaudeToolLLMAdapter(config)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter._get_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 0

    def test_async_client_receives_zero_max_retries(self) -> None:
        """Default RetryPolicy still yields SDK max_retries=0 for async client."""
        config = _make_config()
        adapter = ClaudeToolLLMAdapter(config)

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter._get_async_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 0

    def test_custom_policy_does_not_leak_into_sdk(self) -> None:
        """A generous RetryPolicy budget never re-enables SDK retries."""
        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=7),
        )
        config = _make_config(retry_policy=policy)
        adapter = ClaudeToolLLMAdapter(config)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter._get_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 0

    def test_custom_policy_async_client_not_leaked(self) -> None:
        """Custom RetryPolicy never re-enables async SDK retries either."""
        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=7),
        )
        config = _make_config(retry_policy=policy)
        adapter = ClaudeToolLLMAdapter(config)

        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter._get_async_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 0

    def test_none_retry_policy_still_zero(self) -> None:
        """When retry_policy is None, the SDK clients still get max_retries=0."""
        config = _make_config(retry_policy=None)
        adapter = ClaudeToolLLMAdapter(config)

        with patch("anthropic.Anthropic") as mock_cls:
            mock_cls.return_value = MagicMock()
            adapter._get_client()

            _, kwargs = mock_cls.call_args
            assert kwargs["max_retries"] == 0
