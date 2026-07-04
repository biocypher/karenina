"""Tests for the Claude Agent SDK subscription auth helper."""

from __future__ import annotations

import pytest

from karenina.adapters.claude_agent_sdk.auth import (
    SUBSCRIPTION_AUTH_ENV_VARS,
    subscription_auth_env,
)


@pytest.mark.unit
class TestSubscriptionAuthEnv:
    def test_returns_empty_when_no_token_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in SUBSCRIPTION_AUTH_ENV_VARS:
            monkeypatch.delenv(name, raising=False)

        assert subscription_auth_env() == {}

    def test_forwards_oauth_token_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in SUBSCRIPTION_AUTH_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-claude-oauth-test")

        env = subscription_auth_env()

        assert env == {"CLAUDE_CODE_OAUTH_TOKEN": "sk-claude-oauth-test"}

    def test_forwards_multiple_subscription_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-value")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-value")

        env = subscription_auth_env()

        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-value"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "auth-value"
