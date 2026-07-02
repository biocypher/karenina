"""Tests for the shared sync-wrapper timeout helper.

compute_sync_wrapper_timeout centralizes the wall-clock bounds the
adapter sync wrappers place around their async dispatch. The bound must
never tighten below the site's historical floor and must scale with
request_timeout, the retry policy, and the number of internal call
sequences.
"""

from __future__ import annotations

import pytest

from karenina.adapters._timeouts import (
    DEEP_AGENTS_SYNC_WRAPPER_FLOOR,
    DEFAULT_SYNC_WRAPPER_FLOOR,
    MCP_TOOL_FETCH_FLOOR,
    PARSE_INTERNAL_CALL_SEQUENCES,
    PORTAL_DISPATCH_FLOOR,
    SEARCH_PROVIDER_FLOOR,
    SYNC_WRAPPER_GRACE_SECONDS,
    compute_sync_wrapper_timeout,
)
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy, TimeoutEscalationConfig


@pytest.mark.unit
class TestFloorBehavior:
    """The returned bound never goes below the floor."""

    def test_none_request_timeout_returns_default_floor(self) -> None:
        assert compute_sync_wrapper_timeout(None) == DEFAULT_SYNC_WRAPPER_FLOOR

    def test_none_request_timeout_returns_explicit_floor(self) -> None:
        assert compute_sync_wrapper_timeout(None, floor=45.0) == 45.0

    def test_small_request_timeout_clamps_to_floor(self) -> None:
        # Default policy: 4 attempts of 10s plus 90s backoff plus 30s grace
        # is 160s, below the 300s floor.
        assert compute_sync_wrapper_timeout(10.0) == DEFAULT_SYNC_WRAPPER_FLOOR

    def test_historical_floor_constants_are_preserved(self) -> None:
        assert DEFAULT_SYNC_WRAPPER_FLOOR == 300.0
        assert DEEP_AGENTS_SYNC_WRAPPER_FLOOR == 600.0
        assert PORTAL_DISPATCH_FLOOR == 600.0
        assert MCP_TOOL_FETCH_FLOOR == 45.0
        assert SEARCH_PROVIDER_FLOOR == 60.0


@pytest.mark.unit
class TestDerivedBound:
    """Bound derivation from request_timeout and the retry policy."""

    def test_large_request_timeout_exceeds_floor(self) -> None:
        # Default timeout policy: max_attempts=3, backoff_max=30.
        # (4 attempts * 100s) + (3 * 30s backoff) + 30s grace = 520s.
        assert compute_sync_wrapper_timeout(100.0) == 520.0

    def test_internal_call_sequences_multiply_the_budget(self) -> None:
        # Per sequence: (4 * 100) + 90 = 490. Times 4 sequences plus grace.
        expected = 490.0 * 4 + SYNC_WRAPPER_GRACE_SECONDS
        assert compute_sync_wrapper_timeout(100.0, internal_call_sequences=4) == expected

    def test_custom_retry_policy_attempt_count(self) -> None:
        policy = RetryPolicy(timeout=CategoryRetryConfig(max_attempts=1, backoff_max=5.0))
        # (2 attempts * 200) + (1 * 5) + 30 = 435.
        assert compute_sync_wrapper_timeout(200.0, retry_policy=policy) == 435.0

    def test_zero_timeout_retries_collapse_to_single_attempt(self) -> None:
        policy = RetryPolicy(timeout=CategoryRetryConfig(max_attempts=0))
        # 1 attempt * 400 + 0 backoff + 30 grace.
        assert compute_sync_wrapper_timeout(400.0, retry_policy=policy) == 430.0

    def test_escalation_sums_escalated_attempts(self) -> None:
        policy = RetryPolicy(
            timeout=CategoryRetryConfig(max_attempts=3, backoff_max=30.0),
            timeout_escalation=TimeoutEscalationConfig(strategy="additive", increment=10.0),
        )
        # Attempts: 100, 110, 120, 130 = 460. Plus 90 backoff plus 30 grace.
        assert compute_sync_wrapper_timeout(100.0, retry_policy=policy) == 580.0

    def test_parse_sequence_constant_matches_parser_flow(self) -> None:
        assert PARSE_INTERNAL_CALL_SEQUENCES == 4


@pytest.mark.unit
class TestPortalBoundDelegation:
    """The langchain parser portal bound delegates to the shared helper."""

    def test_portal_bound_none_when_request_timeout_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from karenina.adapters.langchain.parser import LangChainParserAdapter
        from karenina.schemas.config import ModelConfig

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config = ModelConfig(
            id="t6-parser",
            model_name="gpt-4.1-mini",
            model_provider="openai",
            interface="langchain",
        )
        parser = LangChainParserAdapter(config)
        assert parser._compute_portal_timeout_bound() is None

    def test_portal_bound_matches_helper_with_four_sequences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from karenina.adapters.langchain.parser import LangChainParserAdapter
        from karenina.schemas.config import ModelConfig

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config = ModelConfig(
            id="t6-parser-bound",
            model_name="gpt-4.1-mini",
            model_provider="openai",
            interface="langchain",
            request_timeout=100.0,
        )
        parser = LangChainParserAdapter(config)
        expected = compute_sync_wrapper_timeout(
            100.0,
            retry_policy=config.retry_policy,
            internal_call_sequences=PARSE_INTERNAL_CALL_SEQUENCES,
        )
        assert parser._compute_portal_timeout_bound() == expected
