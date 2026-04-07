"""Tests for retry policy, configuration, and executor."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from karenina.utils.errors import ErrorRegistry
from karenina.utils.retry_policy import (
    CategoryRetryConfig,
    ErrorPatternConfig,
    RetryExecutor,
    RetryPolicy,
)

# ---------------------------------------------------------------------------
# CategoryRetryConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCategoryRetryConfig:
    """CategoryRetryConfig default values and validation."""

    def test_default_values(self) -> None:
        config = CategoryRetryConfig()
        assert config.max_attempts == 0
        assert config.backoff_min == 1.0
        assert config.backoff_max == 10.0
        assert config.backoff_multiplier == 2.0

    def test_custom_values(self) -> None:
        config = CategoryRetryConfig(
            max_attempts=5,
            backoff_min=0.5,
            backoff_max=20.0,
            backoff_multiplier=3.0,
        )
        assert config.max_attempts == 5
        assert config.backoff_min == 0.5
        assert config.backoff_max == 20.0
        assert config.backoff_multiplier == 3.0

    def test_max_attempts_zero_allowed(self) -> None:
        config = CategoryRetryConfig(max_attempts=0)
        assert config.max_attempts == 0

    def test_max_attempts_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CategoryRetryConfig(max_attempts=-1)

    def test_backoff_min_zero_allowed(self) -> None:
        config = CategoryRetryConfig(backoff_min=0.0)
        assert config.backoff_min == 0.0

    def test_backoff_min_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CategoryRetryConfig(backoff_min=-1.0)

    def test_backoff_multiplier_below_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CategoryRetryConfig(backoff_multiplier=0.5)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CategoryRetryConfig(bogus_field=True)

    def test_serialization_roundtrip(self) -> None:
        config = CategoryRetryConfig(max_attempts=4, backoff_min=2.0)
        data = config.model_dump()
        restored = CategoryRetryConfig.model_validate(data)
        assert restored == config


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetryPolicy:
    """RetryPolicy defaults and derive_sdk_max_retries."""

    def test_default_connection(self) -> None:
        policy = RetryPolicy()
        assert policy.connection.max_attempts == 3
        assert policy.connection.backoff_min == 1.0
        assert policy.connection.backoff_max == 10.0

    def test_default_timeout(self) -> None:
        policy = RetryPolicy()
        assert policy.timeout.max_attempts == 3
        assert policy.timeout.backoff_min == 5.0
        assert policy.timeout.backoff_max == 30.0

    def test_default_rate_limit(self) -> None:
        policy = RetryPolicy()
        assert policy.rate_limit.max_attempts == 5
        assert policy.rate_limit.backoff_min == 5.0
        assert policy.rate_limit.backoff_max == 30.0

    def test_default_server_error(self) -> None:
        policy = RetryPolicy()
        assert policy.server_error.max_attempts == 2
        assert policy.server_error.backoff_min == 2.0
        assert policy.server_error.backoff_max == 15.0

    def test_derive_sdk_max_retries_defaults(self) -> None:
        policy = RetryPolicy()
        # max of (3, 3, 5, 2) = 5
        assert policy.derive_sdk_max_retries() == 5

    def test_derive_sdk_max_retries_custom(self) -> None:
        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=1),
            timeout=CategoryRetryConfig(max_attempts=10),
            rate_limit=CategoryRetryConfig(max_attempts=2),
            server_error=CategoryRetryConfig(max_attempts=0),
        )
        assert policy.derive_sdk_max_retries() == 10

    def test_derive_sdk_max_retries_all_zero(self) -> None:
        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=0),
            timeout=CategoryRetryConfig(max_attempts=0),
            rate_limit=CategoryRetryConfig(max_attempts=0),
            server_error=CategoryRetryConfig(max_attempts=0),
        )
        assert policy.derive_sdk_max_retries() == 0

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetryPolicy(unknown_category=CategoryRetryConfig())

    def test_serialization_roundtrip(self) -> None:
        policy = RetryPolicy(
            timeout=CategoryRetryConfig(max_attempts=5, backoff_min=0.1),
        )
        data = policy.model_dump()
        restored = RetryPolicy.model_validate(data)
        assert restored == policy


# ---------------------------------------------------------------------------
# ErrorPatternConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestErrorPatternConfig:
    """ErrorPatternConfig Pydantic model."""

    def test_message_substring_default(self) -> None:
        config = ErrorPatternConfig(pattern="custom", category="connection")
        assert config.match_type == "message_substring"

    def test_type_name_match_type(self) -> None:
        config = ErrorPatternConfig(
            pattern="MyError",
            category="timeout",
            match_type="type_name",
        )
        assert config.match_type == "type_name"
        assert config.category == "timeout"

    def test_invalid_category_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ErrorPatternConfig(pattern="x", category="bogus")

    def test_invalid_match_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ErrorPatternConfig(pattern="x", category="connection", match_type="regex")

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ErrorPatternConfig(pattern="x", category="connection", extra_field="y")

    def test_serialization_roundtrip(self) -> None:
        config = ErrorPatternConfig(
            pattern="SomeError",
            category="server_error",
            match_type="type_name",
        )
        data = config.model_dump()
        restored = ErrorPatternConfig.model_validate(data)
        assert restored == config


# ---------------------------------------------------------------------------
# RetryExecutor (sync)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetryExecutorSync:
    """RetryExecutor.execute: sync category-aware retry logic."""

    def _make_policy(self, **overrides: CategoryRetryConfig) -> RetryPolicy:
        """Create a policy with zero-delay backoffs for fast tests."""
        defaults = {
            "connection": CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
            "timeout": CategoryRetryConfig(max_attempts=1, backoff_min=0, backoff_max=0),
            "rate_limit": CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
            "server_error": CategoryRetryConfig(max_attempts=2, backoff_min=0, backoff_max=0),
        }
        defaults.update(overrides)
        return RetryPolicy(**defaults)

    def test_success_on_first_call(self) -> None:
        executor = RetryExecutor(self._make_policy(), ErrorRegistry())
        result = executor.execute(lambda: 42)
        assert result == 42

    def test_retries_on_connection_error(self) -> None:
        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("connection refused")
            return "ok"

        executor = RetryExecutor(self._make_policy(), ErrorRegistry())
        result = executor.execute(fn)
        assert result == "ok"
        assert call_count == 3

    def test_budget_exhaustion_raises(self) -> None:
        """When retries are exhausted, the last exception is re-raised."""

        def fn() -> None:
            raise ConnectionError("always fails")

        policy = self._make_policy(
            connection=CategoryRetryConfig(max_attempts=2, backoff_min=0, backoff_max=0),
        )
        executor = RetryExecutor(policy, ErrorRegistry())
        with pytest.raises(ConnectionError, match="always fails"):
            executor.execute(fn)

    def test_permanent_error_not_retried(self) -> None:
        call_count = 0

        def fn() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        executor = RetryExecutor(self._make_policy(), ErrorRegistry())
        with pytest.raises(ValueError, match="bad input"):
            executor.execute(fn)
        assert call_count == 1

    def test_timeout_limited_retry(self) -> None:
        """Timeout category has max_attempts=1, so only 1 retry (2 total calls)."""
        call_count = 0

        def fn() -> None:
            nonlocal call_count
            call_count += 1
            raise TimeoutError("timed out")

        executor = RetryExecutor(self._make_policy(), ErrorRegistry())
        with pytest.raises(TimeoutError):
            executor.execute(fn)
        # 1 initial + 1 retry = 2 total
        assert call_count == 2

    def test_zero_max_attempts_means_no_retry(self) -> None:
        call_count = 0

        def fn() -> None:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        policy = self._make_policy(
            connection=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
        )
        executor = RetryExecutor(policy, ErrorRegistry())
        with pytest.raises(ConnectionError):
            executor.execute(fn)
        assert call_count == 1

    def test_budget_resets_between_execute_calls(self) -> None:
        """Each execute() call gets fresh per-category budgets."""
        attempt = 0

        def fn() -> str:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise ConnectionError("first")
            return "ok"

        executor = RetryExecutor(self._make_policy(), ErrorRegistry())
        # First call: fails once, then succeeds
        result1 = executor.execute(fn)
        assert result1 == "ok"

        # Reset for second call
        attempt = 0
        result2 = executor.execute(fn)
        assert result2 == "ok"

    def test_mixed_error_types_use_separate_budgets(self) -> None:
        """Different error categories consume their own budgets independently."""
        sequence = iter(
            [
                ConnectionError("net"),
                ConnectionError("net"),
                TimeoutError("slow"),
                None,  # success
            ]
        )

        def fn() -> str:
            exc = next(sequence)
            if exc is not None:
                raise exc
            return "done"

        # connection: 3 retries, timeout: 1 retry
        executor = RetryExecutor(self._make_policy(), ErrorRegistry())
        result = executor.execute(fn)
        assert result == "done"

    def test_mixed_errors_budget_exhaustion(self) -> None:
        """When one category's budget runs out, even if other categories have budget."""
        sequence = iter(
            [
                ConnectionError("net"),
                ConnectionError("net"),
                ConnectionError("net"),
                ConnectionError("net"),  # 4th connection error, budget is 3
            ]
        )

        def fn() -> str:
            exc = next(sequence)
            if exc is not None:
                raise exc
            return "done"

        executor = RetryExecutor(self._make_policy(), ErrorRegistry())
        with pytest.raises(ConnectionError):
            executor.execute(fn)

    def test_passes_args_and_kwargs(self) -> None:
        def fn(a: int, b: int, c: int = 0) -> int:
            return a + b + c

        executor = RetryExecutor(self._make_policy(), ErrorRegistry())
        result = executor.execute(fn, 1, 2, c=3)
        assert result == 6


# ---------------------------------------------------------------------------
# RetryExecutor (async)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetryExecutorAsync:
    """RetryExecutor.aexecute: async category-aware retry logic."""

    def _make_policy(self, **overrides: CategoryRetryConfig) -> RetryPolicy:
        defaults = {
            "connection": CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
            "timeout": CategoryRetryConfig(max_attempts=1, backoff_min=0, backoff_max=0),
            "rate_limit": CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
            "server_error": CategoryRetryConfig(max_attempts=2, backoff_min=0, backoff_max=0),
        }
        defaults.update(overrides)
        return RetryPolicy(**defaults)

    async def test_success_on_first_call(self) -> None:
        executor = RetryExecutor(self._make_policy(), ErrorRegistry())

        async def fn() -> int:
            return 42

        result = await executor.aexecute(fn)
        assert result == 42

    async def test_retries_on_connection_error(self) -> None:
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("refused")
            return "ok"

        executor = RetryExecutor(self._make_policy(), ErrorRegistry())
        result = await executor.aexecute(fn)
        assert result == "ok"
        assert call_count == 3

    async def test_budget_exhaustion_raises(self) -> None:
        async def fn() -> None:
            raise ConnectionError("always fails")

        policy = self._make_policy(
            connection=CategoryRetryConfig(max_attempts=2, backoff_min=0, backoff_max=0),
        )
        executor = RetryExecutor(policy, ErrorRegistry())
        with pytest.raises(ConnectionError, match="always fails"):
            await executor.aexecute(fn)

    async def test_permanent_error_not_retried(self) -> None:
        call_count = 0

        async def fn() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        executor = RetryExecutor(self._make_policy(), ErrorRegistry())
        with pytest.raises(ValueError, match="bad input"):
            await executor.aexecute(fn)
        assert call_count == 1

    async def test_passes_args_and_kwargs(self) -> None:
        async def fn(a: int, b: int, c: int = 0) -> int:
            return a + b + c

        executor = RetryExecutor(self._make_policy(), ErrorRegistry())
        result = await executor.aexecute(fn, 1, 2, c=3)
        assert result == 6

    async def test_zero_max_attempts_means_no_retry(self) -> None:
        call_count = 0

        async def fn() -> None:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        policy = self._make_policy(
            connection=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
        )
        executor = RetryExecutor(policy, ErrorRegistry())
        with pytest.raises(ConnectionError):
            await executor.aexecute(fn)
        assert call_count == 1
