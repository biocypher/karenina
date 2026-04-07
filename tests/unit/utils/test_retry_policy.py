"""Tests for retry policy, configuration, and executor."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from karenina.utils.errors import ErrorCategory, ErrorRegistry
from karenina.utils.retry_policy import (
    CategoryRetryConfig,
    ErrorPatternConfig,
    RetryExecutor,
    RetryPolicy,
    TimeoutEscalationConfig,
    compute_escalated_timeout,
    track_retries,
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


# ---------------------------------------------------------------------------
# track_retries (contextvar-based retry tracking)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrackRetries:
    """track_retries() context manager records used + budget per category."""

    def _make_policy(self) -> RetryPolicy:
        """Zero-delay policy for fast tests."""
        return RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
            timeout=CategoryRetryConfig(max_attempts=2, backoff_min=0, backoff_max=0),
            rate_limit=CategoryRetryConfig(max_attempts=4, backoff_min=0, backoff_max=0),
            server_error=CategoryRetryConfig(max_attempts=1, backoff_min=0, backoff_max=0),
        )

    def test_initial_tracker_carries_budgets(self) -> None:
        """Tracker is pre-populated with budgets and used=0 for every category."""
        with track_retries(self._make_policy()) as tracker:
            assert tracker == {
                "connection": {"used": 0, "budget": 3},
                "timeout": {"used": 0, "budget": 2},
                "rate_limit": {"used": 0, "budget": 4},
                "server_error": {"used": 0, "budget": 1},
            }

    def test_default_policy_used_when_none(self) -> None:
        """Passing None falls back to RetryPolicy() defaults for budgets."""
        with track_retries(None) as tracker:
            default = RetryPolicy()
            assert tracker["connection"]["budget"] == default.connection.max_attempts
            assert tracker["timeout"]["budget"] == default.timeout.max_attempts
            assert tracker["rate_limit"]["budget"] == default.rate_limit.max_attempts
            assert tracker["server_error"]["budget"] == default.server_error.max_attempts
            assert all(entry["used"] == 0 for entry in tracker.values())

    def test_retries_increment_used_count(self) -> None:
        """Each retry observed by RetryExecutor increments the matching used."""
        policy = self._make_policy()
        executor = RetryExecutor(policy, ErrorRegistry())

        attempts = {"connection": 0, "timeout": 0}

        def fail_then_succeed(category: str) -> str:
            attempts[category] += 1
            if attempts[category] == 1:
                if category == "connection":
                    raise ConnectionError("network down")
                raise TimeoutError("slow")
            return "ok"

        with track_retries(policy) as tracker:
            executor.execute(fail_then_succeed, "connection")
            executor.execute(fail_then_succeed, "timeout")

            assert tracker["connection"] == {"used": 1, "budget": 3}
            assert tracker["timeout"] == {"used": 1, "budget": 2}
            # Untouched categories retain budget and used=0
            assert tracker["rate_limit"] == {"used": 0, "budget": 4}
            assert tracker["server_error"] == {"used": 0, "budget": 1}

    def test_multiple_retries_in_one_call(self) -> None:
        """A function that retries N times produces used=N for that category."""
        policy = self._make_policy()
        executor = RetryExecutor(policy, ErrorRegistry())

        call_count = {"n": 0}

        def fn() -> str:
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("flaky")
            return "ok"

        with track_retries(policy) as tracker:
            executor.execute(fn)
            assert tracker["connection"]["used"] == 2

    def test_budget_exhaustion_still_counted(self) -> None:
        """Even when retries exhaust the budget, every retry attempt is counted."""
        policy = self._make_policy()
        executor = RetryExecutor(policy, ErrorRegistry())

        def fn() -> None:
            raise TimeoutError("always slow")

        with track_retries(policy) as tracker:
            with pytest.raises(TimeoutError):
                executor.execute(fn)
            # max_attempts=2 → 2 retries fired before raising
            assert tracker["timeout"]["used"] == 2
            assert tracker["timeout"]["budget"] == 2

    def test_no_retries_outside_context(self) -> None:
        """Retries that happen outside any track_retries() block are not recorded."""
        policy = self._make_policy()
        executor = RetryExecutor(policy, ErrorRegistry())

        call_count = {"n": 0}

        def fn() -> str:
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ConnectionError("flaky")
            return "ok"

        # No track_retries() wrapper: executor still retries, just nothing
        # gets recorded. Subsequent track_retries() should still start clean.
        executor.execute(fn)

        with track_retries(policy) as tracker:
            assert tracker["connection"]["used"] == 0

    def test_nested_contexts_isolate(self) -> None:
        """Inner track_retries() does not bleed retries into the outer one."""
        policy = self._make_policy()
        executor = RetryExecutor(policy, ErrorRegistry())

        counts = {"outer": 0, "inner": 0}

        def fail_once(key: str) -> str:
            counts[key] += 1
            if counts[key] == 1:
                raise ConnectionError("fail")
            return "ok"

        with track_retries(policy) as outer:
            executor.execute(fail_once, "outer")
            with track_retries(policy) as inner:
                executor.execute(fail_once, "inner")
                assert inner["connection"]["used"] == 1
            assert outer["connection"]["used"] == 1

    async def test_aexecute_retries_recorded(self) -> None:
        """The async retry path also feeds the tracker."""
        policy = self._make_policy()
        executor = RetryExecutor(policy, ErrorRegistry())

        call_count = {"n": 0}

        async def fn() -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise TimeoutError("slow")
            return "ok"

        with track_retries(policy) as tracker:
            await executor.aexecute(fn)
            assert tracker["timeout"]["used"] == 1

    def test_unknown_category_record_is_noop(self) -> None:
        """A retry for a category not in the pre-populated tracker is ignored.

        This guards against a future ErrorCategory being added without
        updating the initial tracker shape: we should never KeyError on
        record, just silently skip it.
        """
        from karenina.utils.retry_policy import _record_retry

        with track_retries(self._make_policy()) as tracker:
            _record_retry(ErrorCategory.PERMANENT)
            assert "permanent" not in tracker


# ---------------------------------------------------------------------------
# TimeoutEscalationConfig
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTimeoutEscalationConfig:
    """TimeoutEscalationConfig schema and per-strategy validation."""

    def test_additive_requires_positive_increment(self) -> None:
        with pytest.raises(ValidationError, match="additive strategy requires increment > 0"):
            TimeoutEscalationConfig(strategy="additive")

    def test_additive_with_increment_ok(self) -> None:
        config = TimeoutEscalationConfig(strategy="additive", increment=30.0)
        assert config.strategy == "additive"
        assert config.increment == 30.0

    def test_multiplicative_requires_multiplier_above_one(self) -> None:
        with pytest.raises(ValidationError, match="multiplicative strategy requires multiplier > 1.0"):
            TimeoutEscalationConfig(strategy="multiplicative", multiplier=1.0)

    def test_multiplicative_with_valid_multiplier_ok(self) -> None:
        config = TimeoutEscalationConfig(strategy="multiplicative", multiplier=2.0)
        assert config.multiplier == 2.0

    def test_linear_requires_max_timeout(self) -> None:
        with pytest.raises(ValidationError, match="linear strategy requires max_timeout"):
            TimeoutEscalationConfig(strategy="linear")

    def test_linear_with_max_timeout_ok(self) -> None:
        config = TimeoutEscalationConfig(strategy="linear", max_timeout=300.0)
        assert config.max_timeout == 300.0

    def test_invalid_strategy_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TimeoutEscalationConfig(strategy="bogus")

    def test_negative_increment_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TimeoutEscalationConfig(strategy="additive", increment=-5.0)

    def test_multiplier_below_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TimeoutEscalationConfig(strategy="multiplicative", multiplier=0.5)

    def test_negative_max_timeout_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TimeoutEscalationConfig(strategy="linear", max_timeout=-1.0)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TimeoutEscalationConfig(strategy="additive", increment=30.0, bogus=True)

    def test_serialization_roundtrip(self) -> None:
        config = TimeoutEscalationConfig(
            strategy="multiplicative",
            multiplier=1.5,
            max_timeout=600.0,
        )
        data = config.model_dump()
        restored = TimeoutEscalationConfig.model_validate(data)
        assert restored == config

    def test_retry_policy_default_no_escalation(self) -> None:
        """RetryPolicy.timeout_escalation defaults to None (backwards compatible)."""
        policy = RetryPolicy()
        assert policy.timeout_escalation is None

    def test_retry_policy_accepts_escalation(self) -> None:
        policy = RetryPolicy(
            timeout_escalation=TimeoutEscalationConfig(
                strategy="additive",
                increment=60.0,
            ),
        )
        assert policy.timeout_escalation is not None
        assert policy.timeout_escalation.strategy == "additive"


# ---------------------------------------------------------------------------
# compute_escalated_timeout
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeEscalatedTimeout:
    """compute_escalated_timeout helper across strategies."""

    def test_none_config_returns_base(self) -> None:
        assert compute_escalated_timeout(120.0, 3, None, max_attempts=3) == 120.0

    def test_none_base_returns_none(self) -> None:
        config = TimeoutEscalationConfig(strategy="additive", increment=10.0)
        assert compute_escalated_timeout(None, 2, config, max_attempts=3) is None

    def test_attempt_zero_returns_base(self) -> None:
        config = TimeoutEscalationConfig(strategy="multiplicative", multiplier=3.0)
        assert compute_escalated_timeout(120.0, 0, config, max_attempts=3) == 120.0

    def test_negative_attempt_returns_base(self) -> None:
        config = TimeoutEscalationConfig(strategy="multiplicative", multiplier=3.0)
        assert compute_escalated_timeout(120.0, -1, config, max_attempts=3) == 120.0

    def test_additive_growth(self) -> None:
        config = TimeoutEscalationConfig(strategy="additive", increment=60.0)
        assert compute_escalated_timeout(120.0, 1, config, max_attempts=3) == 180.0
        assert compute_escalated_timeout(120.0, 2, config, max_attempts=3) == 240.0
        assert compute_escalated_timeout(120.0, 3, config, max_attempts=3) == 300.0

    def test_additive_capped(self) -> None:
        config = TimeoutEscalationConfig(strategy="additive", increment=60.0, max_timeout=200.0)
        assert compute_escalated_timeout(120.0, 1, config, max_attempts=3) == 180.0
        assert compute_escalated_timeout(120.0, 2, config, max_attempts=3) == 200.0
        assert compute_escalated_timeout(120.0, 3, config, max_attempts=3) == 200.0

    def test_multiplicative_growth(self) -> None:
        config = TimeoutEscalationConfig(strategy="multiplicative", multiplier=2.0)
        assert compute_escalated_timeout(120.0, 1, config, max_attempts=3) == 240.0
        assert compute_escalated_timeout(120.0, 2, config, max_attempts=3) == 480.0
        assert compute_escalated_timeout(120.0, 3, config, max_attempts=3) == 960.0

    def test_multiplicative_capped(self) -> None:
        config = TimeoutEscalationConfig(
            strategy="multiplicative",
            multiplier=2.0,
            max_timeout=600.0,
        )
        assert compute_escalated_timeout(120.0, 1, config, max_attempts=3) == 240.0
        assert compute_escalated_timeout(120.0, 2, config, max_attempts=3) == 480.0
        assert compute_escalated_timeout(120.0, 3, config, max_attempts=3) == 600.0

    def test_linear_interpolation(self) -> None:
        """linspace(120, 480, 4) = [120, 240, 360, 480]."""
        config = TimeoutEscalationConfig(strategy="linear", max_timeout=480.0)
        assert compute_escalated_timeout(120.0, 1, config, max_attempts=3) == 240.0
        assert compute_escalated_timeout(120.0, 2, config, max_attempts=3) == 360.0
        assert compute_escalated_timeout(120.0, 3, config, max_attempts=3) == 480.0

    def test_linear_clamped_above_max_attempts(self) -> None:
        config = TimeoutEscalationConfig(strategy="linear", max_timeout=480.0)
        # n exceeds max_attempts -> still capped at max_timeout
        assert compute_escalated_timeout(120.0, 5, config, max_attempts=3) == 480.0

    def test_linear_zero_max_attempts_falls_back_to_cap(self) -> None:
        config = TimeoutEscalationConfig(strategy="linear", max_timeout=480.0)
        assert compute_escalated_timeout(120.0, 1, config, max_attempts=0) == 480.0


# ---------------------------------------------------------------------------
# RetryExecutor.execute_with_timeout / aexecute_with_timeout
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetryExecutorTimeoutEscalationSync:
    """RetryExecutor.execute_with_timeout escalates only on TIMEOUT retries."""

    def _make_policy(
        self,
        *,
        escalation: TimeoutEscalationConfig | None = None,
        timeout_attempts: int = 3,
    ) -> RetryPolicy:
        return RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
            timeout=CategoryRetryConfig(max_attempts=timeout_attempts, backoff_min=0, backoff_max=0),
            rate_limit=CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
            server_error=CategoryRetryConfig(max_attempts=2, backoff_min=0, backoff_max=0),
            timeout_escalation=escalation,
        )

    def test_first_call_receives_base_timeout(self) -> None:
        executor = RetryExecutor(self._make_policy(), ErrorRegistry())
        observed: list[float | None] = []

        def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            return "ok"

        result = executor.execute_with_timeout(fn, timeout=120.0)
        assert result == "ok"
        assert observed == [120.0]

    def test_no_escalation_uses_constant_timeout(self) -> None:
        """Regression: when escalation is None, timeout stays the same on retries."""
        executor = RetryExecutor(self._make_policy(escalation=None), ErrorRegistry())
        observed: list[float | None] = []
        attempts = {"n": 0}

        def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise TimeoutError("slow")
            return "ok"

        result = executor.execute_with_timeout(fn, timeout=120.0)
        assert result == "ok"
        assert observed == [120.0, 120.0, 120.0]

    def test_multiplicative_escalation_on_timeout_retries(self) -> None:
        executor = RetryExecutor(
            self._make_policy(
                escalation=TimeoutEscalationConfig(strategy="multiplicative", multiplier=2.0),
            ),
            ErrorRegistry(),
        )
        observed: list[float | None] = []
        attempts = {"n": 0}

        def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            attempts["n"] += 1
            if attempts["n"] < 4:
                raise TimeoutError("slow")
            return "ok"

        result = executor.execute_with_timeout(fn, timeout=120.0)
        assert result == "ok"
        assert observed == [120.0, 240.0, 480.0, 960.0]

    def test_additive_escalation_with_cap(self) -> None:
        executor = RetryExecutor(
            self._make_policy(
                escalation=TimeoutEscalationConfig(
                    strategy="additive",
                    increment=60.0,
                    max_timeout=200.0,
                ),
            ),
            ErrorRegistry(),
        )
        observed: list[float | None] = []
        attempts = {"n": 0}

        def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            attempts["n"] += 1
            if attempts["n"] < 4:
                raise TimeoutError("slow")
            return "ok"

        executor.execute_with_timeout(fn, timeout=120.0)
        assert observed == [120.0, 180.0, 200.0, 200.0]

    def test_linear_escalation(self) -> None:
        executor = RetryExecutor(
            self._make_policy(
                escalation=TimeoutEscalationConfig(strategy="linear", max_timeout=480.0),
                timeout_attempts=3,
            ),
            ErrorRegistry(),
        )
        observed: list[float | None] = []
        attempts = {"n": 0}

        def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            attempts["n"] += 1
            if attempts["n"] < 4:
                raise TimeoutError("slow")
            return "ok"

        executor.execute_with_timeout(fn, timeout=120.0)
        assert observed == [120.0, 240.0, 360.0, 480.0]

    def test_non_timeout_retry_does_not_escalate(self) -> None:
        """A connection-error retry must not change the timeout passed to fn."""
        executor = RetryExecutor(
            self._make_policy(
                escalation=TimeoutEscalationConfig(strategy="multiplicative", multiplier=2.0),
            ),
            ErrorRegistry(),
        )
        observed: list[float | None] = []
        attempts = {"n": 0}

        def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionError("net")
            return "ok"

        executor.execute_with_timeout(fn, timeout=120.0)
        assert observed == [120.0, 120.0, 120.0]

    def test_mixed_categories_only_advance_on_timeout(self) -> None:
        """Only TIMEOUT failures advance the escalation index."""
        executor = RetryExecutor(
            self._make_policy(
                escalation=TimeoutEscalationConfig(strategy="multiplicative", multiplier=2.0),
            ),
            ErrorRegistry(),
        )
        observed: list[float | None] = []
        sequence = iter(
            [
                ConnectionError("net"),  # call 1: still 120
                TimeoutError("slow"),  # call 2: still 120 (this raises, escalates next)
                ConnectionError("net"),  # call 3: 240 (escalated by previous timeout)
                TimeoutError("slow"),  # call 4: 240 (this raises, escalates next)
                None,  # call 5: 480
            ]
        )

        def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            exc = next(sequence)
            if exc is not None:
                raise exc
            return "ok"

        result = executor.execute_with_timeout(fn, timeout=120.0)
        assert result == "ok"
        assert observed == [120.0, 120.0, 240.0, 240.0, 480.0]

    def test_permanent_error_not_retried(self) -> None:
        executor = RetryExecutor(
            self._make_policy(
                escalation=TimeoutEscalationConfig(strategy="multiplicative", multiplier=2.0),
            ),
            ErrorRegistry(),
        )
        observed: list[float | None] = []

        def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            executor.execute_with_timeout(fn, timeout=120.0)
        assert observed == [120.0]

    def test_budget_exhaustion_re_raises(self) -> None:
        executor = RetryExecutor(
            self._make_policy(
                escalation=TimeoutEscalationConfig(strategy="multiplicative", multiplier=2.0),
                timeout_attempts=2,
            ),
            ErrorRegistry(),
        )
        observed: list[float | None] = []

        def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            raise TimeoutError("always slow")

        with pytest.raises(TimeoutError):
            executor.execute_with_timeout(fn, timeout=120.0)
        # 1 initial + 2 retries = 3 calls; sequence 120, 240, 480
        assert observed == [120.0, 240.0, 480.0]

    def test_none_base_timeout_passes_through(self) -> None:
        """Calling with timeout=None disables escalation gracefully."""
        executor = RetryExecutor(
            self._make_policy(
                escalation=TimeoutEscalationConfig(strategy="multiplicative", multiplier=2.0),
            ),
            ErrorRegistry(),
        )
        observed: list[float | None] = []
        attempts = {"n": 0}

        def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise TimeoutError("slow")
            return "ok"

        executor.execute_with_timeout(fn, timeout=None)
        assert observed == [None, None]

    def test_passes_args_and_kwargs(self) -> None:
        executor = RetryExecutor(self._make_policy(), ErrorRegistry())

        def fn(a: int, b: int, *, timeout: float | None, c: int = 0) -> int:
            assert timeout == 30.0
            return a + b + c

        result = executor.execute_with_timeout(fn, 1, 2, timeout=30.0, c=3)
        assert result == 6


@pytest.mark.unit
class TestRetryExecutorTimeoutEscalationAsync:
    """RetryExecutor.aexecute_with_timeout escalates only on TIMEOUT retries."""

    def _make_policy(
        self,
        *,
        escalation: TimeoutEscalationConfig | None = None,
        timeout_attempts: int = 3,
    ) -> RetryPolicy:
        return RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
            timeout=CategoryRetryConfig(max_attempts=timeout_attempts, backoff_min=0, backoff_max=0),
            rate_limit=CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
            server_error=CategoryRetryConfig(max_attempts=2, backoff_min=0, backoff_max=0),
            timeout_escalation=escalation,
        )

    async def test_first_call_receives_base_timeout(self) -> None:
        executor = RetryExecutor(self._make_policy(), ErrorRegistry())
        observed: list[float | None] = []

        async def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            return "ok"

        await executor.aexecute_with_timeout(fn, timeout=120.0)
        assert observed == [120.0]

    async def test_multiplicative_escalation_on_timeout_retries(self) -> None:
        executor = RetryExecutor(
            self._make_policy(
                escalation=TimeoutEscalationConfig(strategy="multiplicative", multiplier=2.0),
            ),
            ErrorRegistry(),
        )
        observed: list[float | None] = []
        attempts = {"n": 0}

        async def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            attempts["n"] += 1
            if attempts["n"] < 4:
                raise TimeoutError("slow")
            return "ok"

        await executor.aexecute_with_timeout(fn, timeout=120.0)
        assert observed == [120.0, 240.0, 480.0, 960.0]

    async def test_no_escalation_uses_constant_timeout(self) -> None:
        executor = RetryExecutor(self._make_policy(escalation=None), ErrorRegistry())
        observed: list[float | None] = []
        attempts = {"n": 0}

        async def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise TimeoutError("slow")
            return "ok"

        await executor.aexecute_with_timeout(fn, timeout=120.0)
        assert observed == [120.0, 120.0, 120.0]

    async def test_non_timeout_retry_does_not_escalate(self) -> None:
        executor = RetryExecutor(
            self._make_policy(
                escalation=TimeoutEscalationConfig(strategy="multiplicative", multiplier=2.0),
            ),
            ErrorRegistry(),
        )
        observed: list[float | None] = []
        attempts = {"n": 0}

        async def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionError("net")
            return "ok"

        await executor.aexecute_with_timeout(fn, timeout=120.0)
        assert observed == [120.0, 120.0, 120.0]

    async def test_budget_exhaustion_re_raises(self) -> None:
        executor = RetryExecutor(
            self._make_policy(
                escalation=TimeoutEscalationConfig(strategy="additive", increment=60.0),
                timeout_attempts=2,
            ),
            ErrorRegistry(),
        )
        observed: list[float | None] = []

        async def fn(*, timeout: float | None) -> str:
            observed.append(timeout)
            raise TimeoutError("always slow")

        with pytest.raises(TimeoutError):
            await executor.aexecute_with_timeout(fn, timeout=120.0)
        assert observed == [120.0, 180.0, 240.0]

    async def test_passes_args_and_kwargs(self) -> None:
        executor = RetryExecutor(self._make_policy(), ErrorRegistry())

        async def fn(a: int, b: int, *, timeout: float | None, c: int = 0) -> int:
            assert timeout == 30.0
            return a + b + c

        result = await executor.aexecute_with_timeout(fn, 1, 2, timeout=30.0, c=3)
        assert result == 6
