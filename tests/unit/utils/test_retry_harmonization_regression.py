"""Regression tests for retry/error handling harmonization.

These tests verify the key properties that the harmonization must maintain:
no retry multiplication, separate per-category budgets, permanent errors
never retried, correct error classification, custom patterns, serialization
round-trips, removal of old infrastructure, and StreamingTimeoutError dual
inheritance.
"""

from __future__ import annotations

import pytest

from karenina.exceptions import KareninaError, StreamingTimeoutError
from karenina.schemas.config.models import ModelConfig
from karenina.schemas.verification.config import VerificationConfig
from karenina.utils.errors import ErrorCategory, ErrorRegistry
from karenina.utils.retry_policy import (
    CategoryRetryConfig,
    ErrorPatternConfig,
    RetryExecutor,
    RetryPolicy,
)


def _stub_model() -> ModelConfig:
    """Return a minimal ModelConfig for testing."""
    return ModelConfig(
        id="test-model",
        model_provider="openai",
        model_name="test-model",
        interface="langchain",
    )


def _minimal_config(**overrides: object) -> VerificationConfig:
    """Build a VerificationConfig with the minimum required fields.

    Provides a dummy parsing model so VerificationConfig validation passes.
    All overrides are forwarded to the constructor.
    """
    defaults: dict[str, object] = {
        "parsing_models": [_stub_model()],
        "answering_models": [_stub_model()],
    }
    defaults.update(overrides)
    return VerificationConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. No retry multiplication: RetryExecutor budget prevents unbounded retries
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoRetryMultiplication:
    """RetryExecutor budget caps total calls; no layer stacking."""

    def test_timeout_max_attempts_1_gives_exactly_2_calls(self) -> None:
        """With max_attempts=1 for timeout, total calls are 1 initial + 1 retry = 2."""
        call_count = 0

        def fn() -> None:
            nonlocal call_count
            call_count += 1
            raise TimeoutError("timed out")

        policy = RetryPolicy(
            timeout=CategoryRetryConfig(max_attempts=1, backoff_min=0, backoff_max=0),
            connection=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
            rate_limit=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
            server_error=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
        )
        executor = RetryExecutor(policy, ErrorRegistry())

        with pytest.raises(TimeoutError):
            executor.execute(fn)

        # Exactly 2 calls, not 18 (the old 6x3 multiplication)
        assert call_count == 2

    def test_connection_max_attempts_2_gives_exactly_3_calls(self) -> None:
        """With max_attempts=2 for connection, total calls are 1 initial + 2 retries = 3."""
        call_count = 0

        def fn() -> None:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("refused")

        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=2, backoff_min=0, backoff_max=0),
            timeout=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
            rate_limit=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
            server_error=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
        )
        executor = RetryExecutor(policy, ErrorRegistry())

        with pytest.raises(ConnectionError):
            executor.execute(fn)

        assert call_count == 3


# ---------------------------------------------------------------------------
# 2. Mixed error types use separate budgets
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSeparateBudgets:
    """Connection and timeout errors do not share a retry counter."""

    def test_exhaust_timeout_still_retry_connection(self) -> None:
        """After exhausting the timeout budget, connection errors still get retried."""
        sequence = iter(
            [
                TimeoutError("slow"),  # timeout retry 1 (budget exhausted after this)
                TimeoutError("slow again"),  # timeout budget gone, raises
            ]
        )
        call_count = 0

        def fn_timeout() -> None:
            nonlocal call_count
            call_count += 1
            raise next(sequence)

        policy = RetryPolicy(
            timeout=CategoryRetryConfig(max_attempts=1, backoff_min=0, backoff_max=0),
            connection=CategoryRetryConfig(max_attempts=2, backoff_min=0, backoff_max=0),
            rate_limit=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
            server_error=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
        )
        executor = RetryExecutor(policy, ErrorRegistry())

        with pytest.raises(TimeoutError):
            executor.execute(fn_timeout)

        # Timeout exhausted: 1 initial + 1 retry = 2 calls
        assert call_count == 2

        # Now verify connection budget is still fresh in a new execute() call
        conn_count = 0
        conn_sequence = iter(
            [
                ConnectionError("net1"),
                ConnectionError("net2"),
                None,  # success
            ]
        )

        def fn_connection() -> str:
            nonlocal conn_count
            conn_count += 1
            exc = next(conn_sequence)
            if exc is not None:
                raise exc
            return "ok"

        result = executor.execute(fn_connection)
        assert result == "ok"
        assert conn_count == 3

    def test_interleaved_categories_use_own_budgets(self) -> None:
        """Within one execute() call, each category tracks its own counter."""
        errors = iter(
            [
                ConnectionError("net"),
                TimeoutError("slow"),
                ConnectionError("net again"),
                None,  # success
            ]
        )

        def fn() -> str:
            exc = next(errors)
            if exc is not None:
                raise exc
            return "done"

        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=2, backoff_min=0, backoff_max=0),
            timeout=CategoryRetryConfig(max_attempts=1, backoff_min=0, backoff_max=0),
            rate_limit=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
            server_error=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
        )
        executor = RetryExecutor(policy, ErrorRegistry())
        result = executor.execute(fn)
        assert result == "done"


# ---------------------------------------------------------------------------
# 3. Permanent errors never retried
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPermanentErrorsNeverRetried:
    """ValueError, TypeError, and similar produce exactly 1 call."""

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("bad input"),
            TypeError("wrong type"),
            KeyError("missing"),
            AttributeError("no such attr"),
            RuntimeError("generic runtime error"),
        ],
        ids=["ValueError", "TypeError", "KeyError", "AttributeError", "RuntimeError"],
    )
    def test_permanent_error_exactly_one_call(self, exc: Exception) -> None:
        call_count = 0

        def fn() -> None:
            nonlocal call_count
            call_count += 1
            raise exc

        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
            timeout=CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
            rate_limit=CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
            server_error=CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
        )
        executor = RetryExecutor(policy, ErrorRegistry())

        with pytest.raises(type(exc)):
            executor.execute(fn)

        assert call_count == 1


# ---------------------------------------------------------------------------
# 4. ErrorRegistry classifies correctly (spot-check built-in classifications)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestErrorRegistryClassification:
    """Spot-check that built-in classifications are correct."""

    def test_connection_error(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(ConnectionError("reset")) == ErrorCategory.CONNECTION

    def test_timeout_error(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(TimeoutError("timed out")) == ErrorCategory.TIMEOUT

    def test_streaming_timeout_with_partial_content(self) -> None:
        """Partial content = genuine slow response = TIMEOUT."""
        registry = ErrorRegistry()
        err = StreamingTimeoutError("streaming timed out", partial_content="The answer")
        assert registry.classify(err) == ErrorCategory.TIMEOUT

    def test_streaming_timeout_zero_content_is_timeout(self) -> None:
        """Zero content or not, a timeout is a timeout."""
        registry = ErrorRegistry()
        err = StreamingTimeoutError("streaming timed out", partial_content="")
        assert registry.classify(err) == ErrorCategory.TIMEOUT

    def test_rate_limit_error_by_type_name(self) -> None:
        """A class named RateLimitError is classified as RATE_LIMIT."""
        registry = ErrorRegistry()
        exc_class = type("RateLimitError", (Exception,), {})
        assert registry.classify(exc_class("fail")) == ErrorCategory.RATE_LIMIT

    def test_rate_limit_by_message(self) -> None:
        """An exception with 'rate limit' in its message is classified as RATE_LIMIT."""
        registry = ErrorRegistry()
        assert registry.classify(Exception("rate limit exceeded")) == ErrorCategory.RATE_LIMIT

    def test_value_error_is_permanent(self) -> None:
        registry = ErrorRegistry()
        assert registry.classify(ValueError("invalid")) == ErrorCategory.PERMANENT


# ---------------------------------------------------------------------------
# 5. Custom error patterns work end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCustomErrorPatterns:
    """Register a pattern, verify classification propagates to RetryExecutor."""

    def test_custom_message_pattern_classification(self) -> None:
        """Register 'vllm_busy' as RATE_LIMIT; verify classification."""
        registry = ErrorRegistry()
        registry.register_pattern("vllm_busy", ErrorCategory.RATE_LIMIT)
        exc = Exception("server returned vllm_busy")
        assert registry.classify(exc) == ErrorCategory.RATE_LIMIT

    def test_custom_pattern_triggers_retry(self) -> None:
        """An exception matching a custom pattern is retried per its category budget."""
        registry = ErrorRegistry()
        registry.register_pattern("vllm_busy", ErrorCategory.RATE_LIMIT)

        call_count = 0

        def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("server returned vllm_busy")  # noqa: TRY002
            return "ok"

        policy = RetryPolicy(
            rate_limit=CategoryRetryConfig(max_attempts=3, backoff_min=0, backoff_max=0),
            connection=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
            timeout=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
            server_error=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
        )
        executor = RetryExecutor(policy, registry)
        result = executor.execute(fn)
        assert result == "ok"
        assert call_count == 3


# ---------------------------------------------------------------------------
# 6. RetryPolicy serialization round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRetryPolicySerialization:
    """Dump and reload RetryPolicy, verify equality."""

    def test_default_roundtrip(self) -> None:
        policy = RetryPolicy()
        data = policy.model_dump()
        restored = RetryPolicy.model_validate(data)
        assert restored == policy

    def test_custom_roundtrip(self) -> None:
        policy = RetryPolicy(
            connection=CategoryRetryConfig(max_attempts=5, backoff_min=0.5, backoff_max=20.0),
            timeout=CategoryRetryConfig(max_attempts=0),
            rate_limit=CategoryRetryConfig(max_attempts=10, backoff_min=1.0, backoff_max=60.0),
            server_error=CategoryRetryConfig(max_attempts=1, backoff_min=3.0),
        )
        data = policy.model_dump()
        restored = RetryPolicy.model_validate(data)
        assert restored == policy

    def test_json_roundtrip(self) -> None:
        """Verify JSON string serialization and deserialization."""
        policy = RetryPolicy(
            timeout=CategoryRetryConfig(max_attempts=2, backoff_min=0.1),
        )
        json_str = policy.model_dump_json()
        restored = RetryPolicy.model_validate_json(json_str)
        assert restored == policy


# ---------------------------------------------------------------------------
# 7. Old retry infrastructure removed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOldRetryInfrastructureRemoved:
    """Verify that legacy retry symbols cannot be imported."""

    def test_transient_retry_not_importable(self) -> None:
        """TRANSIENT_RETRY was removed; importing it raises ImportError."""
        with pytest.raises(ImportError):
            from karenina.utils.retry import TRANSIENT_RETRY  # noqa: F401

    def test_create_transient_retry_not_importable(self) -> None:
        """create_transient_retry was removed; importing it raises ImportError."""
        with pytest.raises(ImportError):
            from karenina.utils.retry import create_transient_retry  # noqa: F401

    def test_retry_module_does_not_exist(self) -> None:
        """The old retry.py module itself should not exist."""
        with pytest.raises(ImportError):
            import karenina.utils.retry  # noqa: F401


# ---------------------------------------------------------------------------
# 8. VerificationConfig uses RetryPolicy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVerificationConfigRetryFields:
    """VerificationConfig exposes new retry fields, not old ones."""

    def test_retry_policy_exists_and_is_retry_policy(self) -> None:
        config = _minimal_config()
        assert isinstance(config.retry_policy, RetryPolicy)

    def test_custom_error_patterns_exists_and_is_list(self) -> None:
        config = _minimal_config()
        assert isinstance(config.custom_error_patterns, list)

    def test_custom_error_patterns_default_empty(self) -> None:
        config = _minimal_config()
        assert config.custom_error_patterns == []

    def test_old_max_transient_retries_removed(self) -> None:
        config = _minimal_config()
        assert not hasattr(config, "max_transient_retries")

    def test_old_max_scenario_turn_retries_removed(self) -> None:
        config = _minimal_config()
        assert not hasattr(config, "max_scenario_turn_retries")

    def test_retry_policy_custom_values_accepted(self) -> None:
        policy = RetryPolicy(
            timeout=CategoryRetryConfig(max_attempts=5),
        )
        config = _minimal_config(retry_policy=policy)
        assert config.retry_policy.timeout.max_attempts == 5

    def test_custom_error_patterns_accepted(self) -> None:
        patterns = [
            ErrorPatternConfig(pattern="vllm_busy", category="rate_limit"),
        ]
        config = _minimal_config(custom_error_patterns=patterns)
        assert len(config.custom_error_patterns) == 1
        assert config.custom_error_patterns[0].pattern == "vllm_busy"


# ---------------------------------------------------------------------------
# 9. StreamingTimeoutError dual inheritance
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamingTimeoutErrorDualInheritance:
    """StreamingTimeoutError is caught by both TimeoutError and KareninaError."""

    def test_caught_by_except_timeout_error(self) -> None:
        caught = False
        try:
            raise StreamingTimeoutError("streaming timed out")
        except TimeoutError:
            caught = True
        assert caught

    def test_caught_by_except_karenina_error(self) -> None:
        caught = False
        try:
            raise StreamingTimeoutError("streaming timed out")
        except KareninaError:
            caught = True
        assert caught

    def test_isinstance_checks(self) -> None:
        err = StreamingTimeoutError("timed out")
        assert isinstance(err, TimeoutError)
        assert isinstance(err, KareninaError)
        assert isinstance(err, Exception)

    def test_issubclass_checks(self) -> None:
        assert issubclass(StreamingTimeoutError, TimeoutError)
        assert issubclass(StreamingTimeoutError, KareninaError)
