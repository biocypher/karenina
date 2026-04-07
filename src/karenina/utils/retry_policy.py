"""Category-aware retry policy and executor for LLM operations.

This module provides:
- CategoryRetryConfig: per-category retry parameters (attempts, backoff).
- RetryPolicy: groups CategoryRetryConfig for each ErrorCategory.
- ErrorPatternConfig: declarative pattern registration for serialization.
- RetryExecutor: sync/async executor that retries with per-category budgets.
- track_retries: contextvar-based context manager that lets callers record
  retries (with the budgets that were available) over the course of a
  pipeline execution.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ErrorCategory, ErrorRegistry

logger = logging.getLogger(__name__)

__all__ = [
    "CategoryRetryConfig",
    "ErrorPatternConfig",
    "RetryExecutor",
    "RetryPolicy",
    "TimeoutEscalationConfig",
    "compute_escalated_timeout",
    "track_retries",
]

T = TypeVar("T")

# Context-local active retry tracker. Set by track_retries() and read by
# RetryExecutor every time a retry decision is made. Using contextvars makes
# the tracker visible across both sync and async paths within the same logical
# execution context, while remaining isolated between concurrent worker threads
# and asyncio tasks.
#
# Tracker shape: {category_name: {"used": int, "budget": int}}
# where category_name is one of "connection", "timeout", "rate_limit",
# "server_error", and budget reflects the max_attempts in effect for that
# category at the moment track_retries() opened. Pre-populated with budgets so
# consumers can see "we had N attempts available, used K of them" even when no
# retries fired for a given category.
RetryTracker = dict[str, dict[str, int]]
_active_retry_tracker: contextvars.ContextVar[RetryTracker | None] = contextvars.ContextVar(
    "_active_retry_tracker",
    default=None,
)


def _build_initial_tracker(policy: RetryPolicy | None) -> RetryTracker:
    """Pre-populate a tracker with budgets from ``policy``.

    A None policy is treated as the default RetryPolicy. Each retryable
    category gets an entry with ``used=0`` and ``budget=max_attempts``.
    """
    effective = policy or RetryPolicy()
    return {
        ErrorCategory.CONNECTION.value: {"used": 0, "budget": effective.connection.max_attempts},
        ErrorCategory.TIMEOUT.value: {"used": 0, "budget": effective.timeout.max_attempts},
        ErrorCategory.RATE_LIMIT.value: {"used": 0, "budget": effective.rate_limit.max_attempts},
        ErrorCategory.SERVER_ERROR.value: {"used": 0, "budget": effective.server_error.max_attempts},
    }


@contextmanager
def track_retries(policy: RetryPolicy | None = None) -> Iterator[RetryTracker]:
    """Bind a fresh retry tracker to the current execution context.

    The tracker is pre-populated with the per-category budgets from
    ``policy`` (defaulting to ``RetryPolicy()`` when None). Within the
    ``with`` block, every retry decision made by ``RetryExecutor``
    increments ``used`` for the matching category. The tracker is yielded so
    the caller can read counts and budgets after exiting the block.

    Example:
        >>> with track_retries(RetryPolicy()) as tracker:
        ...     adapter.stream_invoke(messages)
        >>> tracker["timeout"]
        {'used': 2, 'budget': 3}
    """
    tracker = _build_initial_tracker(policy)
    token = _active_retry_tracker.set(tracker)
    try:
        yield tracker
    finally:
        _active_retry_tracker.reset(token)


def _record_retry(category: ErrorCategory) -> None:
    """Increment the active retry tracker for ``category`` if one is bound."""
    tracker = _active_retry_tracker.get()
    if tracker is None:
        return
    entry = tracker.get(category.value)
    if entry is not None:
        entry["used"] += 1


class CategoryRetryConfig(BaseModel):
    """Retry parameters for a single error category.

    Attributes:
        max_attempts: Number of retries (not total calls). 0 means no retry.
        backoff_min: Minimum backoff delay in seconds.
        backoff_max: Maximum backoff delay in seconds.
        backoff_multiplier: Multiplier applied to delay after each retry.
    """

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=0, ge=0)
    backoff_min: float = Field(default=1.0, ge=0)
    backoff_max: float = Field(default=10.0, ge=0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)


class TimeoutEscalationConfig(BaseModel):
    """Progressive timeout strategy applied to TIMEOUT-category retries.

    On each retry triggered by a TIMEOUT error, the per-attempt request
    timeout is increased according to the chosen strategy. Other retry
    categories (connection, rate_limit, server_error) keep using the
    base request_timeout unchanged.

    Strategies:
        additive: timeout(n) = min(base + increment * n, max_timeout)
        multiplicative: timeout(n) = min(base * multiplier ** n, max_timeout)
        linear: timeout(n) = base + (max_timeout - base) * n / max_attempts
            where max_attempts is the configured timeout retry budget.
            n=0 returns the base; n=max_attempts returns max_timeout.

    Attributes:
        strategy: Which growth function to use.
        increment: Seconds added per retry. Used by additive only.
        multiplier: Factor applied per retry. Used by multiplicative only.
        max_timeout: Optional cap (additive/multiplicative) or required
            endpoint (linear).
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["additive", "multiplicative", "linear"]
    increment: float = Field(default=0.0, ge=0.0)
    multiplier: float = Field(default=1.0, ge=1.0)
    max_timeout: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _validate_strategy_params(self) -> TimeoutEscalationConfig:
        if self.strategy == "additive" and self.increment <= 0:
            raise ValueError("additive strategy requires increment > 0")
        if self.strategy == "multiplicative" and self.multiplier <= 1.0:
            raise ValueError("multiplicative strategy requires multiplier > 1.0")
        if self.strategy == "linear" and self.max_timeout is None:
            raise ValueError("linear strategy requires max_timeout")
        return self


class RetryPolicy(BaseModel):
    """Grouped retry configuration for all error categories.

    Each field maps to an ErrorCategory and defines the retry budget
    and backoff parameters for that category.

    Attributes:
        connection: Config for connection errors (DNS, network failures).
        timeout: Config for timeout errors.
        rate_limit: Config for rate limiting (429, overloaded).
        server_error: Config for server errors (5xx).
    """

    model_config = ConfigDict(extra="forbid")

    connection: CategoryRetryConfig = Field(
        default_factory=lambda: CategoryRetryConfig(
            max_attempts=3,
            backoff_min=1.0,
            backoff_max=10.0,
        ),
    )
    timeout: CategoryRetryConfig = Field(
        default_factory=lambda: CategoryRetryConfig(
            max_attempts=3,
            backoff_min=5.0,
            backoff_max=30.0,
        ),
    )
    rate_limit: CategoryRetryConfig = Field(
        default_factory=lambda: CategoryRetryConfig(
            max_attempts=5,
            backoff_min=5.0,
            backoff_max=30.0,
        ),
    )
    server_error: CategoryRetryConfig = Field(
        default_factory=lambda: CategoryRetryConfig(
            max_attempts=2,
            backoff_min=2.0,
            backoff_max=15.0,
        ),
    )
    timeout_escalation: TimeoutEscalationConfig | None = Field(
        default=None,
        description=(
            "Progressive timeout strategy applied on retries triggered by "
            "TIMEOUT errors. None means use the same request_timeout on "
            "every retry."
        ),
    )

    def derive_sdk_max_retries(self) -> int:
        """Return the maximum retry count across all categories.

        This value can be passed to SDK clients that accept a single
        max_retries parameter, ensuring no category is starved.

        Returns:
            The highest max_attempts among all categories.
        """
        return max(
            self.connection.max_attempts,
            self.timeout.max_attempts,
            self.rate_limit.max_attempts,
            self.server_error.max_attempts,
        )


class ErrorPatternConfig(BaseModel):
    """Declarative error pattern for serialization in configs.

    Attributes:
        pattern: The string to match (type name or message substring).
        category: Which ErrorCategory to assign.
        match_type: How to match: "type_name" or "message_substring".
    """

    model_config = ConfigDict(extra="forbid")

    pattern: str
    category: Literal["connection", "timeout", "rate_limit", "server_error", "permanent"]
    match_type: Literal["type_name", "message_substring"] = "message_substring"


def _get_category_config(policy: RetryPolicy, category: ErrorCategory) -> CategoryRetryConfig:
    """Look up the CategoryRetryConfig for a given ErrorCategory.

    Args:
        policy: The retry policy.
        category: The error category.

    Returns:
        The matching CategoryRetryConfig, or a zero-retry config for PERMANENT.
    """
    mapping: dict[ErrorCategory, CategoryRetryConfig] = {
        ErrorCategory.CONNECTION: policy.connection,
        ErrorCategory.TIMEOUT: policy.timeout,
        ErrorCategory.RATE_LIMIT: policy.rate_limit,
        ErrorCategory.SERVER_ERROR: policy.server_error,
    }
    return mapping.get(category, CategoryRetryConfig(max_attempts=0))


def _compute_backoff(config: CategoryRetryConfig, attempt: int) -> float:
    """Compute backoff delay with jitter for a retry attempt.

    Uses exponential backoff: delay = min(backoff_min * multiplier^attempt, backoff_max),
    then applies uniform jitter between 0 and the computed delay.

    Args:
        config: Retry config for this category.
        attempt: The retry attempt number (0-indexed).

    Returns:
        Delay in seconds.
    """
    raw = config.backoff_min * (config.backoff_multiplier**attempt)
    clamped = min(raw, config.backoff_max)
    if clamped <= 0:
        return 0.0
    return random.uniform(0, clamped)  # noqa: S311


def compute_escalated_timeout(
    base_timeout: float | None,
    timeout_attempt: int,
    config: TimeoutEscalationConfig | None,
    max_attempts: int,
) -> float | None:
    """Compute the request timeout for the next attempt.

    Args:
        base_timeout: The original timeout configured for the call.
            If None, escalation is a no-op and None is returned.
        timeout_attempt: How many TIMEOUT retries have been used so far,
            counting the upcoming attempt. 0 means the original call
            (returns base unchanged). k in 1..max_attempts means the
            k-th retry.
        config: The escalation configuration. None disables escalation
            and returns base_timeout unchanged.
        max_attempts: The TIMEOUT category retry budget. Used by the
            "linear" strategy to interpolate between base and max.

    Returns:
        The escalated timeout in seconds, or base_timeout if escalation
        is disabled.
    """
    if base_timeout is None or config is None:
        return base_timeout

    n = timeout_attempt
    if n <= 0:
        return base_timeout

    if config.strategy == "additive":
        scaled = base_timeout + config.increment * n
    elif config.strategy == "multiplicative":
        scaled = base_timeout * (config.multiplier**n)
    else:  # linear
        # max_timeout is guaranteed non-None by the validator.
        assert config.max_timeout is not None
        if max_attempts <= 0:
            # Cannot interpolate without a denominator. Fall back to the cap.
            return config.max_timeout
        fraction = min(n / max_attempts, 1.0)
        scaled = base_timeout + (config.max_timeout - base_timeout) * fraction

    if config.max_timeout is not None:
        scaled = min(scaled, config.max_timeout)
    return scaled


class RetryExecutor:
    """Category-aware retry executor with per-category budgets.

    Each call to execute() or aexecute() creates fresh budget counters.
    Errors are classified via the ErrorRegistry, and each category is
    retried up to its configured max_attempts. Permanent errors are
    never retried.

    Args:
        policy: The RetryPolicy defining per-category budgets and backoff.
        registry: The ErrorRegistry used to classify exceptions.

    Example:
        >>> executor = RetryExecutor(RetryPolicy(), ErrorRegistry())
        >>> result = executor.execute(my_function, arg1, arg2)
    """

    def __init__(self, policy: RetryPolicy, registry: ErrorRegistry) -> None:
        self._policy = policy
        self._registry = registry

    def execute(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute fn with category-aware retry (sync).

        Args:
            fn: Callable to invoke.
            *args: Positional arguments forwarded to fn.
            **kwargs: Keyword arguments forwarded to fn.

        Returns:
            The return value of fn on success.

        Raises:
            The last exception if all retries are exhausted, or immediately
            for permanent errors.
        """
        budgets: dict[ErrorCategory, int] = {}
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                category = self._registry.classify(exc)
                config = _get_category_config(self._policy, category)
                used = budgets.get(category, 0)

                if not category.is_retryable() or used >= config.max_attempts:
                    raise

                budgets[category] = used + 1
                _record_retry(category)
                delay = _compute_backoff(config, used)
                logger.warning(
                    "Retrying after %s error (attempt %d/%d): %s",
                    category.value,
                    used + 1,
                    config.max_attempts,
                    exc,
                )
                if delay > 0:
                    time.sleep(delay)

    async def aexecute(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute fn with category-aware retry (async).

        Args:
            fn: Async callable to invoke.
            *args: Positional arguments forwarded to fn.
            **kwargs: Keyword arguments forwarded to fn.

        Returns:
            The return value of fn on success.

        Raises:
            The last exception if all retries are exhausted, or immediately
            for permanent errors.
        """
        budgets: dict[ErrorCategory, int] = {}
        while True:
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                category = self._registry.classify(exc)
                config = _get_category_config(self._policy, category)
                used = budgets.get(category, 0)

                if not category.is_retryable() or used >= config.max_attempts:
                    raise

                budgets[category] = used + 1
                _record_retry(category)
                delay = _compute_backoff(config, used)
                logger.warning(
                    "Retrying after %s error (attempt %d/%d): %s",
                    category.value,
                    used + 1,
                    config.max_attempts,
                    exc,
                )
                if delay > 0:
                    await asyncio.sleep(delay)

    def execute_with_timeout(
        self,
        fn: Any,
        *args: Any,
        timeout: float | None,
        **kwargs: Any,
    ) -> Any:
        """Execute fn with category-aware retry and timeout escalation (sync).

        Like ``execute``, but the ``timeout`` keyword argument forwarded to
        ``fn`` is escalated on each TIMEOUT-category retry according to
        ``self._policy.timeout_escalation``. Other retry categories reuse
        the original ``timeout`` value.

        Args:
            fn: Sync callable to invoke. Must accept ``timeout`` as a keyword
                argument.
            *args: Positional arguments forwarded to fn.
            timeout: Initial wall-clock timeout passed on the first attempt
                and used as the base for escalation. None disables timeout.
            **kwargs: Additional keyword arguments forwarded to fn.

        Returns:
            The return value of fn on success.

        Raises:
            The last exception if all retries are exhausted, or immediately
            for permanent errors.
        """
        budgets: dict[ErrorCategory, int] = {}
        escalation = self._policy.timeout_escalation
        timeout_max_attempts = self._policy.timeout.max_attempts
        current_timeout = timeout
        while True:
            try:
                return fn(*args, timeout=current_timeout, **kwargs)
            except Exception as exc:
                category = self._registry.classify(exc)
                config = _get_category_config(self._policy, category)
                used = budgets.get(category, 0)

                if not category.is_retryable() or used >= config.max_attempts:
                    raise

                budgets[category] = used + 1
                _record_retry(category)
                delay = _compute_backoff(config, used)
                if category is ErrorCategory.TIMEOUT:
                    current_timeout = compute_escalated_timeout(
                        base_timeout=timeout,
                        timeout_attempt=used + 1,
                        config=escalation,
                        max_attempts=timeout_max_attempts,
                    )
                logger.warning(
                    "Retrying after %s error (attempt %d/%d, next timeout=%s): %s",
                    category.value,
                    used + 1,
                    config.max_attempts,
                    current_timeout,
                    exc,
                )
                if delay > 0:
                    time.sleep(delay)

    async def aexecute_with_timeout(
        self,
        fn: Any,
        *args: Any,
        timeout: float | None,
        **kwargs: Any,
    ) -> Any:
        """Execute fn with category-aware retry and timeout escalation (async).

        Like ``aexecute``, but the ``timeout`` keyword argument forwarded to
        ``fn`` is escalated on each TIMEOUT-category retry according to
        ``self._policy.timeout_escalation``. Other retry categories reuse
        the original ``timeout`` value.

        Args:
            fn: Async callable to invoke. Must accept ``timeout`` as a keyword
                argument.
            *args: Positional arguments forwarded to fn.
            timeout: Initial wall-clock timeout passed on the first attempt
                and used as the base for escalation. None disables timeout.
            **kwargs: Additional keyword arguments forwarded to fn.

        Returns:
            The return value of fn on success.

        Raises:
            The last exception if all retries are exhausted, or immediately
            for permanent errors.
        """
        budgets: dict[ErrorCategory, int] = {}
        escalation = self._policy.timeout_escalation
        timeout_max_attempts = self._policy.timeout.max_attempts
        current_timeout = timeout
        while True:
            try:
                return await fn(*args, timeout=current_timeout, **kwargs)
            except Exception as exc:
                category = self._registry.classify(exc)
                config = _get_category_config(self._policy, category)
                used = budgets.get(category, 0)

                if not category.is_retryable() or used >= config.max_attempts:
                    raise

                budgets[category] = used + 1
                _record_retry(category)
                delay = _compute_backoff(config, used)
                if category is ErrorCategory.TIMEOUT:
                    current_timeout = compute_escalated_timeout(
                        base_timeout=timeout,
                        timeout_attempt=used + 1,
                        config=escalation,
                        max_attempts=timeout_max_attempts,
                    )
                logger.warning(
                    "Retrying after %s error (attempt %d/%d, next timeout=%s): %s",
                    category.value,
                    used + 1,
                    config.max_attempts,
                    current_timeout,
                    exc,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
