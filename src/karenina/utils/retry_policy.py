"""Category-aware retry policy and executor for LLM operations.

This module provides:
- CategoryRetryConfig: per-category retry parameters (attempts, backoff).
- RetryPolicy: groups CategoryRetryConfig for each ErrorCategory.
- ErrorPatternConfig: declarative pattern registration for serialization.
- RetryExecutor: sync/async executor that retries with per-category budgets.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .errors import ErrorCategory, ErrorRegistry

logger = logging.getLogger(__name__)

__all__ = [
    "CategoryRetryConfig",
    "ErrorPatternConfig",
    "RetryExecutor",
    "RetryPolicy",
]

T = TypeVar("T")


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
