"""Infrastructure-level resilience for verification parser calls."""

from __future__ import annotations

from typing import TypeVar, cast

from pydantic import BaseModel

from karenina.ports import Message, ParsePortResult, ParserPort
from karenina.utils.errors import ErrorCategory, ErrorRegistry
from karenina.utils.retry_policy import RetryExecutor, RetryPolicy

T = TypeVar("T", bound=BaseModel)


class _CauseAwareErrorRegistry(ErrorRegistry):
    """Classify wrapped parser errors by walking exception causes."""

    def __init__(self, base: ErrorRegistry) -> None:
        self._base = base

    def classify(self, exception: BaseException) -> ErrorCategory:
        fallback = self._base.classify(exception)
        current: BaseException | None = exception
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            category = self._base.classify(current)
            if category is not ErrorCategory.PERMANENT:
                return category
            current = current.__cause__ or current.__context__
        return fallback


def classify_parser_exception(exception: BaseException, error_registry: ErrorRegistry | None = None) -> ErrorCategory:
    """Classify parser errors, including retryable provider errors wrapped in ParseError."""
    registry = _CauseAwareErrorRegistry(error_registry or ErrorRegistry())
    return registry.classify(exception)


def parse_to_pydantic_resilient(
    parser: ParserPort,
    messages: list[Message],
    schema: type[T],
    *,
    retry_policy: RetryPolicy | None,
    error_registry: ErrorRegistry | None = None,
) -> ParsePortResult[T]:
    """Run a ParserPort call through the verification retry policy.

    This boundary keeps retry policy application independent of adapter
    implementation details. It also lets wrapped infrastructure errors, such
    as ``ParseError(... from ConnectionError(...))``, consume the same retry
    budgets as direct provider exceptions.
    """
    executor = RetryExecutor(
        retry_policy or RetryPolicy(),
        _CauseAwareErrorRegistry(error_registry or ErrorRegistry()),
    )
    return cast(ParsePortResult[T], executor.execute(parser.parse_to_pydantic, messages, schema))


__all__ = [
    "classify_parser_exception",
    "parse_to_pydantic_resilient",
]
