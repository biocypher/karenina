"""Infrastructure-level resilience for verification parser calls."""

from __future__ import annotations

from typing import TypeVar, cast

from pydantic import BaseModel

from karenina.ports import Message, ParsePortResult, ParserPort
from karenina.utils.errors import ErrorCategory, ErrorRegistry, declared_category
from karenina.utils.retry_policy import RetryExecutor, RetryPolicy

T = TypeVar("T", bound=BaseModel)


class _CauseAwareErrorRegistry(ErrorRegistry):
    """Classify wrapped parser errors by walking exception causes."""

    def __init__(self, base: ErrorRegistry) -> None:
        self._base = base

    def classify(self, exception: BaseException) -> ErrorCategory:
        chain: list[BaseException] = []
        current: BaseException | None = exception
        while current is not None and all(current is not seen for seen in chain):
            chain.append(current)
            current = current.__cause__ or current.__context__
        # An error that declares its category (an adapter's own error type)
        # decides, so a wrapper's message text cannot override it: a
        # permanent 400 mentioning "1500001 tokens" is not a server error.
        for error in chain:
            if declared_category(error) is not None:
                return self._base.classify(error)
        for error in chain:
            category = self._base.classify(error)
            if category is not ErrorCategory.PERMANENT:
                return category
        return self._base.classify(exception)


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
