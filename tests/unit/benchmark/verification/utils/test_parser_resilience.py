"""Tests for verification-level parser call resilience."""

import pytest
from pydantic import BaseModel

from karenina.ports import Message, ParseError, ParsePortResult, UsageMetadata
from karenina.utils.errors import ErrorRegistry
from karenina.utils.retry_policy import CategoryRetryConfig, RetryPolicy, track_retries


class ParsedFlag(BaseModel):
    flag: bool


class FlakyParser:
    capabilities = None

    def __init__(self, failures: list[BaseException]) -> None:
        self.failures = list(failures)
        self.calls = 0

    def parse_to_pydantic(self, _messages, schema):  # noqa: ANN001
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return ParsePortResult(parsed=schema(flag=True), usage=UsageMetadata(input_tokens=3, output_tokens=1))


def _zero_delay_policy() -> RetryPolicy:
    return RetryPolicy(
        connection=CategoryRetryConfig(max_attempts=2, backoff_min=0, backoff_max=0),
        timeout=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
        rate_limit=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
        server_error=CategoryRetryConfig(max_attempts=0, backoff_min=0, backoff_max=0),
    )


@pytest.mark.unit
def test_parser_resilience_retries_transient_connection_failures():
    from karenina.benchmark.verification.utils.parser_resilience import (
        parse_to_pydantic_resilient,
    )

    parser = FlakyParser([ConnectionError("socket reset")])
    policy = _zero_delay_policy()

    with track_retries(policy) as retry_counts:
        result = parse_to_pydantic_resilient(
            parser,
            [Message.user("extract")],
            ParsedFlag,
            retry_policy=policy,
            error_registry=ErrorRegistry(),
        )

    assert result.parsed.flag is True
    assert parser.calls == 2
    assert retry_counts["connection"]["used"] == 1


@pytest.mark.unit
def test_parser_resilience_classifies_wrapped_parse_error_causes():
    from karenina.benchmark.verification.utils.parser_resilience import (
        parse_to_pydantic_resilient,
    )

    wrapped = ParseError("provider call failed")
    wrapped.__cause__ = ConnectionError("socket reset")
    parser = FlakyParser([wrapped])
    policy = _zero_delay_policy()

    with track_retries(policy) as retry_counts:
        result = parse_to_pydantic_resilient(
            parser,
            [Message.user("extract")],
            ParsedFlag,
            retry_policy=policy,
            error_registry=ErrorRegistry(),
        )

    assert result.parsed.flag is True
    assert parser.calls == 2
    assert retry_counts["connection"]["used"] == 1


@pytest.mark.unit
def test_parser_resilience_does_not_retry_permanent_parse_errors():
    from karenina.benchmark.verification.utils.parser_resilience import (
        parse_to_pydantic_resilient,
    )

    parser = FlakyParser([ParseError("schema validation failed")])
    policy = _zero_delay_policy()

    with pytest.raises(ParseError, match="schema validation failed"):
        parse_to_pydantic_resilient(
            parser,
            [Message.user("extract")],
            ParsedFlag,
            retry_policy=policy,
            error_registry=ErrorRegistry(),
        )

    assert parser.calls == 1
