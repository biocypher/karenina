"""Tests for should_mark_usage_unavailable.

An adapter reporting (0, 0, 0) tokens is indistinguishable from a truly-zero
response without a flag. This helper lets the pipeline mark usage as
unavailable whenever the adapter returned nothing usable, so downstream
analyses can distinguish silent-zero from legitimate-zero.
"""

from dataclasses import dataclass

import pytest

from karenina.benchmark.verification.utils.usage_helpers import (
    should_mark_usage_unavailable,
)
from karenina.ports import UsageMetadata


@dataclass
class _StubLLMResponse:
    content: str = "hi"
    usage: UsageMetadata | None = None
    usage_unavailable: bool = False
    is_partial: bool = False


@pytest.mark.unit
class TestShouldMarkUsageUnavailable:
    def test_all_zero_usage_flags_unavailable(self) -> None:
        zero = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)
        assert should_mark_usage_unavailable(_StubLLMResponse(usage=zero)) is True

    def test_nonzero_usage_does_not_flag(self) -> None:
        real = UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)
        assert should_mark_usage_unavailable(_StubLLMResponse(usage=real)) is False

    def test_missing_usage_flags_unavailable(self) -> None:
        assert should_mark_usage_unavailable(_StubLLMResponse(usage=None)) is True

    def test_explicit_unavailable_is_respected(self) -> None:
        resp = _StubLLMResponse(usage=None, usage_unavailable=True)
        assert should_mark_usage_unavailable(resp) is True

    def test_accepts_agentresult_shape(self) -> None:
        """AgentResult has ``usage`` but no ``usage_unavailable`` attribute.
        The helper must still classify zero/missing usage correctly."""

        @dataclass
        class _StubAgentResult:
            usage: UsageMetadata | None = None

        zero = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)
        assert should_mark_usage_unavailable(_StubAgentResult(usage=zero)) is True
        assert should_mark_usage_unavailable(_StubAgentResult(usage=None)) is True
        real = UsageMetadata(input_tokens=5, output_tokens=5, total_tokens=10)
        assert should_mark_usage_unavailable(_StubAgentResult(usage=real)) is False
