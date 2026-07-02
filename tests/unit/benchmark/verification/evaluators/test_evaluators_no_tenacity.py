"""Behavioral tests proving abstention/sufficiency evaluators do not retry.

After the retry-error-harmonization refactor the adapter returned by
``get_llm()`` owns all retry behavior (via ``RetryExecutor``). The evaluator
functions must therefore invoke the structured LLM exactly once per call
and surface any failure through their ``(performed=False)`` return tuple
instead of looping internally.

These tests exercise that contract behaviorally: a failing LLM mock is
installed and we assert ``invoke`` is reached exactly once and that the
evaluator returns its documented failure tuple rather than raising. This
catches any future regression that re-introduces an in-evaluator retry
loop regardless of how that loop is named (tenacity, hand-rolled, etc.).
"""

from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.evaluators.trace import abstention, sufficiency
from karenina.ports import LLMResponse
from karenina.ports.usage import UsageMetadata
from karenina.schemas.config import ModelConfig


def _parsing_model() -> ModelConfig:
    return ModelConfig(
        id="parser",
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        temperature=0.0,
    )


def _llm_with_structured(structured_llm: MagicMock) -> MagicMock:
    """Wrap a structured-output mock in its parent LLM mock."""
    llm = MagicMock()
    llm.with_structured_output.return_value = structured_llm
    return llm


def _failing_structured_llm() -> MagicMock:
    """A structured-output LLM mock whose ``invoke`` always raises.

    Used to prove the evaluator does not loop: if any retry logic existed
    inside the evaluator, ``invoke.call_count`` would exceed 1.
    """
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = RuntimeError("adapter blew up")
    return structured_llm


@pytest.mark.unit
class TestAbstentionDelegatesRetryToAdapter:
    """detect_abstention must call its LLM exactly once and swallow failures."""

    def test_invoke_called_once_and_failure_is_reported_not_raised(self) -> None:
        structured_llm = _failing_structured_llm()

        with patch.object(abstention, "get_llm", return_value=_llm_with_structured(structured_llm)):
            detected, performed, reasoning, _meta = abstention.detect_abstention(
                raw_llm_response="I cannot answer.",
                parsing_model=_parsing_model(),
                question_text="What is X?",
            )

        # The evaluator must surface the failure, not raise.
        assert performed is False
        assert detected is False
        assert reasoning is None
        # The contract under test: exactly one LLM call, no internal retry.
        assert structured_llm.invoke.call_count == 1


@pytest.mark.unit
class TestSufficiencyDelegatesRetryToAdapter:
    """detect_sufficiency must call its LLM exactly once and swallow failures."""

    def test_invoke_called_once_and_failure_is_reported_not_raised(self) -> None:
        structured_llm = _failing_structured_llm()
        template_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

        with patch.object(sufficiency, "get_llm", return_value=_llm_with_structured(structured_llm)):
            sufficient, performed, reasoning, _meta = sufficiency.detect_sufficiency(
                raw_llm_response="some response",
                parsing_model=_parsing_model(),
                question_text="What is X?",
                template_schema=template_schema,
            )

        # Sufficiency defaults to "treat as sufficient" on evaluator failure.
        assert performed is False
        # The contract under test: exactly one LLM call, no internal retry.
        assert structured_llm.invoke.call_count == 1

    def test_successful_invoke_returns_structured_result(self) -> None:
        """A clean structured response round-trips through detect_sufficiency.

        Pins the non-retry path so the 'exactly once' assertion above cannot
        silently pass because the evaluator short-circuited before invoking.
        """
        result = sufficiency.SufficiencyResult(reasoning="ok", sufficient=True)
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = LLMResponse(
            content="",
            usage=UsageMetadata(input_tokens=1, output_tokens=1, total_tokens=2),
            raw=result,
        )
        llm = MagicMock()
        llm.with_structured_output.return_value = structured_llm
        template_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

        with patch.object(sufficiency, "get_llm", return_value=llm):
            sufficient, performed, reasoning, _meta = sufficiency.detect_sufficiency(
                raw_llm_response="Paris is the capital.",
                parsing_model=_parsing_model(),
                question_text="Capital of France?",
                template_schema=template_schema,
            )

        assert sufficient is True
        assert performed is True
        assert structured_llm.invoke.call_count == 1
