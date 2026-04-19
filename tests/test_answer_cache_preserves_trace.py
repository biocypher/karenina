"""Tests that AnswerTraceCache preserves trace_messages and conversation_context.

These fields are stored as list[dict] on VerificationResultTemplate per
schemas/verification/result_components.py. The cache must persist them so that
parsing-model-variant cache hits can re-emit the structured trace.
"""

import pytest

from karenina.benchmark.verification.utils.cache_helpers import (
    extract_answer_data_from_result,
)
from karenina.schemas.verification import VerificationResult
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultTemplate,
)


def _make_result(
    *,
    trace_messages: list[dict] | None,
    conversation_context: list[dict] | None,
) -> VerificationResult:
    metadata = VerificationResultMetadata(
        question_id="urn:uuid:q-test",
        template_id="0" * 32,
        question_text="What is 2+2?",
        answering=ModelIdentity(interface="openai_endpoint", model_name="qwen-test"),
        parsing=ModelIdentity(interface="openai_endpoint", model_name="qwen-test"),
        execution_time=1.0,
        timestamp="2026-04-19 12:00:00",
        result_id="0" * 16,
    )
    template = VerificationResultTemplate(
        raw_llm_response="--- AI Message ---\n4",
        trace_messages=trace_messages or [],
        conversation_context=conversation_context or [],
    )
    return VerificationResult(metadata=metadata, template=template)


@pytest.mark.unit
class TestCachePreservesStructuredTrace:
    def test_extract_includes_trace_messages(self) -> None:
        """extract_answer_data_from_result must include trace_messages."""
        trace = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        result = _make_result(trace_messages=trace, conversation_context=None)

        cached = extract_answer_data_from_result(result)

        assert "trace_messages" in cached
        assert cached["trace_messages"] == trace

    def test_extract_includes_conversation_context(self) -> None:
        """extract_answer_data_from_result must include conversation_context."""
        context = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        result = _make_result(trace_messages=None, conversation_context=context)

        cached = extract_answer_data_from_result(result)

        assert "conversation_context" in cached
        assert cached["conversation_context"] == context

    def test_extract_empty_trace_serializes_as_none(self) -> None:
        """Empty list should serialize as None (equivalent to absent), matching existing
        retrieval conditional: generate_answer.py checks `if trace_messages_data:` and
        falsy empty list skips re-hydration the same way None does.
        """
        result = _make_result(trace_messages=[], conversation_context=[])

        cached = extract_answer_data_from_result(result)

        assert cached["trace_messages"] is None
        assert cached["conversation_context"] is None

    def test_cache_round_trip_preserves_conversation_context(self) -> None:
        """Round-trip through AnswerTraceCache.set/get preserves conversation_context.

        Regression guard: a cache hit must expose the same conversation_context
        dicts that were stored, so the retrieval path in generate_answer.py can
        re-hydrate them into Messages symmetrically to trace_messages.
        """
        from karenina.utils.answer_cache import AnswerTraceCache

        context = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        result = _make_result(trace_messages=None, conversation_context=context)
        cached = extract_answer_data_from_result(result)

        cache = AnswerTraceCache()
        key = "q-test_model-test_rep0"

        status, _ = cache.get_or_reserve(key)
        assert status == "MISS"
        cache.complete(key, cached)

        status, retrieved = cache.get_or_reserve(key)
        assert status == "HIT"
        assert retrieved is not None
        assert retrieved["conversation_context"] == context
