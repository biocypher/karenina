"""Tests that extract_usage_from_response reads message.usage_metadata.

Under LangChain streaming with stream_usage=True, the returned AIMessage has
tokens on message.usage_metadata but response_metadata["token_usage"] stays
None. Karenina must probe usage_metadata; otherwise we silently emit zeros.
"""

import pytest

from karenina.adapters.langchain.usage import extract_usage_from_response


class _StubMessage:
    """Minimal AIMessage stand-in exposing the two attributes we care about."""

    def __init__(
        self,
        *,
        response_metadata: dict | None = None,
        usage_metadata: dict | None = None,
    ) -> None:
        self.response_metadata = response_metadata or {}
        self.usage_metadata = usage_metadata


@pytest.mark.unit
class TestUsageExtractionStreaming:
    def test_usage_metadata_only_is_read(self) -> None:
        """Streaming path: response_metadata.token_usage is None,
        but usage_metadata carries tokens. Must read from usage_metadata."""
        msg = _StubMessage(
            response_metadata={"token_usage": None, "model_name": "qwen-test"},
            usage_metadata={"input_tokens": 72, "output_tokens": 30, "total_tokens": 102},
        )

        usage = extract_usage_from_response(msg, model_name="openai_endpoint:qwen-test")

        assert usage.input_tokens == 72
        assert usage.output_tokens == 30
        assert usage.total_tokens == 102

    def test_response_metadata_still_wins_when_both_present(self) -> None:
        """Non-streaming path: response_metadata has token_usage. Maintain
        existing preference to avoid changing behaviour on working code paths."""
        msg = _StubMessage(
            response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            usage_metadata={"input_tokens": 999, "output_tokens": 999, "total_tokens": 1998},
        )

        usage = extract_usage_from_response(msg, model_name="openai_endpoint:qwen-test")

        assert usage.input_tokens == 10
        assert usage.output_tokens == 5

    def test_both_empty_yields_zero_usage(self) -> None:
        """Truly missing usage. Extract should not raise; returns zeros."""
        msg = _StubMessage(response_metadata={}, usage_metadata=None)

        usage = extract_usage_from_response(msg, model_name="x")

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
