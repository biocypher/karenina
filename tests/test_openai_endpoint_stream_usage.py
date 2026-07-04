"""Tests that ChatOpenAIEndpoint enables streaming usage by default.

vLLM's OpenAI-compatible endpoint drops the usage block on streaming responses
unless stream_options.include_usage=True (equivalently, stream_usage=True on
langchain-openai's ChatOpenAI). karenina constructs ChatOpenAIEndpoint under
streaming for all openai_endpoint calls, so the default must include usage.
"""

import pytest
from pydantic import SecretStr

from karenina.adapters.langchain.models import ChatOpenAIEndpoint


@pytest.mark.unit
class TestChatOpenAIEndpointStreamUsage:
    def test_default_construction_enables_stream_usage(self) -> None:
        """By default, ChatOpenAIEndpoint must request usage in streaming mode."""
        client = ChatOpenAIEndpoint(
            base_url="http://localhost:8000",
            openai_api_key=SecretStr("EMPTY"),
            model="qwen-test",
        )
        # langchain-openai exposes the flag as `stream_usage` on ChatOpenAI.
        # The request stream_options.include_usage is derived from this.
        assert getattr(client, "stream_usage", False) is True

    def test_caller_can_override_stream_usage(self) -> None:
        """Callers who explicitly disable streaming usage must win."""
        client = ChatOpenAIEndpoint(
            base_url="http://localhost:8000",
            openai_api_key=SecretStr("EMPTY"),
            model="qwen-test",
            stream_usage=False,
        )
        assert client.stream_usage is False
