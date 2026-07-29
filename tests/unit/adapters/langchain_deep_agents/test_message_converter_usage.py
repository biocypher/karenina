"""Tests that DeepAgentsMessageConverter.from_provider populates usage_metadata.

Fix E (extension): the production trace path in
langchain_deep_agents/agent.py builds ``trace_messages`` via
``self._converter.from_provider(lc_messages)`` rather than the direct
``deep_agents_messages_to_trace_messages`` helper. The converter must
attach per-call usage to each assistant ``Message`` so that
``Message.to_dict()`` emits it into ``template.trace_messages[*]``.
"""

from __future__ import annotations

import pytest

from karenina.adapters.langchain_deep_agents.messages import DeepAgentsMessageConverter
from karenina.ports import Role


@pytest.mark.unit
class TestDeepAgentsConverterUsageMetadata:
    def test_modern_usage_metadata_propagates(self) -> None:
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="Hello world",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
        )

        converter = DeepAgentsMessageConverter()
        [out] = converter.from_provider([msg])

        assert out.role == Role.ASSISTANT
        assert out.usage_metadata == {"input_tokens": 100, "output_tokens": 50}

    def test_to_dict_emits_usage_metadata_after_conversion(self) -> None:
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="Hello",
            usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        )

        converter = DeepAgentsMessageConverter()
        [out] = converter.from_provider([msg])

        assert out.to_dict()["usage_metadata"] == {
            "input_tokens": 7,
            "output_tokens": 3,
        }

    def test_cache_fields_propagate_when_reported(self) -> None:
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="cached",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 50,
            },
        )

        converter = DeepAgentsMessageConverter()
        [out] = converter.from_provider([msg])

        assert out.usage_metadata == {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 50,
        }

    def test_response_metadata_token_usage_fallback(self) -> None:
        """Older LangChain adapters use response_metadata.token_usage with
        prompt_tokens / completion_tokens — converter must rename keys.
        """
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="Legacy adapter response",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 75,
                    "total_tokens": 275,
                }
            },
        )

        converter = DeepAgentsMessageConverter()
        [out] = converter.from_provider([msg])

        assert out.usage_metadata == {"input_tokens": 200, "output_tokens": 75}

    def test_no_usage_yields_none(self) -> None:
        from langchain_core.messages import AIMessage

        msg = AIMessage(content="no usage")

        converter = DeepAgentsMessageConverter()
        [out] = converter.from_provider([msg])

        assert out.usage_metadata is None
        assert "usage_metadata" not in out.to_dict()

    def test_modern_usage_metadata_wins_over_response_metadata(self) -> None:
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="both",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            response_metadata={"token_usage": {"prompt_tokens": 999, "completion_tokens": 888}},
        )

        converter = DeepAgentsMessageConverter()
        [out] = converter.from_provider([msg])

        # The modern usage_metadata path is preferred per _extract_ai_usage_metadata.
        assert out.usage_metadata == {"input_tokens": 10, "output_tokens": 5}
