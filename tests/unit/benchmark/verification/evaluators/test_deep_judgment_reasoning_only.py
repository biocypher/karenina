"""Unit tests for deep judgment reasoning-only path.

This module tests the reasoning-only branch of deep_judgment_parse(),
which skips excerpt extraction and generates per-attribute reasoning
directly from the response before using ParserPort for parameter extraction.

Tests verify:
- Reasoning-only mode skips excerpts (excerpts dict is empty)
- Reasoning is populated for each attribute
- Metadata reflects reasoning-only (no "excerpts" in stages_completed)
- No excerpt retry logic runs (excerpt_retry_count == 0)
- Usage tracking is called for reasoning and parsing stages
"""

import json
from unittest.mock import Mock

import pytest
from pydantic import Field

from karenina.benchmark.verification.evaluators.template.deep_judgment import (
    _reasoning_only_parse,
    deep_judgment_parse,
)
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.parser import ParsePortResult
from karenina.ports.usage import UsageMetadata
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import BaseAnswer
from karenina.schemas.verification import VerificationConfig

# =============================================================================
# Test fixtures
# =============================================================================


class DrugAnswer(BaseAnswer):
    """Test answer template with two attributes."""

    id: str = ""
    drug_target: str = Field(default="", description="The protein target of the drug")
    mechanism: str = Field(default="", description="The mechanism of action")


def _make_llm_response(content: str) -> Mock:
    """Create a mock LLMResponse with content and usage."""
    resp = Mock()
    resp.content = content
    resp.usage = UsageMetadata(input_tokens=50, output_tokens=30, total_tokens=80)
    return resp


def _make_parsing_llm(invoke_responses: list[str]) -> Mock:
    """Create a mock LLMPort with sequential invoke responses."""
    mock_llm = Mock()
    call_index = [0]

    def mock_invoke(messages):
        idx = call_index[0] % len(invoke_responses)
        call_index[0] += 1
        return _make_llm_response(invoke_responses[idx])

    mock_llm.invoke.side_effect = mock_invoke
    return mock_llm


def _make_parser(parsed_answer: BaseAnswer) -> Mock:
    """Create a mock ParserPort returning a fixed parsed answer."""
    mock_parser = Mock()
    mock_parser.capabilities = PortCapabilities()
    mock_parser.parse_to_pydantic.return_value = ParsePortResult(
        parsed=parsed_answer,
        usage=UsageMetadata(input_tokens=40, output_tokens=20, total_tokens=60),
    )
    return mock_parser


def _make_config(reasoning_only: bool = True) -> VerificationConfig:
    """Create a VerificationConfig with reasoning-only mode."""
    parsing_model = ModelConfig(
        id="test-parsing",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        temperature=0.1,
        interface="langchain",
        system_prompt="You are a helpful assistant.",
    )
    return VerificationConfig(
        answering_models=[parsing_model],
        parsing_models=[parsing_model],
        deep_judgment_mode="reasoning_only" if reasoning_only else "full",
    )


def _make_model_config() -> ModelConfig:
    """Create a test ModelConfig."""
    return ModelConfig(
        id="test-model",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        temperature=0.1,
        interface="langchain",
        system_prompt="You are a helpful assistant.",
    )


# =============================================================================
# Tests for _reasoning_only_parse helper
# =============================================================================


@pytest.mark.unit
class TestReasoningOnlyParse:
    """Tests for the _reasoning_only_parse() helper function."""

    def test_returns_empty_excerpts(self):
        """Reasoning-only mode must return empty excerpts dict."""
        reasoning_json = json.dumps(
            {
                "drug_target": "The response mentions BCL-2 as the target.",
                "mechanism": "The response describes inhibition of BCL-2.",
            }
        )
        parsed = DrugAnswer(drug_target="BCL-2", mechanism="inhibition")
        parsing_llm = _make_parsing_llm([reasoning_json])
        parser = _make_parser(parsed)

        result_answer, excerpts, reasoning, metadata = _reasoning_only_parse(
            raw_llm_response="BCL-2 is the target and the mechanism is inhibition.",
            RawAnswer=DrugAnswer,
            parsing_llm=parsing_llm,
            parser=parser,
            question_text="What is the drug target?",
            config=_make_config(),
            generic_system_prompt="You are an expert.",
            attr_guidance="- drug_target: The protein target\n- mechanism: The mechanism of action",
            attribute_names=["drug_target", "mechanism"],
            usage_tracker=None,
            parsing_model_str=None,
            prompt_config=None,
            parsing_model=_make_model_config(),
            combined_system_prompt="You are an expert.",
        )

        assert excerpts == {}

    def test_reasoning_populated_for_each_attribute(self):
        """Reasoning dict must contain an entry for every attribute."""
        reasoning_json = json.dumps(
            {
                "drug_target": "BCL-2 is mentioned as the target.",
                "mechanism": "Inhibition is the described mechanism.",
            }
        )
        parsed = DrugAnswer(drug_target="BCL-2", mechanism="inhibition")
        parsing_llm = _make_parsing_llm([reasoning_json])
        parser = _make_parser(parsed)

        _, _, reasoning, _ = _reasoning_only_parse(
            raw_llm_response="BCL-2 is the target and the mechanism is inhibition.",
            RawAnswer=DrugAnswer,
            parsing_llm=parsing_llm,
            parser=parser,
            question_text="What is the drug target?",
            config=_make_config(),
            generic_system_prompt="You are an expert.",
            attr_guidance="- drug_target: The protein target\n- mechanism: The mechanism of action",
            attribute_names=["drug_target", "mechanism"],
            usage_tracker=None,
            parsing_model_str=None,
            prompt_config=None,
            parsing_model=_make_model_config(),
            combined_system_prompt="You are an expert.",
        )

        assert "drug_target" in reasoning
        assert "mechanism" in reasoning
        assert reasoning["drug_target"] == "BCL-2 is mentioned as the target."
        assert reasoning["mechanism"] == "Inhibition is the described mechanism."

    def test_metadata_reflects_reasoning_only(self):
        """Metadata must have reasoning_only=True and no 'excerpts' in stages_completed."""
        reasoning_json = json.dumps(
            {
                "drug_target": "Target reasoning.",
                "mechanism": "Mechanism reasoning.",
            }
        )
        parsed = DrugAnswer(drug_target="BCL-2", mechanism="inhibition")
        parsing_llm = _make_parsing_llm([reasoning_json])
        parser = _make_parser(parsed)

        _, _, _, metadata = _reasoning_only_parse(
            raw_llm_response="BCL-2 is the target.",
            RawAnswer=DrugAnswer,
            parsing_llm=parsing_llm,
            parser=parser,
            question_text="What is the drug target?",
            config=_make_config(),
            generic_system_prompt="You are an expert.",
            attr_guidance="- drug_target: The protein target\n- mechanism: The mechanism of action",
            attribute_names=["drug_target", "mechanism"],
            usage_tracker=None,
            parsing_model_str=None,
            prompt_config=None,
            parsing_model=_make_model_config(),
            combined_system_prompt="You are an expert.",
        )

        assert metadata["reasoning_only"] is True
        assert "excerpts" not in metadata["stages_completed"]
        assert metadata["stages_completed"] == ["reasoning", "parameters"]
        assert metadata["model_calls"] == 2
        assert metadata["excerpt_retry_count"] == 0
        assert metadata["attributes_without_excerpts"] == []

    def test_no_excerpt_retry_logic(self):
        """Excerpt retry count must be 0: no excerpt logic runs."""
        reasoning_json = json.dumps(
            {
                "drug_target": "Reasoning for drug target.",
                "mechanism": "Reasoning for mechanism.",
            }
        )
        parsed = DrugAnswer(drug_target="BCL-2", mechanism="inhibition")
        parsing_llm = _make_parsing_llm([reasoning_json])
        parser = _make_parser(parsed)

        _, _, _, metadata = _reasoning_only_parse(
            raw_llm_response="Some response.",
            RawAnswer=DrugAnswer,
            parsing_llm=parsing_llm,
            parser=parser,
            question_text="What is the drug target?",
            config=_make_config(),
            generic_system_prompt="You are an expert.",
            attr_guidance="- drug_target: desc\n- mechanism: desc",
            attribute_names=["drug_target", "mechanism"],
            usage_tracker=None,
            parsing_model_str=None,
            prompt_config=None,
            parsing_model=_make_model_config(),
            combined_system_prompt="You are an expert.",
        )

        assert metadata["excerpt_retry_count"] == 0

    def test_llm_called_once_for_reasoning(self):
        """The LLM must be invoked exactly once for the reasoning stage."""
        reasoning_json = json.dumps(
            {
                "drug_target": "Target reasoning.",
                "mechanism": "Mechanism reasoning.",
            }
        )
        parsed = DrugAnswer(drug_target="BCL-2", mechanism="inhibition")
        parsing_llm = _make_parsing_llm([reasoning_json])
        parser = _make_parser(parsed)

        _reasoning_only_parse(
            raw_llm_response="Some response.",
            RawAnswer=DrugAnswer,
            parsing_llm=parsing_llm,
            parser=parser,
            question_text="What is the drug target?",
            config=_make_config(),
            generic_system_prompt="You are an expert.",
            attr_guidance="- drug_target: desc\n- mechanism: desc",
            attribute_names=["drug_target", "mechanism"],
            usage_tracker=None,
            parsing_model_str=None,
            prompt_config=None,
            parsing_model=_make_model_config(),
            combined_system_prompt="You are an expert.",
        )

        assert parsing_llm.invoke.call_count == 1

    def test_parser_called_once_for_parameters(self):
        """ParserPort.parse_to_pydantic must be called exactly once."""
        reasoning_json = json.dumps(
            {
                "drug_target": "Target reasoning.",
                "mechanism": "Mechanism reasoning.",
            }
        )
        parsed = DrugAnswer(drug_target="BCL-2", mechanism="inhibition")
        parsing_llm = _make_parsing_llm([reasoning_json])
        parser = _make_parser(parsed)

        _reasoning_only_parse(
            raw_llm_response="Some response.",
            RawAnswer=DrugAnswer,
            parsing_llm=parsing_llm,
            parser=parser,
            question_text="What is the drug target?",
            config=_make_config(),
            generic_system_prompt="You are an expert.",
            attr_guidance="- drug_target: desc\n- mechanism: desc",
            attribute_names=["drug_target", "mechanism"],
            usage_tracker=None,
            parsing_model_str=None,
            prompt_config=None,
            parsing_model=_make_model_config(),
            combined_system_prompt="You are an expert.",
        )

        assert parser.parse_to_pydantic.call_count == 1

    def test_parsed_answer_returned(self):
        """The parsed answer from ParserPort must be returned as-is."""
        reasoning_json = json.dumps(
            {
                "drug_target": "Target reasoning.",
                "mechanism": "Mechanism reasoning.",
            }
        )
        expected = DrugAnswer(drug_target="BCL-2", mechanism="inhibition")
        parsing_llm = _make_parsing_llm([reasoning_json])
        parser = _make_parser(expected)

        result_answer, _, _, _ = _reasoning_only_parse(
            raw_llm_response="Some response.",
            RawAnswer=DrugAnswer,
            parsing_llm=parsing_llm,
            parser=parser,
            question_text="What is the drug target?",
            config=_make_config(),
            generic_system_prompt="You are an expert.",
            attr_guidance="- drug_target: desc\n- mechanism: desc",
            attribute_names=["drug_target", "mechanism"],
            usage_tracker=None,
            parsing_model_str=None,
            prompt_config=None,
            parsing_model=_make_model_config(),
            combined_system_prompt="You are an expert.",
        )

        assert result_answer.drug_target == "BCL-2"
        assert result_answer.mechanism == "inhibition"

    def test_usage_tracker_called_for_reasoning_and_parsing(self):
        """Usage tracker must receive calls for reasoning and parsing stages."""
        reasoning_json = json.dumps(
            {
                "drug_target": "Target reasoning.",
                "mechanism": "Mechanism reasoning.",
            }
        )
        parsed = DrugAnswer(drug_target="BCL-2", mechanism="inhibition")
        parsing_llm = _make_parsing_llm([reasoning_json])
        parser = _make_parser(parsed)
        tracker = Mock()

        _reasoning_only_parse(
            raw_llm_response="Some response.",
            RawAnswer=DrugAnswer,
            parsing_llm=parsing_llm,
            parser=parser,
            question_text="What is the drug target?",
            config=_make_config(),
            generic_system_prompt="You are an expert.",
            attr_guidance="- drug_target: desc\n- mechanism: desc",
            attribute_names=["drug_target", "mechanism"],
            usage_tracker=tracker,
            parsing_model_str="gpt-4.1-mini",
            prompt_config=None,
            parsing_model=_make_model_config(),
            combined_system_prompt="You are an expert.",
        )

        # Should be called for reasoning and parsing
        assert tracker.track_call.call_count == 2
        reasoning_call = tracker.track_call.call_args_list[0]
        assert reasoning_call[0][0] == "deep_judgment_reasoning"
        assert reasoning_call[0][1] == "gpt-4.1-mini"

        parsing_call = tracker.track_call.call_args_list[1]
        assert parsing_call[0][0] == "parsing"
        assert parsing_call[0][1] == "gpt-4.1-mini"

    def test_handles_malformed_reasoning_json_gracefully(self):
        """If reasoning JSON is invalid, reasoning dict should be empty but not crash."""
        parsing_llm = _make_parsing_llm(["not valid json at all"])
        parsed = DrugAnswer(drug_target="BCL-2", mechanism="inhibition")
        parser = _make_parser(parsed)

        _, _, reasoning, metadata = _reasoning_only_parse(
            raw_llm_response="Some response.",
            RawAnswer=DrugAnswer,
            parsing_llm=parsing_llm,
            parser=parser,
            question_text="What is the drug target?",
            config=_make_config(),
            generic_system_prompt="You are an expert.",
            attr_guidance="- drug_target: desc\n- mechanism: desc",
            attribute_names=["drug_target", "mechanism"],
            usage_tracker=None,
            parsing_model_str=None,
            prompt_config=None,
            parsing_model=_make_model_config(),
            combined_system_prompt="You are an expert.",
        )

        # Graceful degradation: empty reasoning, parsing still proceeds
        assert reasoning == {}
        assert metadata["stages_completed"] == ["reasoning", "parameters"]

    def test_handles_markdown_fenced_reasoning_json(self):
        """Reasoning JSON wrapped in markdown fences should be parsed correctly."""
        reasoning_json = json.dumps(
            {
                "drug_target": "Target reasoning.",
                "mechanism": "Mechanism reasoning.",
            }
        )
        fenced = f"```json\n{reasoning_json}\n```"
        parsed = DrugAnswer(drug_target="BCL-2", mechanism="inhibition")
        parsing_llm = _make_parsing_llm([fenced])
        parser = _make_parser(parsed)

        _, _, reasoning, _ = _reasoning_only_parse(
            raw_llm_response="Some response.",
            RawAnswer=DrugAnswer,
            parsing_llm=parsing_llm,
            parser=parser,
            question_text="What is the drug target?",
            config=_make_config(),
            generic_system_prompt="You are an expert.",
            attr_guidance="- drug_target: desc\n- mechanism: desc",
            attribute_names=["drug_target", "mechanism"],
            usage_tracker=None,
            parsing_model_str=None,
            prompt_config=None,
            parsing_model=_make_model_config(),
            combined_system_prompt="You are an expert.",
        )

        assert reasoning["drug_target"] == "Target reasoning."
        assert reasoning["mechanism"] == "Mechanism reasoning."


# =============================================================================
# Tests for deep_judgment_parse() branching
# =============================================================================


@pytest.mark.unit
class TestDeepJudgmentParseReasoningOnlyBranch:
    """Tests that deep_judgment_parse() dispatches to reasoning-only when config says so."""

    def test_reasoning_only_flag_triggers_reasoning_only_path(self):
        """When deep_judgment_mode='reasoning_only', excerpts are empty and metadata shows reasoning_only."""
        reasoning_json = json.dumps(
            {
                "drug_target": "Target reasoning from response.",
                "mechanism": "Mechanism reasoning from response.",
            }
        )
        parsed = DrugAnswer(drug_target="BCL-2", mechanism="inhibition")
        parsing_llm = _make_parsing_llm([reasoning_json])
        parser = _make_parser(parsed)
        config = _make_config(reasoning_only=True)
        model = _make_model_config()

        result_answer, excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response="BCL-2 is the target and the mechanism is inhibition.",
            RawAnswer=DrugAnswer,
            parsing_model=model,
            parsing_llm=parsing_llm,
            parser=parser,
            question_text="What is the drug target?",
            config=config,
            format_instructions="",
            combined_system_prompt="You are an expert.",
        )

        # Reasoning-only path indicators
        assert excerpts == {}
        assert metadata["reasoning_only"] is True
        assert "excerpts" not in metadata["stages_completed"]
        assert metadata["model_calls"] == 2

    def test_standard_path_when_reasoning_only_false(self):
        """When deep_judgment_mode='full', the standard excerpt path runs."""
        # Standard path needs: excerpt JSON, reasoning JSON, then parser
        excerpt_json = json.dumps(
            {
                "drug_target": [{"text": "BCL-2 is the target", "confidence": "high"}],
                "mechanism": [{"text": "mechanism is inhibition", "confidence": "high"}],
            }
        )
        reasoning_json = json.dumps(
            {
                "drug_target": "BCL-2 reasoning.",
                "mechanism": "Inhibition reasoning.",
            }
        )
        parsed = DrugAnswer(drug_target="BCL-2", mechanism="inhibition")
        parsing_llm = _make_parsing_llm([excerpt_json, reasoning_json])
        parser = _make_parser(parsed)
        config = _make_config(reasoning_only=False)
        model = _make_model_config()

        _, _, _, metadata = deep_judgment_parse(
            raw_llm_response="BCL-2 is the target and mechanism is inhibition of the protein.",
            RawAnswer=DrugAnswer,
            parsing_model=model,
            parsing_llm=parsing_llm,
            parser=parser,
            question_text="What is the drug target?",
            config=config,
            format_instructions="",
            combined_system_prompt="You are an expert.",
        )

        # Standard path indicators
        assert "excerpts" in metadata["stages_completed"]
        assert metadata.get("reasoning_only") is not True
