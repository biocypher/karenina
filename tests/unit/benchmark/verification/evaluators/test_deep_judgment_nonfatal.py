"""Behavioral tests for non-fatal hallucination assessment in deep judgment.

Stage 1.5 (per-excerpt hallucination assessment) runs only when search
enhancement produced ``search_results`` for at least one excerpt. The LLM
call inside that stage is wrapped in a broad try/except: a failure must
log a warning and continue, leaving per-excerpt ``hallucination_risk``
unset. Stage 2's risk aggregation then defaults every attribute to
``"high"`` (the conservative fallback when no per-excerpt risk is
available).

These tests exercise that contract end-to-end through the real
``deep_judgment_parse`` SUT: the parsing LLM is sequenced to fail exactly
on the Stage 1.5 assessment call, the search tool is stubbed so
``search_performed`` flips True, and we assert the function returns
normally with ``hallucination_risk == {"<attr>": "high"}`` for every
attribute that had search results.
"""

import json
from unittest.mock import Mock

import pytest
from pydantic import Field

from karenina.benchmark.verification.evaluators.template import deep_judgment
from karenina.benchmark.verification.evaluators.template.deep_judgment import deep_judgment_parse
from karenina.ports.capabilities import PortCapabilities
from karenina.ports.parser import ParsePortResult
from karenina.ports.usage import UsageMetadata
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import BaseAnswer
from karenina.schemas.verification import VerificationConfig


class DrugAnswer(BaseAnswer):
    """Single-attribute answer template used by every test in this module."""

    id: str = ""
    drug_target: str = Field(default="", description="The protein target of the drug")


def _llm_response(content: str) -> Mock:
    resp = Mock()
    resp.content = content
    resp.usage = UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)
    return resp


def _parsing_model() -> ModelConfig:
    return ModelConfig(
        id="parser",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        interface="langchain",
        system_prompt="You are a helpful assistant.",
        temperature=0.1,
    )


def _config_with_search() -> VerificationConfig:
    return VerificationConfig(
        answering_models=[_parsing_model()],
        parsing_models=[_parsing_model()],
        deep_judgment_mode="full",
        deep_judgment_search_enabled=True,
        deep_judgment_search_tool="tavily",
    )


def _parser_returning(answer: BaseAnswer) -> Mock:
    parser = Mock()
    parser.capabilities = PortCapabilities()
    parser.parse_to_pydantic.return_value = ParsePortResult(
        parsed=answer,
        usage=UsageMetadata(input_tokens=20, output_tokens=10, total_tokens=30),
    )
    return parser


@pytest.mark.unit
class TestHallucinationAssessmentNonFatal:
    """Stage 1.5 hallucination assessment failure must not break deep judgment."""

    def test_assessment_failure_defaults_each_attr_to_high_risk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A raising Stage 1.5 LLM call leaves hallucination_risk at the 'high' fallback.

        The parsing LLM is sequenced to:
          call 1 (excerpt extraction) -> valid excerpt JSON for ``drug_target``
          call 2 (Stage 1.5 assessment) -> raises RuntimeError
          call 3 (Stage 2 reasoning) -> valid reasoning JSON

        The search tool is stubbed to attach ``search_results`` to the
        excerpt, which is what flips ``search_performed`` to True and gates
        Stage 1.5. The contract: deep_judgment_parse returns normally,
        ``"excerpt_hallucination_assessment"`` is NOT in stages_completed,
        and metadata['hallucination_risk']['drug_target'] == 'high'.
        """
        excerpt_json = json.dumps({"drug_target": [{"text": "BCL-2 is the target", "confidence": "high"}]})
        reasoning_json = json.dumps({"drug_target": "BCL-2 reasoning."})
        parsed = DrugAnswer(drug_target="BCL-2")

        call_count = [0]

        def fake_invoke(_messages):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("assessment LLM exploded")
            return _llm_response(excerpt_json if call_count[0] == 1 else reasoning_json)

        parsing_llm = Mock()
        parsing_llm.invoke.side_effect = fake_invoke

        # Stub the search tool so each excerpt gets search_results attached,
        # flipping search_performed=True and gating Stage 1.5.
        def fake_search_tool(_tool_name):
            def _run(queries):
                # One batch of dict results per query; deep_judgment stores
                # them verbatim when the items are already dicts.
                return [[{"title": "t", "content": "c", "url": "u"}] for _ in queries]

            return _run

        monkeypatch.setattr(deep_judgment, "create_search_tool", fake_search_tool)

        returned_answer, _excerpts, reasoning, metadata = deep_judgment_parse(
            raw_llm_response="BCL-2 is the target of the drug.",
            RawAnswer=DrugAnswer,
            parsing_model=_parsing_model(),
            parsing_llm=parsing_llm,
            parser=_parser_returning(parsed),
            question_text="What is the drug target?",
            config=_config_with_search(),
            format_instructions="",
            combined_system_prompt="You are an expert.",
        )

        # The assessment failure did not propagate.
        assert returned_answer.drug_target == "BCL-2"
        assert reasoning == {"drug_target": "BCL-2 reasoning."}
        # All three LLM calls happened (excerpts, failed assessment, reasoning).
        assert parsing_llm.invoke.call_count == 3
        # Stage 1.5 is recorded as attempted-but-skipped: its stage name is
        # absent while Stage 2 ('reasoning') still ran.
        assert "excerpt_hallucination_assessment" not in metadata["stages_completed"]
        assert "reasoning" in metadata["stages_completed"]
        # Conservative fallback: every searched attribute defaults to 'high'.
        assert metadata["hallucination_risk"] == {"drug_target": "high"}

    def test_assessment_success_records_per_excerpt_risk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A successful Stage 1.5 with a 'low' risk is reflected in metadata.

        Pins the non-fatal path so the 'assessment failure' test above cannot
        silently pass because Stage 1.5 was never reached. Here the assessment
        LLM returns a valid ``excerpt_assessments`` payload marking the
        excerpt as 'low' risk, and metadata['hallucination_risk'] must mirror
        that instead of falling back to 'high'.
        """
        excerpt_json = json.dumps({"drug_target": [{"text": "BCL-2 is the target", "confidence": "high"}]})
        assessment_json = json.dumps(
            {"excerpt_assessments": [{"excerpt_id": "0", "hallucination_risk": "low", "justification": "ok"}]}
        )
        reasoning_json = json.dumps({"drug_target": "BCL-2 reasoning."})
        parsed = DrugAnswer(drug_target="BCL-2")

        responses = [excerpt_json, assessment_json, reasoning_json]
        parsing_llm = Mock()
        parsing_llm.invoke.side_effect = [_llm_response(c) for c in responses]

        def fake_search_tool(_tool_name):
            def _run(queries):
                return [[{"title": "t", "content": "c", "url": "u"}] for _ in queries]

            return _run

        monkeypatch.setattr(deep_judgment, "create_search_tool", fake_search_tool)

        _answer, _excerpts, _reasoning, metadata = deep_judgment_parse(
            raw_llm_response="BCL-2 is the target of the drug.",
            RawAnswer=DrugAnswer,
            parsing_model=_parsing_model(),
            parsing_llm=parsing_llm,
            parser=_parser_returning(parsed),
            question_text="What is the drug target?",
            config=_config_with_search(),
            format_instructions="",
            combined_system_prompt="You are an expert.",
        )

        assert "excerpt_hallucination_assessment" in metadata["stages_completed"]
        assert metadata["hallucination_risk"] == {"drug_target": "low"}
