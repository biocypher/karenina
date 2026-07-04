"""Hot integration test for LLMRubricTrait template kind with a real LLM call.

This test verifies that template-kind LLM rubric traits work end-to-end against
an actual Anthropic API, confirming that the structured-output call produces a
validated instance of the user-defined Pydantic schema and that the evaluator
flattens the result into dotted keys (``trait.field``).

Tests are marked with:
- @pytest.mark.integration
- @pytest.mark.rubric

Run with:
    pytest tests/integration/test_llm_trait_template_api.py -v

Requires: ANTHROPIC_API_KEY environment variable to be set.
"""

import os

import pytest
from pydantic import BaseModel, Field

from karenina.benchmark.verification.evaluators.rubric.evaluator import RubricEvaluator
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import LLMRubricTrait, Rubric

pytestmark = [
    pytest.mark.integration,
    pytest.mark.rubric,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set; skipping real API tests",
    ),
]

MODEL_NAME = "claude-sonnet-4-6"
MODEL_PROVIDER = "anthropic"
INTERFACE = "langchain"


class CitationCheck(BaseModel):
    """Template schema exercised by the hot test.

    Kept to primitive field types (bool, int, list[str]) so the round-trip
    through JSON Schema and the rubric kind validator stays within the
    supported template surface.
    """

    has_citations: bool = Field(description="True if the response includes any citations.")
    citation_count: int = Field(description="Number of distinct citations in the response.")
    cited_sources: list[str] = Field(description="Source identifiers cited (e.g., author names or titles).")


def _make_model_config() -> ModelConfig:
    return ModelConfig(
        id="template-hot-test",
        model_provider=MODEL_PROVIDER,
        model_name=MODEL_NAME,
        interface=INTERFACE,
        temperature=0.0,
    )


def test_template_trait_live_api() -> None:
    """End-to-end: template-kind trait produces dotted-key results from claude-sonnet-4-6."""
    question = "What are two commonly cited papers that introduced transformer models?"
    answer = (
        "The transformer architecture was introduced by Vaswani et al. 2017 "
        "in the paper 'Attention Is All You Need', and BERT by Devlin et al. 2018 "
        "built on it with bidirectional pre-training."
    )

    trait = LLMRubricTrait(
        name="citations",
        description=(
            "Extract citation evidence from the response: whether citations "
            "appear, how many, and which sources are referenced."
        ),
        kind=CitationCheck,
        higher_is_better=None,
    )
    rubric = Rubric(llm_traits=[trait])

    evaluator = RubricEvaluator(
        _make_model_config(),
        evaluation_strategy="sequential",
    )

    results, labels, usage_list = evaluator.evaluate_rubric(question, answer, rubric)

    # Dotted keys for each template field
    assert set(results.keys()) == {
        "citations.has_citations",
        "citations.citation_count",
        "citations.cited_sources",
    }, f"unexpected result keys: {sorted(results.keys())}"

    # Field types match the Pydantic schema
    assert isinstance(results["citations.has_citations"], bool)
    assert isinstance(results["citations.citation_count"], int)
    assert isinstance(results["citations.cited_sources"], list)
    assert all(isinstance(s, str) for s in results["citations.cited_sources"])

    # Ground-truth-style sanity checks on the response content
    assert results["citations.has_citations"] is True
    assert results["citations.citation_count"] >= 2

    # Literal labels are not produced for template traits
    assert labels is None

    # Usage metadata is recorded for the one template call
    assert len(usage_list) == 1
    assert usage_list[0]  # non-empty dict
