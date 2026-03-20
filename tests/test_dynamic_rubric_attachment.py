"""Tests for dynamic_rubric / question_dynamic_rubric field additions.

Verifies that Question accepts an optional question_dynamic_rubric dict
and that VerificationContext accepts an optional dynamic_rubric field
holding a DynamicRubric instance.
"""

import pytest

from karenina.benchmark.verification.stages.core.base import VerificationContext
from karenina.schemas.config import ModelConfig
from karenina.schemas.entities.question import Question
from karenina.schemas.entities.rubric import DynamicRubric, LLMRubricTrait

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_model_config() -> ModelConfig:
    """Return a minimal ModelConfig for testing."""
    return ModelConfig(
        id="test-model",
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        temperature=0.0,
    )


@pytest.fixture
def minimal_context(minimal_model_config: ModelConfig) -> VerificationContext:
    """Return a minimal VerificationContext for testing."""
    return VerificationContext(
        question_id="test-question-1",
        template_id="template-hash-123",
        question_text="What is BCL2?",
        template_code=(
            "from pydantic import Field\n"
            "from karenina.schemas.entities import BaseAnswer\n\n"
            "class Answer(BaseAnswer):\n"
            "    target: str = Field(description='The target')\n\n"
            "    def verify(self) -> bool:\n"
            "        return self.target.lower() == 'bcl2'\n"
        ),
        answering_model=minimal_model_config,
        parsing_model=minimal_model_config,
    )


# =============================================================================
# Question Tests
# =============================================================================


@pytest.mark.unit
class TestQuestionDynamicRubric:
    """Tests for the question_dynamic_rubric field on Question."""

    def test_default_none(self) -> None:
        q = Question(question="What is BCL2?", raw_answer="BCL2 is a protein.")
        assert q.question_dynamic_rubric is None

    def test_accepts_dict(self) -> None:
        q = Question(
            question="What is BCL2?",
            raw_answer="BCL2 is a protein.",
            question_dynamic_rubric={"llm_traits": []},
        )
        assert q.question_dynamic_rubric == {"llm_traits": []}

    def test_roundtrip_through_model_dump(self) -> None:
        payload = {"llm_traits": [{"name": "safety", "description": "Is it safe?"}]}
        q = Question(
            question="What is BCL2?",
            raw_answer="BCL2 is a protein.",
            question_dynamic_rubric=payload,
        )
        dumped = q.model_dump()
        assert dumped["question_dynamic_rubric"] == payload

    def test_none_excluded_from_dump_when_unset(self) -> None:
        q = Question(question="What is BCL2?", raw_answer="BCL2 is a protein.")
        dumped = q.model_dump(exclude_none=True)
        assert "question_dynamic_rubric" not in dumped


# =============================================================================
# VerificationContext Tests
# =============================================================================


@pytest.mark.unit
class TestVerificationContextDynamicRubric:
    """Tests for the dynamic_rubric field on VerificationContext."""

    def test_default_none(self, minimal_context: VerificationContext) -> None:
        assert minimal_context.dynamic_rubric is None

    def test_accepts_dynamic_rubric(self, minimal_model_config: ModelConfig) -> None:
        trait = LLMRubricTrait(
            name="safety",
            description="Is the response safe for a general audience?",
            kind="boolean",
        )
        dr = DynamicRubric(llm_traits=[trait])

        ctx = VerificationContext(
            question_id="test-q",
            template_id="tmpl-hash",
            question_text="What is BCL2?",
            template_code="class Answer: pass",
            answering_model=minimal_model_config,
            parsing_model=minimal_model_config,
            dynamic_rubric=dr,
        )
        assert ctx.dynamic_rubric is dr
        assert len(ctx.dynamic_rubric.llm_traits) == 1
        assert ctx.dynamic_rubric.llm_traits[0].name == "safety"

    def test_coexists_with_static_rubric(self, minimal_model_config: ModelConfig) -> None:
        """Both rubric and dynamic_rubric can be set independently."""
        from karenina.schemas.entities import Rubric

        trait = LLMRubricTrait(
            name="clarity",
            description="Is the response clear?",
            kind="boolean",
        )
        static = Rubric(llm_traits=[trait])
        dynamic = DynamicRubric(llm_traits=[trait])

        ctx = VerificationContext(
            question_id="test-q",
            template_id="tmpl-hash",
            question_text="What is BCL2?",
            template_code="class Answer: pass",
            answering_model=minimal_model_config,
            parsing_model=minimal_model_config,
            rubric=static,
            dynamic_rubric=dynamic,
        )
        assert ctx.rubric is static
        assert ctx.dynamic_rubric is dynamic
