"""Integration test configuration and fixtures.

This conftest provides fixtures for integration testing that combine multiple
components while using FixtureBackedLLMClient for deterministic LLM responses.

Key fixtures:
- template_evaluator: TemplateEvaluator with fixture-backed LLM
- rubric_evaluator: RubricEvaluator with fixture-backed LLM
- trace_with_citations: Sample LLM response with citation patterns
- trace_without_citations: Sample LLM response without citations
- answer_templates: Dictionary of loaded Answer classes from fixtures
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from karenina.benchmark.verification.evaluators import RubricEvaluator, TemplateEvaluator
from karenina.ports import LLMPort, LLMResponse
from karenina.ports.usage import UsageMetadata
from karenina.schemas.domain import BaseAnswer, LLMRubricTrait, RegexTrait, Rubric
from karenina.schemas.workflow import ModelConfig

# Import FixtureBackedLLMClient from root conftest
from tests.conftest import FixtureBackedLLMClient


class LLMPortFixtureAdapter:
    """Adapts FixtureBackedLLMClient to implement LLMPort interface.

    This adapter wraps the fixture-backed client to provide the LLMPort
    protocol required by evaluators.
    """

    def __init__(self, fixture_client: FixtureBackedLLMClient):
        self._client = fixture_client

    def invoke(self, messages: list[Any]) -> LLMResponse:
        """Invoke LLM and return LLMResponse."""
        from langchain_core.messages import AIMessage as LCAIMessage
        from langchain_core.messages import HumanMessage, SystemMessage

        from karenina.ports.messages import Message, Role

        # Convert Message objects to LangChain format
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, Message):
                # Convert port Message to LangChain message
                text = msg.text  # Extract text from TextContent blocks
                if msg.role == Role.SYSTEM:
                    langchain_messages.append(SystemMessage(content=text))
                elif msg.role == Role.USER:
                    langchain_messages.append(HumanMessage(content=text))
                elif msg.role == Role.ASSISTANT:
                    langchain_messages.append(LCAIMessage(content=text))
                else:
                    # Default to HumanMessage for other roles
                    langchain_messages.append(HumanMessage(content=text))
            else:
                # Already a LangChain message or dict
                langchain_messages.append(msg)

        ai_message = self._client.invoke(langchain_messages)

        # Convert AIMessage usage to UsageMetadata
        usage_data = ai_message.usage_metadata or {}
        usage = UsageMetadata(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return LLMResponse(content=ai_message.content, usage=usage, raw=ai_message)

    async def ainvoke(self, messages: list[Any]) -> LLMResponse:
        """Async invoke - calls sync version."""
        return self.invoke(messages)

    def with_structured_output(self, schema: type[BaseModel], *, max_retries: int | None = None) -> "LLMPort":
        """Return a structured output adapter."""
        return _StructuredOutputFixtureAdapter(self._client, schema, max_retries)


class _StructuredOutputFixtureAdapter:
    """Adapter for structured output using fixture-backed client."""

    def __init__(self, client: FixtureBackedLLMClient, schema: type[BaseModel], max_retries: int | None = None):
        self._client = client
        self._schema = schema
        self._max_retries = max_retries

    def invoke(self, messages: list[Any]) -> LLMResponse:
        """Invoke and parse response to schema."""
        import json

        from langchain_core.messages import AIMessage as LCAIMessage
        from langchain_core.messages import HumanMessage, SystemMessage

        from karenina.ports.messages import Message, Role

        # Convert Message objects to LangChain format
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, Message):
                # Convert port Message to LangChain message
                text = msg.text  # Extract text from TextContent blocks
                if msg.role == Role.SYSTEM:
                    langchain_messages.append(SystemMessage(content=text))
                elif msg.role == Role.USER:
                    langchain_messages.append(HumanMessage(content=text))
                elif msg.role == Role.ASSISTANT:
                    langchain_messages.append(LCAIMessage(content=text))
                else:
                    # Default to HumanMessage for other roles
                    langchain_messages.append(HumanMessage(content=text))
            else:
                # Already a LangChain message or dict
                langchain_messages.append(msg)

        ai_message = self._client.invoke(langchain_messages)

        # Convert AIMessage usage to UsageMetadata
        usage_data = ai_message.usage_metadata or {}
        usage = UsageMetadata(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # Try to parse content as JSON and validate with schema
        content = ai_message.content
        try:
            data = json.loads(content)
            parsed = self._schema.model_validate(data)
        except (json.JSONDecodeError, Exception):
            # Return raw content if parsing fails
            parsed = None

        return LLMResponse(content=content, usage=usage, raw=parsed)

    async def ainvoke(self, messages: list[Any]) -> LLMResponse:
        """Async invoke - calls sync version."""
        return self.invoke(messages)

    def with_structured_output(self, schema: type[BaseModel], *, max_retries: int | None = None) -> "LLMPort":
        """Chain structured output (replace schema)."""
        return _StructuredOutputFixtureAdapter(self._client, schema, max_retries)


# =============================================================================
# Model Configuration Fixtures
# =============================================================================


@pytest.fixture
def parsing_model_config() -> ModelConfig:
    """Return a ModelConfig for parsing/evaluation models.

    Uses langchain interface with claude-haiku-4-5 for consistency
    with captured fixtures.
    """
    return ModelConfig(
        id="test-parser",
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        temperature=0.0,
        interface="langchain",
    )


# =============================================================================
# Evaluator Fixtures (with fixture-backed LLM)
# =============================================================================


@pytest.fixture
def llm_port_adapter(llm_client: FixtureBackedLLMClient) -> LLMPortFixtureAdapter:
    """Return an LLMPort adapter wrapping the fixture client."""
    return LLMPortFixtureAdapter(llm_client)


@pytest.fixture
def template_evaluator(
    parsing_model_config: ModelConfig,
    llm_client: FixtureBackedLLMClient,
) -> TemplateEvaluator:
    """Create a TemplateEvaluator with fixture-backed LLM.

    This evaluator uses captured LLM responses for deterministic testing.
    The Answer class must be set per-test using evaluator.answer_class = MyAnswer.

    Note: TemplateEvaluator still uses init_chat_model_unified (not LLMPort),
    so we patch that function and return the raw fixture client which has
    the LangChain .invoke() interface.

    Example:
        def test_parse(template_evaluator, simple_answer):
            template_evaluator.answer_class = simple_answer
            result = template_evaluator.parse_response(...)
    """

    # Create a minimal Answer class for initialization
    class MinimalAnswer(BaseAnswer):
        value: str = ""

        def verify(self) -> bool:
            return True

    # Patch get_llm since TemplateEvaluator uses factory functions
    with patch("karenina.benchmark.verification.evaluators.template.evaluator.get_llm") as mock_get_llm:
        mock_get_llm.return_value = llm_client
        evaluator = TemplateEvaluator(
            model_config=parsing_model_config,
            answer_class=MinimalAnswer,
        )

    return evaluator


@pytest.fixture
def rubric_evaluator(
    parsing_model_config: ModelConfig,
    llm_port_adapter: LLMPortFixtureAdapter,
) -> RubricEvaluator:
    """Create a RubricEvaluator with fixture-backed LLM.

    This evaluator uses captured LLM responses for deterministic testing.

    Example:
        def test_rubric_eval(rubric_evaluator, sample_rubric):
            results, labels, usage = rubric_evaluator.evaluate_rubric(
                question="What is 2+2?",
                answer="The answer is 4.",
                rubric=sample_rubric,
            )
    """
    with patch("karenina.benchmark.verification.evaluators.rubric_evaluator.get_llm") as mock_get_llm:
        mock_get_llm.return_value = llm_port_adapter
        evaluator = RubricEvaluator(
            model_config=parsing_model_config,
            evaluation_strategy="batch",
        )

    return evaluator


# =============================================================================
# Trace Fixtures (Sample LLM Responses)
# =============================================================================


@pytest.fixture
def trace_with_citations() -> str:
    """Return a sample LLM response trace with citation patterns [1], [2], etc.

    This trace includes:
    - Multiple citation references
    - Scientific content
    - Proper paragraph structure
    """
    return """BCL2 is a proto-oncogene located on chromosome 18q21.33 [1]. It encodes
a protein that plays a critical role in regulating apoptosis, making it an important
target in cancer research [2].

The BCL2 protein belongs to the BCL-2 family of proteins, which includes both
pro-apoptotic and anti-apoptotic members [1][3]. BCL2 specifically inhibits
programmed cell death by preventing the release of cytochrome c from mitochondria [2].

Overexpression of BCL2 has been observed in various cancers, including:
- B-cell lymphomas (where it was first discovered) [1]
- Chronic lymphocytic leukemia [4]
- Breast cancer [5]

References:
[1] Tsujimoto et al., Science, 1985
[2] Hockenbery et al., Nature, 1990
[3] Adams & Cory, Science, 1998
[4] Robertson et al., Blood, 2007
[5] Krajewski et al., Cancer Res, 1995"""


@pytest.fixture
def trace_without_citations() -> str:
    """Return a sample LLM response trace without citation patterns.

    This trace includes:
    - Scientific content
    - No citation markers
    - Clean prose style
    """
    return """BCL2 is a proto-oncogene located on chromosome 18q21.33. It encodes
a protein that plays a critical role in regulating apoptosis, making it an important
target in cancer research.

The BCL2 protein belongs to the BCL-2 family of proteins, which includes both
pro-apoptotic and anti-apoptotic members. BCL2 specifically inhibits programmed
cell death by preventing the release of cytochrome c from mitochondria.

Overexpression of BCL2 has been observed in various cancers, including B-cell
lymphomas (where it was first discovered), chronic lymphocytic leukemia, and
breast cancer. This makes it an important therapeutic target."""


@pytest.fixture
def trace_with_abstention() -> str:
    """Return a sample LLM response that indicates abstention/refusal.

    This trace demonstrates a model declining to answer.
    """
    return """I apologize, but I cannot provide specific medical advice or dosage
recommendations. This type of information should only come from a qualified
healthcare provider who can evaluate your individual situation.

Please consult with your doctor or pharmacist for personalized guidance
on this matter."""


@pytest.fixture
def trace_with_hedging() -> str:
    """Return a sample LLM response that hedges but still provides an answer.

    This trace demonstrates uncertainty language with substantive content.
    """
    return """While I cannot be completely certain, the evidence suggests that
BCL2 is the most likely candidate gene in this case.

The gene is located on chromosome 18q21.33 and encodes an anti-apoptotic protein.
Based on the symptoms described, BCL2 overexpression could explain the observed
phenotype, though additional testing would be needed to confirm this hypothesis."""


# =============================================================================
# Answer Template Fixtures
# =============================================================================


@pytest.fixture
def simple_answer() -> type[BaseAnswer]:
    """Return the simple extraction Answer class from fixtures.

    This template has a single string field for basic extraction testing.
    """
    from tests.fixtures.templates.simple_extraction import Answer

    return Answer


@pytest.fixture
def multi_field_answer() -> type[BaseAnswer]:
    """Return the multi-field Answer class from fixtures.

    This template has nested structures, lists, and optional fields.
    """
    from tests.fixtures.templates.multi_field import Answer

    return Answer


@pytest.fixture
def answer_with_correct_dict() -> type[BaseAnswer]:
    """Return the Answer class with ground truth from model_post_init.

    This template demonstrates setting correct values via model_post_init.
    """
    from tests.fixtures.templates.with_correct_dict import Answer

    return Answer


@pytest.fixture
def answer_templates(
    simple_answer: type[BaseAnswer],
    multi_field_answer: type[BaseAnswer],
    answer_with_correct_dict: type[BaseAnswer],
) -> dict[str, type[BaseAnswer]]:
    """Return dictionary of all available Answer template classes.

    Keys are template names, values are Answer classes.
    """
    return {
        "simple_extraction": simple_answer,
        "multi_field": multi_field_answer,
        "with_correct_dict": answer_with_correct_dict,
    }


# =============================================================================
# Rubric Fixtures
# =============================================================================


@pytest.fixture
def boolean_rubric() -> Rubric:
    """Return a rubric with a single boolean LLM trait.

    Tests clarity/quality binary assessment.
    """
    clarity_trait = LLMRubricTrait(
        name="clarity",
        description="The response is clear, unambiguous, and easy to understand",
        kind="boolean",
    )
    return Rubric(llm_traits=[clarity_trait])


@pytest.fixture
def scored_rubric() -> Rubric:
    """Return a rubric with a scored LLM trait (1-5 scale).

    Tests quality rating on a numeric scale.
    """
    completeness_trait = LLMRubricTrait(
        name="completeness",
        description="The response thoroughly addresses all aspects of the question",
        kind="score",
        min_score=1,
        max_score=5,
    )
    return Rubric(llm_traits=[completeness_trait])


@pytest.fixture
def multi_trait_rubric() -> Rubric:
    """Return a rubric with multiple trait types.

    Includes boolean, scored, and regex traits for comprehensive testing.
    """
    accuracy_trait = LLMRubricTrait(
        name="accuracy",
        description="The response contains factually correct information",
        kind="boolean",
    )
    helpfulness_trait = LLMRubricTrait(
        name="helpfulness",
        description="The response is helpful and addresses the user's actual need",
        kind="score",
        min_score=1,
        max_score=5,
    )
    citation_trait = RegexTrait(
        name="has_citations",
        pattern=r"\[\d+\]",
        description="Response includes numeric citations like [1], [2]",
    )
    return Rubric(
        llm_traits=[accuracy_trait, helpfulness_trait],
        regex_traits=[citation_trait],
    )


@pytest.fixture
def citation_regex_rubric() -> Rubric:
    """Return a rubric with only regex traits for citation checking.

    Tests deterministic pattern matching without LLM calls.
    """
    citation_trait = RegexTrait(
        name="has_citations",
        pattern=r"\[\d+\]",
        description="Response includes numeric citations like [1], [2]",
    )
    url_trait = RegexTrait(
        name="has_urls",
        pattern=r"https?://\S+",
        description="Response includes URLs",
    )
    return Rubric(regex_traits=[citation_trait, url_trait])


@pytest.fixture
def literal_sentiment_rubric() -> Rubric:
    """Return a rubric with a literal kind LLM trait for sentiment classification.

    Tests categorical classification with three classes.
    """
    sentiment_trait = LLMRubricTrait(
        name="sentiment",
        description="Classify the emotional tone of the response",
        kind="literal",
        classes={
            "negative": "Response expresses criticism, disappointment, or pessimism",
            "neutral": "Response is factual and emotionally neutral",
            "positive": "Response expresses optimism, satisfaction, or enthusiasm",
        },
        higher_is_better=True,  # positive > neutral > negative
    )
    return Rubric(llm_traits=[sentiment_trait])


@pytest.fixture
def multi_literal_rubric() -> Rubric:
    """Return a rubric with multiple literal kind LLM traits.

    Tests batch evaluation of multiple categorical classifications.
    """
    sentiment_trait = LLMRubricTrait(
        name="sentiment",
        description="Classify the emotional tone of the response",
        kind="literal",
        classes={
            "negative": "Response expresses criticism, disappointment, or pessimism",
            "neutral": "Response is factual and emotionally neutral",
            "positive": "Response expresses optimism, satisfaction, or enthusiasm",
        },
        higher_is_better=True,
    )
    response_type_trait = LLMRubricTrait(
        name="response_type",
        description="Classify the type of response given",
        kind="literal",
        classes={
            "factual": "Response presents objective facts or data",
            "opinion": "Response expresses subjective views or preferences",
            "speculative": "Response discusses possibilities or hypotheticals",
            "refusal": "Response declines to answer or redirects the question",
        },
        higher_is_better=False,  # Order doesn't imply quality
    )
    return Rubric(llm_traits=[sentiment_trait, response_type_trait])


@pytest.fixture
def mixed_rubric_with_literal() -> Rubric:
    """Return a rubric mixing literal, boolean, and score traits.

    Tests that literal traits can coexist with other trait kinds.
    """
    sentiment_trait = LLMRubricTrait(
        name="sentiment",
        description="Classify the emotional tone of the response",
        kind="literal",
        classes={
            "negative": "Response expresses criticism, disappointment, or pessimism",
            "neutral": "Response is factual and emotionally neutral",
            "positive": "Response expresses optimism, satisfaction, or enthusiasm",
        },
        higher_is_better=True,
    )
    accuracy_trait = LLMRubricTrait(
        name="accuracy",
        description="The response contains factually correct information",
        kind="boolean",
        higher_is_better=True,
    )
    completeness_trait = LLMRubricTrait(
        name="completeness",
        description="The response thoroughly addresses all aspects of the question",
        kind="score",
        min_score=1,
        max_score=5,
        higher_is_better=True,
    )
    return Rubric(llm_traits=[sentiment_trait, accuracy_trait, completeness_trait])


# =============================================================================
# Integration Test Helpers
# =============================================================================

# NOTE: fixtures_dir is inherited from the root conftest.py at tests/conftest.py
# Do not redefine it here - pytest will automatically use the parent fixture


@pytest.fixture
def checkpoint_fixtures_dir(fixtures_dir: Path) -> Path:
    """Return the path to checkpoint fixture files."""
    return fixtures_dir / "checkpoints"


def load_checkpoint_fixture(checkpoint_fixtures_dir: Path, name: str) -> Any:
    """Load a checkpoint fixture by name.

    Args:
        checkpoint_fixtures_dir: Path to checkpoints directory
        name: Fixture name (without .jsonld extension)

    Returns:
        Loaded Benchmark instance
    """
    from karenina import Benchmark

    fixture_path = checkpoint_fixtures_dir / f"{name}.jsonld"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Checkpoint fixture not found: {fixture_path}")
    return Benchmark.load(fixture_path)


@pytest.fixture
def minimal_benchmark(checkpoint_fixtures_dir: Path) -> Any:
    """Load the minimal checkpoint fixture (1 question).

    This benchmark has a single simple question for basic testing.
    """
    return load_checkpoint_fixture(checkpoint_fixtures_dir, "minimal")


@pytest.fixture
def multi_question_benchmark(checkpoint_fixtures_dir: Path) -> Any:
    """Load the multi-question checkpoint fixture (5 questions).

    This benchmark has diverse questions for comprehensive testing.
    """
    return load_checkpoint_fixture(checkpoint_fixtures_dir, "multi_question")


@pytest.fixture
def benchmark_with_results(checkpoint_fixtures_dir: Path) -> Any:
    """Load the checkpoint fixture with existing verification results.

    This benchmark has pre-computed results for testing resume/aggregation.
    """
    return load_checkpoint_fixture(checkpoint_fixtures_dir, "with_results")
