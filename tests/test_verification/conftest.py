"""Shared fixtures for verification stage tests."""

import pytest

from karenina.benchmark.verification.stage import VerificationContext
from karenina.schemas import ModelConfig
from karenina.schemas.domain import LLMRubricTrait, Rubric


@pytest.fixture
def basic_context() -> VerificationContext:
    """Create a basic VerificationContext for testing."""
    answering_model = ModelConfig(
        id="test-answering",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        temperature=0.1,
        interface="langchain",
        system_prompt="You are a helpful assistant.",
    )

    parsing_model = ModelConfig(
        id="test-parsing",
        model_provider="openai",
        model_name="gpt-4.1-mini",
        temperature=0.0,
        interface="langchain",
        system_prompt="Parse the response according to the template.",
    )

    return VerificationContext(
        question_id="test_q123",
        template_id="test_t456",
        question_text="What is 2 + 2?",
        template_code="""class Answer(BaseAnswer):
    result: int = Field(description="The arithmetic result")
    correct: dict = Field(description="Correct answer")

    def verify(self) -> bool:
        return self.result == 4
""",
        answering_model=answering_model,
        parsing_model=parsing_model,
        rubric=None,
        few_shot_enabled=False,
        abstention_enabled=False,
        deep_judgment_enabled=False,
    )


@pytest.fixture
def valid_template() -> str:
    """Return a valid answer template."""
    return """class Answer(BaseAnswer):
    result: int = Field(description="The arithmetic result")
    correct: dict = Field(description="Correct answer")

    def verify(self) -> bool:
        return self.result == 4
"""


@pytest.fixture
def invalid_template_syntax() -> str:
    """Return a template with syntax errors."""
    return """class Answer(BaseAnswer):
    result: int = Field(description="The arithmetic result"
    # Missing closing parenthesis and verify method
"""


@pytest.fixture
def invalid_template_missing_verify() -> str:
    """Return a template missing the verify method."""
    return """class Answer(BaseAnswer):
    result: int = Field(description="The arithmetic result")
    correct: dict = Field(description="Correct answer")
"""


@pytest.fixture
def sample_rubric() -> Rubric:
    """Create a sample rubric for testing."""
    return Rubric(
        llm_traits=[
            LLMRubricTrait(
                name="Accuracy",
                description="Is the answer accurate?",
                kind="score",
                min_score=1,
                max_score=10,
            ),
            LLMRubricTrait(
                name="Completeness",
                description="Is the answer complete?",
                kind="score",
                min_score=1,
                max_score=10,
            ),
        ],
    )


@pytest.fixture
def sample_llm_response() -> str:
    """Return a sample LLM response for testing."""
    return """The answer to 2 + 2 is 4. This is a basic arithmetic operation."""


@pytest.fixture
def sample_parsed_answer_dict() -> dict:
    """Return a sample parsed answer dictionary."""
    return {
        "result": 4,
        "correct": {"value": 4, "explanation": "Basic arithmetic"},
    }


@pytest.fixture
def deep_judgment_template() -> str:
    """Return a template suitable for deep-judgment parsing."""
    return """class Answer(BaseAnswer):
    reasoning: str = Field(description="Reasoning behind the answer")
    conclusion: str = Field(description="Final conclusion")
    correct: dict = Field(description="Correct answer")

    def verify(self) -> bool:
        return "4" in self.conclusion
"""


@pytest.fixture
def deep_judgment_response() -> str:
    """Return a response that requires deep-judgment parsing."""
    return """To solve this problem, I need to perform basic arithmetic.

The calculation 2 + 2 equals 4.

Therefore, the answer is 4.
"""


@pytest.fixture
def answering_model() -> ModelConfig:
    """Create a standard answering model config."""
    return ModelConfig(
        id="test-answering",
        model_provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.1,
        interface="langchain",
        system_prompt="You are a helpful assistant.",
    )


@pytest.fixture
def parsing_model() -> ModelConfig:
    """Create a standard parsing model config."""
    return ModelConfig(
        id="test-parsing",
        model_provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
        interface="langchain",
        system_prompt="Parse the response according to the template.",
    )
