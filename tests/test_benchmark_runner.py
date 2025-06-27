import warnings
from unittest.mock import MagicMock, patch

from pydantic import Field

from karenina.benchmark.runner import run_benchmark
from karenina.schemas.answer_class import BaseAnswer


# Answer class for testing
class BenchmarkTestAnswer(BaseAnswer):
    answer: str = Field(description="Test answer")
    confidence: float = Field(default=1.0, description="Confidence level")

    def model_post_init(self, __context):
        self.id = "test"
        self.correct = True


def test_run_benchmark_success():
    """Test successful benchmark run with matching keys."""
    # Mock data
    question_dict = {"q1": "What is 2+2?", "q2": "What is the capital of France?"}
    response_dict = {"q1": "2+2 equals 4", "q2": "The capital of France is Paris"}
    answer_templates = {"q1": BenchmarkTestAnswer, "q2": BenchmarkTestAnswer}

    # Mock LLM response
    mock_response_1 = BenchmarkTestAnswer(answer="4", confidence=0.9)
    mock_response_2 = BenchmarkTestAnswer(answer="Paris", confidence=0.95)

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.side_effect = [mock_response_1, mock_response_2]
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("karenina.llm.interface.init_chat_model", return_value=mock_llm):
        result = run_benchmark(question_dict, response_dict, answer_templates)

        assert len(result) == 2
        assert "q1" in result
        assert "q2" in result
        assert result["q1"] == mock_response_1
        assert result["q2"] == mock_response_2

        # Verify LLM was called correctly
        assert mock_llm.with_structured_output.call_count == 2
        assert mock_structured_llm.invoke.call_count == 2


def test_run_benchmark_mismatched_keys():
    """Test benchmark run with mismatched keys triggers warning."""
    question_dict = {"q1": "What is 2+2?", "q2": "What is the capital of France?", "q3": "Extra question"}
    response_dict = {
        "q1": "2+2 equals 4",
        "q2": "The capital of France is Paris",
        # Missing q3
    }
    answer_templates = {
        "q1": BenchmarkTestAnswer,
        "q2": BenchmarkTestAnswer,
        "q4": BenchmarkTestAnswer,  # Different key
    }

    mock_response = BenchmarkTestAnswer(answer="test", confidence=0.8)
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_response
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("karenina.llm.interface.init_chat_model", return_value=mock_llm):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_benchmark(question_dict, response_dict, answer_templates)

            # Check that warning was raised
            assert len(w) == 1
            assert "different keys" in str(w[0].message)

            # Check that only common keys are processed
            assert len(result) == 2  # q1 and q2 are common
            assert "q1" in result
            assert "q2" in result


def test_run_benchmark_empty_intersection():
    """Test benchmark run with no common keys."""
    question_dict = {"q1": "Question 1"}
    response_dict = {"q2": "Response 2"}
    answer_templates = {"q3": BenchmarkTestAnswer}

    mock_llm = MagicMock()

    with patch("karenina.llm.interface.init_chat_model", return_value=mock_llm):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_benchmark(question_dict, response_dict, answer_templates)

            # Check that warning was raised
            assert len(w) == 1
            assert "different keys" in str(w[0].message)

            # Check that no items are processed
            assert len(result) == 0


def test_run_benchmark_single_item():
    """Test benchmark run with a single item."""
    question_dict = {"single": "Single question?"}
    response_dict = {"single": "Single response"}
    answer_templates = {"single": BenchmarkTestAnswer}

    mock_response = BenchmarkTestAnswer(answer="single_answer", confidence=0.5)
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_response
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("karenina.llm.interface.init_chat_model", return_value=mock_llm):
        result = run_benchmark(question_dict, response_dict, answer_templates)

        assert len(result) == 1
        assert "single" in result
        assert result["single"] == mock_response


def test_run_benchmark_llm_init_parameters():
    """Test that LLM is initialized with correct parameters."""
    question_dict = {"test": "Test question?"}
    response_dict = {"test": "Test response"}
    answer_templates = {"test": BenchmarkTestAnswer}

    mock_response = BenchmarkTestAnswer(answer="test", confidence=1.0)
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_response
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("karenina.llm.interface.init_chat_model", return_value=mock_llm) as mock_init:
        run_benchmark(question_dict, response_dict, answer_templates)

        # Verify LLM was initialized with expected parameters
        mock_init.assert_called_once_with(
            model="gemini-2.5-flash-preview-05-20", model_provider="google_genai"
        )


def test_run_benchmark_message_format():
    """Test that messages are formatted correctly for the LLM."""
    question_dict = {"test": "Test question?"}
    response_dict = {"test": "Test response"}
    answer_templates = {"test": BenchmarkTestAnswer}

    mock_response = BenchmarkTestAnswer(answer="test", confidence=1.0)
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_response
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("karenina.llm.interface.init_chat_model", return_value=mock_llm):
        run_benchmark(question_dict, response_dict, answer_templates)

        # Check that invoke was called with proper message structure
        mock_structured_llm.invoke.assert_called_once()
        messages = mock_structured_llm.invoke.call_args[0][0]

        assert len(messages) == 2
        # Check that messages contain the question and response
        user_message_content = messages[1].content
        assert "Test question?" in user_message_content
        assert "Test response" in user_message_content


def test_run_benchmark_identical_keys():
    """Test benchmark run when all dictionaries have identical keys."""
    question_dict = {"q1": "Question 1", "q2": "Question 2"}
    response_dict = {"q1": "Response 1", "q2": "Response 2"}
    answer_templates = {"q1": BenchmarkTestAnswer, "q2": BenchmarkTestAnswer}

    mock_response = BenchmarkTestAnswer(answer="test", confidence=1.0)
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_response
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("karenina.llm.interface.init_chat_model", return_value=mock_llm):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_benchmark(question_dict, response_dict, answer_templates)

            # No warning should be raised when keys are identical
            assert len(w) == 0
            assert len(result) == 2
