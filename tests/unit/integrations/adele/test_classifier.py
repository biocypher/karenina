"""Unit tests for ADeLe question classifier."""

from unittest.mock import MagicMock

import pytest

from karenina.integrations.adele import (
    QuestionClassificationResult,
    QuestionClassifier,
)
from karenina.integrations.adele.schemas import AdeleTraitInfo
from karenina.ports.usage import UsageMetadata


class TestQuestionClassificationResult:
    """Tests for QuestionClassificationResult schema."""

    def test_basic_result(self) -> None:
        """Test creating a basic result."""
        result = QuestionClassificationResult(
            question_id="q1",
            question_text="What is 2+2?",
            scores={"attention_and_scan": 0, "volume": 1},
            labels={"attention_and_scan": "none", "volume": "very_low"},
            model="test-model",
            classified_at="2026-01-18T00:00:00Z",
        )

        assert result.question_id == "q1"
        assert result.question_text == "What is 2+2?"
        assert result.scores["attention_and_scan"] == 0
        assert result.labels["attention_and_scan"] == "none"
        assert result.model == "test-model"

    def test_to_checkpoint_metadata(self) -> None:
        """Test converting result to checkpoint metadata format."""
        result = QuestionClassificationResult(
            question_id="q1",
            question_text="Test question",
            scores={"attention_and_scan": 3},
            labels={"attention_and_scan": "intermediate"},
            model="haiku",
            classified_at="2026-01-18T12:00:00Z",
        )

        metadata = result.to_checkpoint_metadata()

        assert "adele_classification" in metadata
        assert metadata["adele_classification"]["scores"] == {"attention_and_scan": 3}
        assert metadata["adele_classification"]["labels"] == {"attention_and_scan": "intermediate"}
        assert metadata["adele_classification"]["model"] == "haiku"
        assert metadata["adele_classification"]["classified_at"] == "2026-01-18T12:00:00Z"

    def test_from_checkpoint_metadata(self) -> None:
        """Test creating result from checkpoint metadata."""
        metadata = {
            "adele_classification": {
                "scores": {"volume": 4},
                "labels": {"volume": "high"},
                "model": "sonnet",
                "classified_at": "2026-01-18T00:00:00Z",
            }
        }

        result = QuestionClassificationResult.from_checkpoint_metadata(
            metadata, question_id="q2", question_text="Complex question"
        )

        assert result is not None
        assert result.question_id == "q2"
        assert result.scores["volume"] == 4
        assert result.labels["volume"] == "high"
        assert result.model == "sonnet"

    def test_from_checkpoint_metadata_missing(self) -> None:
        """Test that from_checkpoint_metadata returns None if no classification."""
        metadata = {"other_data": "value"}

        result = QuestionClassificationResult.from_checkpoint_metadata(metadata)

        assert result is None

    def test_get_summary(self) -> None:
        """Test summary generation."""
        result = QuestionClassificationResult(
            question_text="Test",
            scores={"attention_and_scan": 3, "volume": -1},
            labels={"attention_and_scan": "intermediate", "volume": "invalid_class"},
        )

        summary = result.get_summary()

        assert summary["attention_and_scan"] == "intermediate (3)"
        assert summary["volume"] == "error: invalid_class"


class TestAdeleTraitInfo:
    """Tests for AdeleTraitInfo schema."""

    def test_basic_info(self) -> None:
        """Test creating basic trait info."""
        info = AdeleTraitInfo(
            name="attention_and_scan",
            code="AS",
            description="Attention requirements",
            classes={"none": "Level 0", "very_low": "Level 1"},
            class_names=["none", "very_low"],
        )

        assert info.name == "attention_and_scan"
        assert info.code == "AS"
        assert len(info.classes) == 2


class TestQuestionClassifier:
    """Tests for QuestionClassifier class."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        classifier = QuestionClassifier()

        assert classifier._llm is None
        assert classifier._model_name == "claude-3-5-haiku-latest"
        assert classifier._provider == "anthropic"
        assert classifier._temperature == 0.0

    def test_initialization_custom(self) -> None:
        """Test custom initialization."""
        mock_llm = MagicMock()
        classifier = QuestionClassifier(
            llm=mock_llm,
            model_name="gpt-4",
            provider="openai",
            temperature=0.5,
        )

        assert classifier._llm is mock_llm
        assert classifier._model_name == "gpt-4"
        assert classifier._provider == "openai"
        assert classifier._temperature == 0.5

    def test_build_system_prompt(self) -> None:
        """Test system prompt generation."""
        classifier = QuestionClassifier()
        prompt = classifier._build_system_prompt()

        assert "ADeLe" in prompt
        assert "QUESTION" in prompt
        assert "cognitive complexity" in prompt.lower()
        assert "JSON" in prompt

    def test_build_user_prompt(self) -> None:
        """Test user prompt generation."""
        from karenina.integrations.adele import get_adele_trait

        classifier = QuestionClassifier()
        traits = [get_adele_trait("attention_and_scan"), get_adele_trait("volume")]
        prompt = classifier._build_user_prompt("What is 2+2?", traits)

        assert "What is 2+2?" in prompt
        assert "attention_and_scan" in prompt
        assert "volume" in prompt
        assert "classifications" in prompt

    def test_classify_single_mocked(self) -> None:
        """Test classify_single with mocked LLM call."""
        # Create a mock LLM that supports with_structured_output
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        # Mock the response from structured LLM
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_result.classifications = {
            "attention_and_scan": "none",
            "volume": "very_low",
        }
        mock_response.raw = mock_result
        mock_response.usage = UsageMetadata(input_tokens=100, output_tokens=50, total_tokens=150)
        mock_structured_llm.invoke.return_value = mock_response

        classifier = QuestionClassifier(llm=mock_llm)

        # Classify with only 2 traits for simplicity
        result = classifier.classify_single(
            question_text="What is 2+2?",
            trait_names=["attention_and_scan", "volume"],
            question_id="q1",
        )

        assert result.question_id == "q1"
        assert result.question_text == "What is 2+2?"
        assert result.scores["attention_and_scan"] == 0  # "none" -> index 0
        assert result.scores["volume"] == 1  # "very_low" -> index 1
        assert result.labels["attention_and_scan"] == "none"
        assert result.labels["volume"] == "very_low"
        # Usage metadata is now converted from dataclass
        assert result.usage_metadata["input_tokens"] == 100
        assert result.usage_metadata["output_tokens"] == 50

    def test_classify_batch_mocked(self) -> None:
        """Test classify_batch with mocked LLM calls."""
        # Create a mock LLM that supports with_structured_output
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        # Mock response for batch mode
        mock_response = MagicMock()
        mock_result = MagicMock()
        mock_result.classifications = {
            "attention_and_scan": "intermediate",
            "volume": "high",
        }
        mock_response.raw = mock_result
        mock_response.usage = UsageMetadata(input_tokens=100, output_tokens=50, total_tokens=150)
        mock_structured_llm.invoke.return_value = mock_response

        classifier = QuestionClassifier(llm=mock_llm)

        # Track progress
        progress_calls = []

        def on_progress(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        # Classify batch
        questions = [
            ("q1", "Simple question?"),
            ("q2", "Complex question?"),
        ]
        results = classifier.classify_batch(
            questions=questions,
            trait_names=["attention_and_scan", "volume"],
            on_progress=on_progress,
        )

        assert len(results) == 2
        assert "q1" in results
        assert "q2" in results
        assert results["q1"].scores["attention_and_scan"] == 3  # "intermediate" -> index 3
        assert len(progress_calls) == 2
        assert progress_calls[-1] == (2, 2)

    def test_validate_classifications_valid(self) -> None:
        """Test validation of valid classifications."""
        from karenina.integrations.adele import get_adele_trait

        classifier = QuestionClassifier()
        traits = [get_adele_trait("attention_and_scan")]
        classifications = {"attention_and_scan": "intermediate"}

        scores, labels = classifier._validate_classifications(classifications, traits)

        assert scores["attention_and_scan"] == 3
        assert labels["attention_and_scan"] == "intermediate"

    def test_validate_classifications_case_insensitive(self) -> None:
        """Test that classification validation is case-insensitive."""
        from karenina.integrations.adele import get_adele_trait

        classifier = QuestionClassifier()
        traits = [get_adele_trait("attention_and_scan")]
        classifications = {"attention_and_scan": "INTERMEDIATE"}

        scores, labels = classifier._validate_classifications(classifications, traits)

        assert scores["attention_and_scan"] == 3
        assert labels["attention_and_scan"] == "intermediate"

    def test_validate_classifications_invalid_class(self) -> None:
        """Test validation with invalid class name."""
        from karenina.integrations.adele import get_adele_trait

        classifier = QuestionClassifier()
        traits = [get_adele_trait("attention_and_scan")]
        classifications = {"attention_and_scan": "invalid_class"}

        scores, labels = classifier._validate_classifications(classifications, traits)

        assert scores["attention_and_scan"] == -1
        assert labels["attention_and_scan"] == "invalid_class"

    def test_validate_classifications_missing_trait(self) -> None:
        """Test validation when trait is missing from response."""
        from karenina.integrations.adele import get_adele_trait

        classifier = QuestionClassifier()
        traits = [get_adele_trait("attention_and_scan"), get_adele_trait("volume")]
        classifications = {"attention_and_scan": "low"}  # volume missing

        scores, labels = classifier._validate_classifications(classifications, traits)

        assert scores["attention_and_scan"] == 2
        assert scores["volume"] == -1
        assert labels["volume"] == "[MISSING_FROM_RESPONSE]"


class TestQuestionClassifierIntegration:
    """Integration tests for QuestionClassifier (require API keys)."""

    @pytest.mark.skip(reason="Requires API key - run manually with ANTHROPIC_API_KEY set")
    def test_classify_single_real_api(self) -> None:
        """Test classify_single with real API call."""
        classifier = QuestionClassifier()
        result = classifier.classify_single(
            question_text="What is the capital of France?",
            trait_names=["attention_and_scan", "volume"],
        )

        assert result.scores["attention_and_scan"] >= 0
        assert result.scores["volume"] >= 0
        assert result.labels["attention_and_scan"] in [
            "none",
            "very_low",
            "low",
            "intermediate",
            "high",
            "very_high",
        ]
