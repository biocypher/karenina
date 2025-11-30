"""Tests for rubric parsing utilities."""

import pytest

from karenina.benchmark.verification.evaluators.rubric_parsing import (
    parse_boolean_from_text,
    parse_raw_response,
    parse_score_from_text,
)
from karenina.schemas.workflow.rubric_outputs import (
    BatchRubricScores,
    ConfusionMatrixOutput,
    SingleBooleanScore,
    SingleNumericScore,
)


class TestParseRawResponse:
    """Test parse_raw_response with various input formats."""

    def test_parse_batch_scores_direct_json(self) -> None:
        """Test parsing direct JSON for BatchRubricScores."""
        response = '{"scores": {"accuracy": true, "quality": 4}}'
        result = parse_raw_response(response, BatchRubricScores)

        assert result.scores["accuracy"] is True
        assert result.scores["quality"] == 4

    def test_parse_batch_scores_with_wrapper(self) -> None:
        """Test parsing JSON with scores wrapper."""
        response = '{"scores": {"trait1": false, "trait2": 3}}'
        result = parse_raw_response(response, BatchRubricScores)

        assert result.scores["trait1"] is False
        assert result.scores["trait2"] == 3

    def test_parse_batch_scores_without_wrapper(self) -> None:
        """Test parsing JSON without scores wrapper (legacy format)."""
        response = '{"accuracy": true, "completeness": 4}'
        result = parse_raw_response(response, BatchRubricScores)

        assert result.scores["accuracy"] is True
        assert result.scores["completeness"] == 4

    def test_parse_batch_scores_from_markdown(self) -> None:
        """Test extracting JSON from markdown code blocks."""
        response = """Here is my evaluation:
```json
{"scores": {"accuracy": true}}
```
"""
        result = parse_raw_response(response, BatchRubricScores)
        assert result.scores["accuracy"] is True

    def test_parse_batch_scores_with_trailing_comma(self) -> None:
        """Test JSON repair for trailing comma."""
        # Trailing comma is a common LLM error
        response = '{"scores": {"accuracy": true,}}'
        result = parse_raw_response(response, BatchRubricScores)
        assert result.scores["accuracy"] is True

    def test_parse_single_boolean_score(self) -> None:
        """Test parsing SingleBooleanScore."""
        response = '{"result": true}'
        result = parse_raw_response(response, SingleBooleanScore)
        assert result.result is True

    def test_parse_single_boolean_score_false(self) -> None:
        """Test parsing SingleBooleanScore with false value."""
        response = '{"result": false}'
        result = parse_raw_response(response, SingleBooleanScore)
        assert result.result is False

    def test_parse_single_numeric_score(self) -> None:
        """Test parsing SingleNumericScore."""
        response = '{"score": 4}'
        result = parse_raw_response(response, SingleNumericScore)
        assert result.score == 4

    def test_parse_confusion_matrix_output(self) -> None:
        """Test parsing ConfusionMatrixOutput."""
        response = '{"tp": ["a", "b"], "fn": ["c"], "fp": [], "tn": ["d"]}'
        result = parse_raw_response(response, ConfusionMatrixOutput)

        assert result.tp == ["a", "b"]
        assert result.fn == ["c"]
        assert result.fp == []
        assert result.tn == ["d"]

    def test_parse_confusion_matrix_empty_lists(self) -> None:
        """Test parsing ConfusionMatrixOutput with empty lists."""
        response = '{"tp": [], "fn": [], "fp": [], "tn": []}'
        result = parse_raw_response(response, ConfusionMatrixOutput)

        assert result.tp == []
        assert result.fn == []
        assert result.fp == []
        assert result.tn == []

    def test_parse_from_mixed_text(self) -> None:
        """Test extracting JSON from mixed text with reasoning."""
        response = """Let me analyze this response carefully.

After reviewing the answer, here is my evaluation:
{"scores": {"accuracy": true, "completeness": 3}}

The accuracy is high because the answer correctly identifies the main points."""

        result = parse_raw_response(response, BatchRubricScores)
        assert result.scores["accuracy"] is True
        assert result.scores["completeness"] == 3

    def test_parse_already_model_instance(self) -> None:
        """Test passing already-parsed model instance."""
        existing = BatchRubricScores(scores={"test": True})
        result = parse_raw_response(existing, BatchRubricScores)
        assert result is existing

    def test_parse_invalid_json_raises_error(self) -> None:
        """Test that completely invalid JSON raises ValueError."""
        response = "This is not JSON at all"
        with pytest.raises(ValueError, match="Could not parse response"):
            parse_raw_response(response, BatchRubricScores)


class TestParseBooleanFromText:
    """Test boolean parsing from text."""

    def test_parse_true_keyword(self) -> None:
        """Test parsing 'true' keyword."""
        assert parse_boolean_from_text("true") is True
        assert parse_boolean_from_text("TRUE") is True
        assert parse_boolean_from_text("True") is True

    def test_parse_false_keyword(self) -> None:
        """Test parsing 'false' keyword."""
        assert parse_boolean_from_text("false") is False
        assert parse_boolean_from_text("FALSE") is False
        assert parse_boolean_from_text("False") is False

    def test_parse_yes_no(self) -> None:
        """Test parsing yes/no."""
        assert parse_boolean_from_text("yes") is True
        assert parse_boolean_from_text("no") is False
        assert parse_boolean_from_text("YES") is True
        assert parse_boolean_from_text("NO") is False

    def test_parse_numeric(self) -> None:
        """Test parsing numeric values."""
        assert parse_boolean_from_text("1") is True
        assert parse_boolean_from_text("0") is False

    def test_parse_from_sentence(self) -> None:
        """Test extracting boolean from longer text."""
        assert parse_boolean_from_text("Yes, the answer is good.") is True
        assert parse_boolean_from_text("No, this is wrong.") is False
        assert parse_boolean_from_text("The result is true.") is True
        assert parse_boolean_from_text("The evaluation is false.") is False

    def test_parse_pass_fail(self) -> None:
        """Test parsing pass/fail keywords."""
        assert parse_boolean_from_text("pass") is True
        assert parse_boolean_from_text("passed") is True
        assert parse_boolean_from_text("fail") is False
        assert parse_boolean_from_text("failed") is False

    def test_parse_unknown_returns_none(self) -> None:
        """Test that text without clear keywords returns None."""
        assert parse_boolean_from_text("maybe") is None
        # Note: text containing keywords like "no" in "unknown" will match
        assert parse_boolean_from_text("") is None


class TestParseScoreFromText:
    """Test score parsing from text."""

    def test_parse_direct_number(self) -> None:
        """Test parsing direct numeric value."""
        assert parse_score_from_text("4", 1, 5) == 4
        assert parse_score_from_text("3", 1, 5) == 3
        assert parse_score_from_text("1", 1, 5) == 1

    def test_parse_number_in_sentence(self) -> None:
        """Test extracting number from sentence."""
        assert parse_score_from_text("I rate this a 4 out of 5", 1, 5) == 4
        assert parse_score_from_text("Score: 3", 1, 5) == 3
        assert parse_score_from_text("The rating is 5", 1, 5) == 5

    def test_parse_clamps_above_max(self) -> None:
        """Test that scores above max are clamped."""
        assert parse_score_from_text("10", 1, 5) == 5
        assert parse_score_from_text("100", 1, 5) == 5

    def test_parse_clamps_below_min(self) -> None:
        """Test that scores below min are clamped."""
        assert parse_score_from_text("0", 1, 5) == 1
        assert parse_score_from_text("-5", 1, 5) == 1

    def test_parse_no_number_returns_none(self) -> None:
        """Test that text without numbers returns None."""
        assert parse_score_from_text("no number here", 1, 5) is None
        assert parse_score_from_text("", 1, 5) is None

    def test_parse_takes_first_number(self) -> None:
        """Test that first number is used when multiple present."""
        assert parse_score_from_text("4 out of 5 stars", 1, 5) == 4
        assert parse_score_from_text("Score 3, Rating 5", 1, 5) == 3


class TestPydanticModels:
    """Test the Pydantic models themselves."""

    def test_batch_rubric_scores_validation(self) -> None:
        """Test BatchRubricScores model validation."""
        scores = BatchRubricScores(scores={"trait1": True, "trait2": 4})
        assert scores.scores["trait1"] is True
        assert scores.scores["trait2"] == 4

    def test_single_boolean_score_validation(self) -> None:
        """Test SingleBooleanScore model validation."""
        score = SingleBooleanScore(result=True)
        assert score.result is True

        score = SingleBooleanScore(result=False)
        assert score.result is False

    def test_single_numeric_score_validation(self) -> None:
        """Test SingleNumericScore model validation."""
        score = SingleNumericScore(score=4)
        assert score.score == 4

    def test_confusion_matrix_output_defaults(self) -> None:
        """Test ConfusionMatrixOutput with default empty lists."""
        output = ConfusionMatrixOutput()
        assert output.tp == []
        assert output.fn == []
        assert output.fp == []
        assert output.tn == []

    def test_confusion_matrix_output_with_values(self) -> None:
        """Test ConfusionMatrixOutput with values."""
        output = ConfusionMatrixOutput(
            tp=["found1", "found2"],
            fn=["missing1"],
            fp=["wrong1"],
            tn=["correct_absent"],
        )
        assert output.tp == ["found1", "found2"]
        assert output.fn == ["missing1"]
        assert output.fp == ["wrong1"]
        assert output.tn == ["correct_absent"]
