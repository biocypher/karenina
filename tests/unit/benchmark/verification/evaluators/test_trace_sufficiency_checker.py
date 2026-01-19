"""Unit tests for sufficiency_checker evaluator.

Tests cover:
- _strip_markdown_fences helper function
- is_retryable_error helper function
- detect_sufficiency function logic
  - Handling of various LLM response formats
  - Error handling and failure modes
  - Return value semantics

Note: These tests use mocking instead of actual LLM calls.
For fixture-backed tests with real LLM responses, see integration tests.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from karenina.benchmark.verification.evaluators.trace_sufficiency_checker import (
    SufficiencyResult,
    detect_sufficiency,
)
from karenina.benchmark.verification.utils.error_helpers import is_retryable_error
from karenina.benchmark.verification.utils.json_helpers import strip_markdown_fences as _strip_markdown_fences
from karenina.ports import LLMResponse
from karenina.ports.usage import UsageMetadata
from karenina.schemas.workflow import ModelConfig


def _create_mock_llm_response(
    content: str,
    raw: SufficiencyResult | None = None,
) -> LLMResponse:
    """Create a mock LLMResponse for testing."""
    return LLMResponse(
        content=content,
        usage=UsageMetadata(input_tokens=100, output_tokens=50, total_tokens=150),
        raw=raw,
    )


# =============================================================================
# Helper Function Tests: _strip_markdown_fences
# =============================================================================


@pytest.mark.unit
class TestStripMarkdownFences:
    """Test _strip_markdown_fences helper function."""

    def test_removes_json_fences(self) -> None:
        """Verify ```json fences are removed."""
        text = '```json\n{"key": "value"}\n```'
        result = _strip_markdown_fences(text)
        assert result == '{"key": "value"}'

    def test_removes_plain_fences(self) -> None:
        """Verify plain ``` fences are removed."""
        text = '```\n{"key": "value"}\n```'
        result = _strip_markdown_fences(text)
        assert result == '{"key": "value"}'

    def test_handles_no_fences(self) -> None:
        """Verify text without fences is returned unchanged."""
        text = '{"key": "value"}'
        result = _strip_markdown_fences(text)
        assert result == '{"key": "value"}'

    def test_handles_extra_whitespace(self) -> None:
        """Verify extra whitespace is stripped."""
        text = '  ```json\n{"key": "value"}\n```  '
        result = _strip_markdown_fences(text)
        assert result == '{"key": "value"}'

    def test_handles_non_string_input(self) -> None:
        """Verify non-string input is returned unchanged."""
        result = _strip_markdown_fences(None)
        assert result is None

        result = _strip_markdown_fences(123)
        assert result == 123

    def test_handles_empty_string(self) -> None:
        """Verify empty string is handled."""
        result = _strip_markdown_fences("")
        assert result == ""

    def test_only_opening_fence(self) -> None:
        """Verify text with only opening fence is handled."""
        text = '```json\n{"key": "value"}'
        result = _strip_markdown_fences(text)
        # Should remove opening but not closing (doesn't end with ```)
        assert result == '{"key": "value"}'


# =============================================================================
# Helper Function Tests: is_retryable_error
# =============================================================================


@pytest.mark.unit
class TestIsRetryableError:
    """Test is_retryable_error helper function."""

    @pytest.mark.parametrize(
        "exception_cls,exception_msg,expected",
        [
            (Exception, "Connection refused", True),
            (Exception, "Request timeout", True),
            (Exception, "Rate limit exceeded", True),
            (Exception, "Error 429", True),
            (Exception, "Error 503", True),
            (Exception, "Error 502", True),
            (Exception, "Error 500", True),
            (Exception, "Network error", True),
            (Exception, "Temporary failure", True),
            (Exception, "Connection reset", True),
            (Exception, "Timed out waiting for response", True),
        ],
        ids=[
            "connection_refused",
            "timeout",
            "rate_limit",
            "429",
            "503",
            "502",
            "500",
            "network",
            "temporary_failure",
            "connection_reset",
            "timed_out",
        ],
    )
    def test_retryable_error_messages(self, exception_cls: type[Exception], exception_msg: str, expected: bool) -> None:
        """Verify retryable error messages are detected."""
        exc = exception_cls(exception_msg)
        assert is_retryable_error(exc) == expected

    @pytest.mark.parametrize(
        "exception_cls",
        [
            ConnectionError,
            TimeoutError,
        ],
        ids=["ConnectionError", "TimeoutError"],
    )
    def test_retryable_exception_types(self, exception_cls: type[Exception]) -> None:
        """Verify retryable exception types are detected."""
        exc = exception_cls("Some error")
        assert is_retryable_error(exc) is True

    def test_non_retryable_error(self) -> None:
        """Verify non-retryable errors return False."""
        exc = ValueError("Invalid input")
        assert is_retryable_error(exc) is False

    def test_json_decode_error_not_retryable(self) -> None:
        """Verify JSONDecodeError is not retryable."""
        exc = json.JSONDecodeError("Invalid JSON", "", 0)
        assert is_retryable_error(exc) is False


# =============================================================================
# detect_sufficiency Function Tests
# =============================================================================


@pytest.fixture
def parsing_model_config() -> ModelConfig:
    """Return a minimal ModelConfig for testing."""
    return ModelConfig(
        id="test-parser",
        model_provider="anthropic",
        model_name="claude-haiku-4-5",
        temperature=0.0,
    )


@pytest.fixture
def sample_template_schema() -> dict:
    """Return a sample template schema for testing."""
    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the question",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score from 0 to 1",
            },
        },
        "required": ["answer"],
    }


@pytest.mark.unit
class TestDetectSufficiency:
    """Test detect_sufficiency function."""

    def test_sufficient_response_returns_true(
        self,
        parsing_model_config: ModelConfig,
        sample_template_schema: dict,
    ) -> None:
        """Verify sufficient response returns (True, True, reasoning, metadata)."""
        result = SufficiencyResult(
            reasoning="The response clearly states the answer is Paris.",
            sufficient=True,
        )
        mock_response = _create_mock_llm_response(content="", raw=result)

        with patch("karenina.benchmark.verification.evaluators.trace_sufficiency_checker.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.return_value = mock_response
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            sufficient, check_performed, reasoning, metadata = detect_sufficiency(
                raw_llm_response="The capital of France is Paris.",
                parsing_model=parsing_model_config,
                question_text="What is the capital of France?",
                template_schema=sample_template_schema,
            )

            assert sufficient is True
            assert check_performed is True
            assert reasoning is not None
            assert "Paris" in reasoning

    def test_insufficient_response_returns_false(
        self,
        parsing_model_config: ModelConfig,
        sample_template_schema: dict,
    ) -> None:
        """Verify insufficient response returns (False, True, reasoning, metadata)."""
        result = SufficiencyResult(
            reasoning="The response does not provide the requested answer.",
            sufficient=False,
        )
        mock_response = _create_mock_llm_response(content="", raw=result)

        with patch("karenina.benchmark.verification.evaluators.trace_sufficiency_checker.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.return_value = mock_response
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            sufficient, check_performed, reasoning, metadata = detect_sufficiency(
                raw_llm_response="I don't know.",
                parsing_model=parsing_model_config,
                question_text="What is the capital of France?",
                template_schema=sample_template_schema,
            )

            assert sufficient is False
            assert check_performed is True
            assert reasoning is not None

    def test_json_decode_error_defaults_to_sufficient(
        self,
        parsing_model_config: ModelConfig,
        sample_template_schema: dict,
    ) -> None:
        """Verify JSON parsing failure defaults to sufficient=True, check_performed=False."""
        # No raw result, and content is invalid JSON - triggers fallback parse failure
        mock_response = _create_mock_llm_response(content="Not valid JSON {{", raw=None)

        with patch("karenina.benchmark.verification.evaluators.trace_sufficiency_checker.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.return_value = mock_response
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            sufficient, check_performed, reasoning, metadata = detect_sufficiency(
                raw_llm_response="The answer is 42.",
                parsing_model=parsing_model_config,
                question_text="What is the meaning of life?",
                template_schema=sample_template_schema,
            )

            # Should default to sufficient on failure
            assert sufficient is True
            assert check_performed is False
            assert reasoning is None

    def test_missing_sufficient_key_defaults_to_true(
        self,
        parsing_model_config: ModelConfig,
        sample_template_schema: dict,
    ) -> None:
        """Verify missing 'sufficient' key defaults to True via fallback parsing."""
        # No raw result, content has valid JSON but missing 'sufficient' key
        content = json.dumps({"reasoning": "Some reasoning but forgot the key."})
        mock_response = _create_mock_llm_response(content=content, raw=None)

        with patch("karenina.benchmark.verification.evaluators.trace_sufficiency_checker.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.return_value = mock_response
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            sufficient, check_performed, reasoning, metadata = detect_sufficiency(
                raw_llm_response="The answer is Paris.",
                parsing_model=parsing_model_config,
                question_text="What is the capital of France?",
                template_schema=sample_template_schema,
            )

            # Missing key defaults to True via fallback parsing
            assert sufficient is True
            assert check_performed is True

    def test_handles_markdown_wrapped_response(
        self,
        parsing_model_config: ModelConfig,
        sample_template_schema: dict,
    ) -> None:
        """Verify markdown-wrapped JSON response is properly parsed via fallback."""
        # No raw result, content has markdown-wrapped JSON - triggers fallback parsing
        content = '```json\n{"reasoning": "Response is sufficient.", "sufficient": true}\n```'
        mock_response = _create_mock_llm_response(content=content, raw=None)

        with patch("karenina.benchmark.verification.evaluators.trace_sufficiency_checker.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.return_value = mock_response
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            sufficient, check_performed, reasoning, metadata = detect_sufficiency(
                raw_llm_response="The answer is 42.",
                parsing_model=parsing_model_config,
                question_text="What is X?",
                template_schema=sample_template_schema,
            )

            assert sufficient is True
            assert check_performed is True
            assert reasoning is not None

    def test_non_retryable_error_returns_sufficient_default(
        self,
        parsing_model_config: ModelConfig,
        sample_template_schema: dict,
    ) -> None:
        """Verify non-retryable errors default to sufficient=True."""
        with patch("karenina.benchmark.verification.evaluators.trace_sufficiency_checker.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.side_effect = ValueError("Invalid model config")
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            sufficient, check_performed, reasoning, metadata = detect_sufficiency(
                raw_llm_response="Some response.",
                parsing_model=parsing_model_config,
                question_text="What is X?",
                template_schema=sample_template_schema,
            )

            # Non-retryable error defaults to sufficient
            assert sufficient is True
            assert check_performed is False
            assert reasoning is None


@pytest.mark.unit
class TestDetectSufficiencyReturnSemantics:
    """Test the return value semantics of detect_sufficiency.

    Important semantic difference from abstention:
    - sufficient=True means response IS sufficient (good)
    - sufficient=False means response is INSUFFICIENT (bad, should fail)
    """

    def test_return_value_documentation(
        self,
        parsing_model_config: ModelConfig,
        sample_template_schema: dict,
    ) -> None:
        """Document and verify the return value semantics."""
        # When sufficient=True: response has info for all fields (good)
        result_true = SufficiencyResult(reasoning="OK", sufficient=True)
        mock_response_true = _create_mock_llm_response(content="", raw=result_true)

        with patch("karenina.benchmark.verification.evaluators.trace_sufficiency_checker.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.return_value = mock_response_true
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            sufficient, _, _, _ = detect_sufficiency(
                raw_llm_response="Paris",
                parsing_model=parsing_model_config,
                question_text="Capital?",
                template_schema=sample_template_schema,
            )

            # sufficient=True means we CAN populate the template
            assert sufficient is True

        # When sufficient=False: response lacks info (bad, should fail verification)
        result_false = SufficiencyResult(reasoning="Missing", sufficient=False)
        mock_response_false = _create_mock_llm_response(content="", raw=result_false)

        with patch("karenina.benchmark.verification.evaluators.trace_sufficiency_checker.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_structured_llm = MagicMock()
            mock_structured_llm.invoke.return_value = mock_response_false
            mock_llm.with_structured_output.return_value = mock_structured_llm
            mock_get_llm.return_value = mock_llm

            sufficient, _, _, _ = detect_sufficiency(
                raw_llm_response="I don't know",
                parsing_model=parsing_model_config,
                question_text="Capital?",
                template_schema=sample_template_schema,
            )

            # sufficient=False means we CANNOT populate the template
            assert sufficient is False
