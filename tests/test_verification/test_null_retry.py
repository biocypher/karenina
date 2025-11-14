"""Tests for null-value retry functionality in verification_utils."""

from unittest.mock import Mock, patch

from karenina.benchmark.verification.utils.usage_tracker import UsageTracker
from karenina.benchmark.verification.verification_utils import (
    _extract_null_fields_from_error,
    _retry_parse_with_null_feedback,
)


class TestExtractNullFieldsFromError:
    """Test suite for _extract_null_fields_from_error."""

    def test_extract_from_json_with_null_values(self) -> None:
        """Test extraction of null fields from JSON string."""
        failed_json = '{"id": null, "value": 42, "name": null}'
        error_str = "Validation error: id is required"

        null_fields = _extract_null_fields_from_error(error_str, failed_json)

        assert "id" in null_fields
        assert "name" in null_fields
        assert "value" not in null_fields

    def test_extract_from_pydantic_error_text(self) -> None:
        """Test extraction from Pydantic error message format."""
        error_str = """1 validation error for Answer
african_american_population_allele_frequency
  Input should be a valid number [type=float_type, input_value=None, input_type=NoneType]"""

        null_fields = _extract_null_fields_from_error(error_str, None)

        assert "african_american_population_allele_frequency" in null_fields

    def test_extract_multiple_fields_from_error(self) -> None:
        """Test extraction of multiple null fields from error text."""
        error_str = """3 validation errors for Answer
field_one
  Input should be a valid number [type=float_type, input_value=None, input_type=NoneType]
field_two
  Input should be a valid string [type=string_type, input_value=None, input_type=NoneType]
field_three
  Input should be a valid boolean [type=bool_type, input_value=None, input_type=NoneType]"""

        null_fields = _extract_null_fields_from_error(error_str, None)

        assert "field_one" in null_fields
        assert "field_two" in null_fields
        assert "field_three" in null_fields
        assert len(null_fields) == 3

    def test_no_null_fields_detected(self) -> None:
        """Test when error is not related to null values."""
        error_str = """1 validation error for Answer
value
  Input should be less than 100 [type=less_than, input_value=150]"""

        null_fields = _extract_null_fields_from_error(error_str, None)

        assert len(null_fields) == 0

    def test_invalid_json_fallback_to_error_parsing(self) -> None:
        """Test fallback to error text parsing when JSON is invalid."""
        failed_json = "Not valid JSON at all"
        error_str = """1 validation error for Answer
test_field
  Input should be a valid number [type=float_type, input_value=None, input_type=NoneType]"""

        null_fields = _extract_null_fields_from_error(error_str, failed_json)

        assert "test_field" in null_fields

    def test_filters_out_common_keywords(self) -> None:
        """Test that common error keywords are not extracted as field names."""
        error_str = """1 validation error for Answer
test_value
  Input should be a valid number [type=float_type, input_value=None, input_type=NoneType]"""

        null_fields = _extract_null_fields_from_error(error_str, None)

        # Should extract the actual field name, not keywords
        assert "test_value" in null_fields
        assert "Answer" not in null_fields
        assert "Input" not in null_fields
        assert "For" not in null_fields

    def test_empty_error_string(self) -> None:
        """Test handling of empty error string."""
        null_fields = _extract_null_fields_from_error("", None)
        assert len(null_fields) == 0

    def test_deduplication_of_field_names(self) -> None:
        """Test that duplicate field names are deduplicated."""
        failed_json = '{"field": null}'
        error_str = """2 validation errors for Answer
field
  Input should be a valid number [type=float_type, input_value=None, input_type=NoneType]
field
  Input should be a valid number [type=float_type, input_value=None, input_type=NoneType]"""

        null_fields = _extract_null_fields_from_error(error_str, failed_json)

        # Should only appear once
        assert null_fields.count("field") == 1


class TestRetryParseWithNullFeedback:
    """Test suite for _retry_parse_with_null_feedback."""

    @patch("karenina.benchmark.verification.utils.parsing._strip_markdown_fences")
    @patch("karenina.benchmark.verification.verification_utils._invoke_llm_with_retry")
    def test_successful_retry_after_null_feedback(
        self,
        mock_invoke_llm: Mock,
        mock_strip_fences: Mock,
    ) -> None:
        """Test successful parsing after retry with null-value feedback."""
        # Mock the parser
        mock_parser = Mock()
        parsed_result = Mock()
        mock_parser.parse.return_value = parsed_result

        # Mock LLM invocation
        mock_invoke_llm.return_value = (
            '{"id": "test-123", "value": 0.0}',  # raw_response
            [],  # messages
            {"input_tokens": 100, "output_tokens": 50},  # usage_metadata
            None,  # agent_response
        )

        # Mock markdown fence stripping
        mock_strip_fences.return_value = '{"id": "test-123", "value": 0.0}'

        # Create mocks
        mock_llm = Mock()
        original_messages = [Mock()]
        failed_response = '{"id": null, "value": null}'
        error = ValueError(
            "1 validation error for Answer\nvalue\n  Input should be a valid number [type=float_type, input_value=None, input_type=NoneType]"
        )

        # Execute retry
        result, usage = _retry_parse_with_null_feedback(
            parsing_llm=mock_llm,
            parser=mock_parser,
            original_messages=original_messages,
            failed_response=failed_response,
            error=error,
        )

        # Verify retry was successful
        assert result == parsed_result
        assert usage == {"input_tokens": 100, "output_tokens": 50}
        mock_invoke_llm.assert_called_once()
        mock_parser.parse.assert_called_once()

    @patch("karenina.benchmark.verification.verification_utils._invoke_llm_with_retry")
    def test_retry_fails_returns_none(self, mock_invoke_llm: Mock) -> None:
        """Test that retry returns None when parsing still fails."""
        # Mock the parser to fail on retry
        mock_parser = Mock()
        mock_parser.parse.side_effect = Exception("Still cannot parse")

        # Mock LLM invocation
        mock_invoke_llm.return_value = (
            '{"id": "test", "value": "invalid"}',
            [],
            {},
            None,
        )

        # Create mocks
        mock_llm = Mock()
        original_messages = [Mock()]
        failed_response = '{"id": null, "value": null}'
        error = Exception("Parsing failed: null values")

        # Execute retry
        result, usage = _retry_parse_with_null_feedback(
            parsing_llm=mock_llm,
            parser=mock_parser,
            original_messages=original_messages,
            failed_response=failed_response,
            error=error,
        )

        # Verify retry returned None on failure
        assert result is None
        assert usage == {}

    def test_skips_retry_for_non_null_errors(self) -> None:
        """Test that retry is skipped when error is not null-related."""
        mock_parser = Mock()
        mock_llm = Mock()
        original_messages = [Mock()]
        failed_response = '{"id": "test", "value": 150}'
        error = Exception("Value must be less than 100")

        # Execute retry
        result, usage = _retry_parse_with_null_feedback(
            parsing_llm=mock_llm,
            parser=mock_parser,
            original_messages=original_messages,
            failed_response=failed_response,
            error=error,
        )

        # Verify retry was skipped
        assert result is None
        assert usage == {}
        mock_llm.invoke.assert_not_called()

    @patch("karenina.benchmark.verification.utils.parsing._strip_markdown_fences")
    @patch("karenina.benchmark.verification.verification_utils._invoke_llm_with_retry")
    def test_tracks_usage_when_tracker_provided(
        self,
        mock_invoke_llm: Mock,
        mock_strip_fences: Mock,
    ) -> None:
        """Test that usage is tracked when usage_tracker is provided."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser.parse.return_value = Mock()

        # Mock format: (raw_response, messages, usage_metadata, agent_response)
        mock_invoke_llm.return_value = (
            '{"id": "test", "value": 0.0}',
            [],
            {"test-model": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}},
            None,
        )

        mock_strip_fences.return_value = '{"id": "test", "value": 0.0}'

        # Create usage tracker
        usage_tracker = UsageTracker()

        # Execute retry with tracker
        error = ValueError(
            "1 validation error for Answer\nvalue\n  Input should be a valid number [type=float_type, input_value=None, input_type=NoneType]"
        )

        _retry_parse_with_null_feedback(
            parsing_llm=Mock(),
            parser=mock_parser,
            original_messages=[Mock()],
            failed_response='{"id": null, "value": null}',
            error=error,
            usage_tracker=usage_tracker,
            model_str="test-model",
        )

        # Verify usage was tracked
        summary = usage_tracker.get_stage_summary("parsing_null_retry")
        assert summary is not None
        assert summary["input_tokens"] == 100
        assert summary["output_tokens"] == 50
        assert summary["total_tokens"] == 150
        assert summary["model"] == "test-model"

    @patch("karenina.benchmark.verification.utils.parsing._strip_markdown_fences")
    @patch("karenina.benchmark.verification.verification_utils._invoke_llm_with_retry")
    def test_feedback_message_includes_field_names(
        self,
        mock_invoke_llm: Mock,
        mock_strip_fences: Mock,
    ) -> None:
        """Test that feedback message includes the null field names."""
        # Setup mocks
        mock_parser = Mock()
        mock_parser.parse.return_value = Mock()

        mock_strip_fences.return_value = '{"id": "test", "value": 0.0}'

        def capture_messages(*args, **kwargs):
            # Capture the messages sent to LLM
            messages = args[1] if len(args) > 1 else kwargs.get("messages", [])
            # Check that feedback mentions the null fields
            feedback_content = messages[-1].content
            assert "id" in feedback_content or "value" in feedback_content
            assert "null" in feedback_content.lower()
            return ('{"id": "test", "value": 0.0}', messages, {}, None)

        mock_invoke_llm.side_effect = capture_messages

        # Execute retry
        error = ValueError(
            "1 validation error for Answer\nvalue\n  Input should be a valid number [type=float_type, input_value=None, input_type=NoneType]"
        )

        _retry_parse_with_null_feedback(
            parsing_llm=Mock(),
            parser=mock_parser,
            original_messages=[Mock()],
            failed_response='{"id": null, "value": null}',
            error=error,
        )

        # Assert is done in the capture_messages function
        mock_invoke_llm.assert_called_once()

    @patch("karenina.benchmark.verification.utils.parsing._strip_markdown_fences")
    @patch("karenina.benchmark.verification.verification_utils._invoke_llm_with_retry")
    def test_handles_multiple_null_fields(
        self,
        mock_invoke_llm: Mock,
        mock_strip_fences: Mock,
    ) -> None:
        """Test retry with multiple null fields in error."""
        mock_parser = Mock()
        mock_parser.parse.return_value = Mock()

        # Mock format with proper usage metadata structure
        mock_invoke_llm.return_value = (
            '{"field1": 0.0, "field2": "", "field3": false}',
            [],
            {"gpt-4.1-mini": {"input_tokens": 50, "output_tokens": 30, "total_tokens": 80}},
            None,
        )

        mock_strip_fences.return_value = '{"field1": 0.0, "field2": "", "field3": false}'

        # Create error with multiple null fields (proper Pydantic format)
        failed_response = '{"field1": null, "field2": null, "field3": null}'
        error_str = """3 validation errors for Answer
field1
  Input should be a valid number [type=float_type, input_value=None, input_type=NoneType]
field2
  Input should be a valid string [type=str_type, input_value=None, input_type=NoneType]
field3
  Input should be a valid boolean [type=bool_type, input_value=None, input_type=NoneType]"""
        error = Exception(error_str)

        result, _ = _retry_parse_with_null_feedback(
            parsing_llm=Mock(),
            parser=mock_parser,
            original_messages=[Mock()],
            failed_response=failed_response,
            error=error,
        )

        # Verify retry succeeded
        assert result is not None
        mock_parser.parse.assert_called_once()
