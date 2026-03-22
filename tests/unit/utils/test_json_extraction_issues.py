"""Tests for JSON extraction bug fixes.

Issue 081: extract_json_from_response skips validation when input starts with { or [.
"""

import pytest

from karenina.utils.json_extraction import extract_json_from_response


@pytest.mark.unit
class TestExtractJsonFromResponseValidation:
    """Issue 081: early return on brace-starting input must validate with json.loads."""

    def test_valid_json_starting_with_brace_returns_as_is(self):
        """Valid JSON starting with { should still be returned."""
        result = extract_json_from_response('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_valid_json_array_starting_with_bracket_returns_as_is(self):
        """Valid JSON starting with [ should still be returned."""
        result = extract_json_from_response("[1, 2, 3]")
        assert result == "[1, 2, 3]"

    def test_invalid_json_starting_with_brace_falls_through(self):
        """Invalid JSON starting with { should not be returned as-is.

        Before the fix, extract_json_from_response would return the raw
        text without validation when it starts with { or [. After the fix,
        it should attempt extraction or raise ValueError.
        """
        # This is invalid JSON: truncated object
        invalid = '{"key": "value", "nested": {"broken'
        with pytest.raises(ValueError, match="Could not extract JSON"):
            extract_json_from_response(invalid)

    def test_invalid_json_starting_with_bracket_falls_through(self):
        """Invalid JSON starting with [ should not be returned as-is."""
        invalid = '[1, 2, "unterminated'
        with pytest.raises(ValueError, match="Could not extract JSON"):
            extract_json_from_response(invalid)

    def test_brace_start_with_trailing_text_extracts_json(self):
        """Input starting with { but containing trailing text should extract the JSON portion."""
        text = '{"key": "value"} and some extra text here'
        result = extract_json_from_response(text)
        assert result == '{"key": "value"}'
