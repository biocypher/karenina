"""Tests for JSON extraction and parsing utilities."""

import json

import pytest

from karenina.utils.json_extraction import (
    extract_balanced_braces,
    extract_json_from_response,
    extract_json_from_text,
    is_invalid_json_error,
    strip_markdown_fences,
)


@pytest.mark.unit
class TestStripMarkdownFences:
    """Tests for strip_markdown_fences."""

    def test_removes_json_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_removes_plain_fence(self):
        text = '```\n{"key": "value"}\n```'
        assert strip_markdown_fences(text) == '{"key": "value"}'

    def test_removes_fence_with_language_tag(self):
        text = '```python\nprint("hello")\n```'
        assert strip_markdown_fences(text) == 'print("hello")'

    def test_extracts_json_from_mixed_text(self):
        text = 'The answer is {"field": "value"} as shown.'
        result = strip_markdown_fences(text)
        assert result == '{"field": "value"}'

    def test_returns_none_for_none_input(self):
        assert strip_markdown_fences(None) is None

    def test_returns_non_string_as_is(self):
        assert strip_markdown_fences(42) == 42

    def test_handles_partial_opening_fence(self):
        text = '```json\n{"key": "value"}'
        result = strip_markdown_fences(text)
        assert '"key"' in result
        assert "```" not in result

    def test_handles_partial_closing_fence(self):
        text = '{"key": "value"}\n```'
        result = strip_markdown_fences(text)
        assert "```" not in result

    def test_strips_whitespace(self):
        text = '```json\n  {"key": "value"}  \n```'
        result = strip_markdown_fences(text)
        assert result == '{"key": "value"}'


@pytest.mark.unit
class TestExtractJsonFromText:
    """Tests for extract_json_from_text."""

    def test_extracts_simple_json(self):
        text = 'The answer is {"field": "value"} done.'
        result = extract_json_from_text(text)
        assert result == '{"field": "value"}'

    def test_extracts_nested_json(self):
        text = 'Output: {"a": 1, "b": {"c": 2}}'
        result = extract_json_from_text(text)
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": {"c": 2}}

    def test_returns_none_for_no_json(self):
        assert extract_json_from_text("no json here") is None

    def test_prefers_last_json_object(self):
        text = '{"first": 1} some text {"second": 2}'
        result = extract_json_from_text(text)
        parsed = json.loads(result)
        assert parsed == {"second": 2}

    def test_skips_invalid_json_candidates(self):
        text = '{invalid json} and {"valid": true}'
        result = extract_json_from_text(text)
        parsed = json.loads(result)
        assert parsed == {"valid": True}

    def test_returns_none_for_empty_string(self):
        assert extract_json_from_text("") is None


@pytest.mark.unit
class TestExtractBalancedBraces:
    """Tests for extract_balanced_braces."""

    def test_simple_object(self):
        text = '{"key": "value"}'
        assert extract_balanced_braces(text, 0) == '{"key": "value"}'

    def test_nested_braces(self):
        text = '{"a": {"b": 1}}'
        assert extract_balanced_braces(text, 0) == '{"a": {"b": 1}}'

    def test_braces_in_strings(self):
        text = '{"text": "has { and }"}'
        result = extract_balanced_braces(text, 0)
        assert result == '{"text": "has { and }"}'

    def test_escaped_quotes(self):
        text = '{"text": "say \\"hello\\""}'
        result = extract_balanced_braces(text, 0)
        assert result is not None
        assert result.startswith("{")
        assert result.endswith("}")

    def test_returns_none_for_non_brace_start(self):
        assert extract_balanced_braces("abc", 0) is None

    def test_returns_none_for_out_of_range(self):
        assert extract_balanced_braces("abc", 10) is None

    def test_returns_none_for_unbalanced(self):
        assert extract_balanced_braces("{unclosed", 0) is None

    def test_starts_at_offset(self):
        text = 'prefix {"key": "value"} suffix'
        assert extract_balanced_braces(text, 7) == '{"key": "value"}'


@pytest.mark.unit
class TestExtractJsonFromResponse:
    """Tests for extract_json_from_response."""

    def test_returns_direct_json_object(self):
        text = '{"key": "value"}'
        assert extract_json_from_response(text) == '{"key": "value"}'

    def test_returns_direct_json_array(self):
        text = "[1, 2, 3]"
        assert extract_json_from_response(text) == "[1, 2, 3]"

    def test_extracts_from_markdown_block(self):
        text = '```json\n{"key": "value"}\n```'
        assert extract_json_from_response(text) == '{"key": "value"}'

    def test_extracts_from_mixed_text(self):
        text = 'Here is the result: {"key": "value"}'
        result = extract_json_from_response(text)
        assert result == '{"key": "value"}'

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError, match="Could not extract JSON"):
            extract_json_from_response("no json here at all")

    def test_strips_whitespace(self):
        text = '  {"key": "value"}  '
        assert extract_json_from_response(text) == '{"key": "value"}'


@pytest.mark.unit
class TestIsInvalidJsonError:
    """Tests for is_invalid_json_error."""

    def test_detects_json_decode_error(self):
        try:
            json.loads("not json")
        except json.JSONDecodeError as e:
            assert is_invalid_json_error(e) is True

    def test_detects_invalid_json_message(self):
        error = ValueError("invalid json in response")
        assert is_invalid_json_error(error) is True

    def test_detects_expecting_value(self):
        error = ValueError("Expecting value: line 1")
        assert is_invalid_json_error(error) is True

    def test_returns_false_for_unrelated_error(self):
        error = RuntimeError("connection timeout")
        assert is_invalid_json_error(error) is False

    def test_returns_false_for_generic_value_error(self):
        error = ValueError("something completely different")
        assert is_invalid_json_error(error) is False
