"""Tests for template parsing helpers bug fixes.

Issue 098: _extract_attribute_descriptions uses regex to parse JSON schema.
Issue 102: format_excerpts_for_reasoning raises TypeError inconsistently.
"""

import json

import pytest

from karenina.benchmark.verification.utils.template_parsing_helpers import (
    _extract_attribute_descriptions,
    format_excerpts_for_reasoning,
)


@pytest.mark.unit
class TestExtractAttributeDescriptionsJsonParsing:
    """Issue 098: replace regex with json.loads + dict navigation."""

    def test_simple_schema_extracts_descriptions(self):
        """Basic schema with descriptions should work."""
        schema = json.dumps(
            {
                "properties": {
                    "drug_target": {
                        "description": "The target protein",
                        "type": "string",
                    },
                    "mechanism": {
                        "description": "Mechanism of action",
                        "type": "string",
                    },
                }
            }
        )
        result = _extract_attribute_descriptions(schema, ["drug_target", "mechanism"])
        assert result == {
            "drug_target": "The target protein",
            "mechanism": "Mechanism of action",
        }

    def test_description_with_escaped_quotes(self):
        """Descriptions containing escaped quotes should be extracted correctly.

        The old regex pattern [^}]* would fail when descriptions contain
        escaped quotes because the regex cannot distinguish escaped quotes
        from the closing brace boundary.
        """
        schema = json.dumps(
            {
                "properties": {
                    "answer": {
                        "description": 'The "correct" answer to the question',
                        "type": "string",
                    }
                }
            }
        )
        result = _extract_attribute_descriptions(schema, ["answer"])
        assert result == {"answer": 'The "correct" answer to the question'}

    def test_nested_objects_in_schema(self):
        """Schemas with nested objects should be parsed correctly.

        The old regex [^}]* fails on nested objects because the first }
        from the nested object terminates the match prematurely.
        """
        schema = json.dumps(
            {
                "properties": {
                    "result": {
                        "description": "The result value",
                        "type": "object",
                        "properties": {
                            "sub_field": {"type": "string"},
                        },
                    }
                }
            }
        )
        result = _extract_attribute_descriptions(schema, ["result"])
        assert result == {"result": "The result value"}

    def test_missing_attribute_returns_fallback(self):
        """Attributes not in schema should get fallback description."""
        schema = json.dumps({"properties": {}})
        result = _extract_attribute_descriptions(schema, ["missing_field"])
        assert result == {"missing_field": "Evidence for missing_field"}

    def test_invalid_json_schema_returns_fallbacks(self):
        """Invalid JSON schema string should produce fallback descriptions."""
        result = _extract_attribute_descriptions("not valid json", ["field_a"])
        assert result == {"field_a": "Evidence for field_a"}

    def test_attribute_without_description_key(self):
        """Attribute present but lacking a description key should get fallback."""
        schema = json.dumps(
            {
                "properties": {
                    "score": {
                        "type": "integer",
                    }
                }
            }
        )
        result = _extract_attribute_descriptions(schema, ["score"])
        assert result == {"score": "Evidence for score"}


@pytest.mark.unit
class TestFormatExcerptsForReasoningTypeError:
    """Issue 102: format_excerpts_for_reasoning should not raise TypeError."""

    def test_non_list_search_results_returns_string_not_raises(self):
        """When search_results is not a list, return a string with a warning instead of raising.

        Before the fix, passing a non-list search_results would raise TypeError.
        After the fix, it should log a warning and return a string.
        """
        excerpts = {
            "drug_target": [
                {
                    "text": "some excerpt",
                    "confidence": "high",
                    "similarity_score": 0.9,
                    "search_results": "this is not a list",
                }
            ]
        }
        # Should NOT raise TypeError
        result = format_excerpts_for_reasoning(excerpts)
        assert isinstance(result, str)
        assert "drug_target" in result

    def test_dict_search_results_returns_string_not_raises(self):
        """When search_results is a dict instead of list, should not raise."""
        excerpts = {
            "mechanism": [
                {
                    "text": "some text",
                    "confidence": "medium",
                    "similarity_score": 0.5,
                    "search_results": {"key": "value"},
                }
            ]
        }
        result = format_excerpts_for_reasoning(excerpts)
        assert isinstance(result, str)
        assert "mechanism" in result

    def test_valid_list_search_results_still_works(self):
        """Valid list search_results should continue to work."""
        excerpts = {
            "answer": [
                {
                    "text": "some text",
                    "confidence": "high",
                    "similarity_score": 0.95,
                    "search_results": [{"title": "Result 1", "content": "Content 1", "url": "http://example.com"}],
                }
            ]
        }
        result = format_excerpts_for_reasoning(excerpts)
        assert isinstance(result, str)
        assert "Result 1" in result

    def test_empty_excerpts_returns_default(self):
        """Empty dict should return default message (existing behavior)."""
        result = format_excerpts_for_reasoning({})
        assert result == "(No excerpts extracted)"
