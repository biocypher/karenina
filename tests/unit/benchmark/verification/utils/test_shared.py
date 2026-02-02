"""Unit tests for verification shared utility functions.

Tests the consolidated utilities:
- strip_markdown_fences(): Remove markdown code fences (karenina.utils.json_extraction)
- extract_json_from_text(): Extract JSON from mixed text (karenina.utils.json_extraction)
- extract_balanced_braces(): Extract balanced brace expressions (karenina.utils.json_extraction)
- is_retryable_error(): Check for transient errors (error_helpers)
- is_openai_endpoint_llm(): Check for OpenAI-compatible endpoints (llm_detection)
- parse_tool_output(): Parse search tool output (search_helpers)
"""

import pytest

from karenina.benchmark.verification.utils.error_helpers import is_retryable_error
from karenina.benchmark.verification.utils.llm_detection import is_openai_endpoint_llm
from karenina.benchmark.verification.utils.search_helpers import parse_tool_output
from karenina.schemas import SearchResultItem
from karenina.utils.json_extraction import (
    extract_balanced_braces,
    extract_json_from_text,
    strip_markdown_fences,
)

# =============================================================================
# strip_markdown_fences tests
# =============================================================================


@pytest.mark.unit
def test_strip_markdown_fences_basic() -> None:
    """Test basic markdown fence removal."""
    text = '```json\n{"field": "value"}\n```'
    result = strip_markdown_fences(text)
    assert result == '{"field": "value"}'


@pytest.mark.unit
def test_strip_markdown_fences_no_language_tag() -> None:
    """Test fence removal without language tag."""
    text = '```\n{"field": "value"}\n```'
    result = strip_markdown_fences(text)
    assert result == '{"field": "value"}'


@pytest.mark.unit
def test_strip_markdown_fences_with_reasoning() -> None:
    """Test extraction of JSON from text with reasoning."""
    text = 'Let me analyze this... the answer is {"field": "value"}'
    result = strip_markdown_fences(text)
    assert result == '{"field": "value"}'


@pytest.mark.unit
def test_strip_markdown_fences_none_input() -> None:
    """Test that None input returns None."""
    result = strip_markdown_fences(None)
    assert result is None


@pytest.mark.unit
def test_strip_markdown_fences_partial_opening() -> None:
    """Test partial fence removal - only opening fence."""
    text = '```json\n{"field": "value"}'
    result = strip_markdown_fences(text)
    assert result == '{"field": "value"}'


@pytest.mark.unit
def test_strip_markdown_fences_partial_closing() -> None:
    """Test partial fence removal - only closing fence."""
    text = '{"field": "value"}\n```'
    result = strip_markdown_fences(text)
    assert result == '{"field": "value"}'


@pytest.mark.unit
def test_strip_markdown_fences_plain_json() -> None:
    """Test that plain JSON without fences is returned as-is."""
    text = '{"field": "value"}'
    result = strip_markdown_fences(text)
    assert result == '{"field": "value"}'


# =============================================================================
# extract_json_from_text tests
# =============================================================================


@pytest.mark.unit
def test_extract_json_from_text_basic() -> None:
    """Test basic JSON extraction."""
    text = 'The answer is {"field": "value"} as shown.'
    result = extract_json_from_text(text)
    assert result == '{"field": "value"}'


@pytest.mark.unit
def test_extract_json_from_text_nested() -> None:
    """Test extraction of nested JSON objects."""
    text = 'Output: {"a": 1, "b": {"c": 2}}'
    result = extract_json_from_text(text)
    assert result == '{"a": 1, "b": {"c": 2}}'


@pytest.mark.unit
def test_extract_json_from_text_last_object() -> None:
    """Test that last valid JSON object is preferred."""
    text = '{"invalid: json"} and then {"valid": "json"}'
    result = extract_json_from_text(text)
    assert result == '{"valid": "json"}'


@pytest.mark.unit
def test_extract_json_from_text_no_json() -> None:
    """Test that None is returned when no JSON found."""
    text = "This is plain text without any JSON."
    result = extract_json_from_text(text)
    assert result is None


@pytest.mark.unit
def test_extract_json_from_text_invalid_json() -> None:
    """Test that invalid JSON is not returned."""
    text = "This has {invalid json}"
    result = extract_json_from_text(text)
    assert result is None


@pytest.mark.unit
def test_extract_json_from_text_braces_in_string() -> None:
    """Test that braces inside strings are handled correctly."""
    text = 'Result: {"text": "has { and } inside"}'
    result = extract_json_from_text(text)
    assert result == '{"text": "has { and } inside"}'


# =============================================================================
# extract_balanced_braces tests
# =============================================================================


@pytest.mark.unit
def test_extract_balanced_braces_simple() -> None:
    """Test simple balanced braces extraction."""
    text = '{"a": 1}'
    result = extract_balanced_braces(text, 0)
    assert result == '{"a": 1}'


@pytest.mark.unit
def test_extract_balanced_braces_nested() -> None:
    """Test nested braces extraction."""
    text = '{"outer": {"inner": 1}}'
    result = extract_balanced_braces(text, 0)
    assert result == '{"outer": {"inner": 1}}'


@pytest.mark.unit
def test_extract_balanced_braces_with_offset() -> None:
    """Test extraction starting at non-zero offset."""
    text = 'prefix {"a": 1} suffix'
    result = extract_balanced_braces(text, 7)
    assert result == '{"a": 1}'


@pytest.mark.unit
def test_extract_balanced_braces_unbalanced() -> None:
    """Test that unbalanced braces return None."""
    text = '{"a": 1'
    result = extract_balanced_braces(text, 0)
    assert result is None


@pytest.mark.unit
def test_extract_balanced_braces_not_starting_with_brace() -> None:
    """Test that non-brace start returns None."""
    text = 'abc{"a": 1}'
    result = extract_balanced_braces(text, 0)
    assert result is None


@pytest.mark.unit
def test_extract_balanced_braces_escaped_quotes() -> None:
    """Test handling of escaped quotes."""
    text = r'{"text": "say \"hello\""}'
    result = extract_balanced_braces(text, 0)
    assert result == r'{"text": "say \"hello\""}'


# =============================================================================
# is_retryable_error tests
# =============================================================================


@pytest.mark.unit
def test_is_retryable_error_connection_error() -> None:
    """Test that ConnectionError is retryable."""
    error = ConnectionError("Connection refused")
    assert is_retryable_error(error) is True


@pytest.mark.unit
def test_is_retryable_error_timeout() -> None:
    """Test that timeout errors are retryable."""
    error = TimeoutError("Request timed out")
    assert is_retryable_error(error) is True


@pytest.mark.unit
def test_is_retryable_error_rate_limit_message() -> None:
    """Test that rate limit errors (by message) are retryable."""
    error = Exception("Error 429: Rate limit exceeded")
    assert is_retryable_error(error) is True


@pytest.mark.unit
def test_is_retryable_error_500_error() -> None:
    """Test that 500 errors are retryable."""
    error = Exception("Server error 500")
    assert is_retryable_error(error) is True


@pytest.mark.unit
def test_is_retryable_error_503_error() -> None:
    """Test that 503 errors are retryable."""
    error = Exception("Service unavailable 503")
    assert is_retryable_error(error) is True


@pytest.mark.unit
def test_is_retryable_error_value_error() -> None:
    """Test that ValueError is not retryable."""
    error = ValueError("Invalid input")
    assert is_retryable_error(error) is False


@pytest.mark.unit
def test_is_retryable_error_key_error() -> None:
    """Test that KeyError is not retryable."""
    error = KeyError("missing key")
    assert is_retryable_error(error) is False


@pytest.mark.unit
def test_is_retryable_error_network_keyword() -> None:
    """Test that network-related errors are retryable."""
    error = Exception("Network unreachable")
    assert is_retryable_error(error) is True


# =============================================================================
# is_openai_endpoint_llm tests
# =============================================================================


# Define mock classes at module level for is_openai_endpoint_llm tests
class ChatOpenAIEndpoint:
    """Mock class with the expected endpoint class name."""

    pass


class ChatOpenAI:
    """Mock ChatOpenAI class from langchain_openai module."""

    pass


# Simulate a module path with "interface" in it
ChatOpenAI.__module__ = "langchain_openai"


class ChatOpenAIFromInterface(ChatOpenAI):
    """Mock ChatOpenAI class from an interface module."""

    pass


ChatOpenAIFromInterface.__module__ = "some.interface.module"


@pytest.mark.unit
def test_is_openai_endpoint_llm_by_class_name() -> None:
    """Test detection by class name."""
    mock_llm = ChatOpenAIEndpoint()
    assert is_openai_endpoint_llm(mock_llm) is True


@pytest.mark.unit
def test_is_openai_endpoint_llm_by_base_url() -> None:
    """Test detection by custom base URL."""

    class MockChatOpenAI:
        openai_api_base = "https://custom-api.example.com/v1"

    MockChatOpenAI.__module__ = "langchain_openai"
    mock_llm = MockChatOpenAI()
    assert is_openai_endpoint_llm(mock_llm) is True


@pytest.mark.unit
def test_is_openai_endpoint_llm_official_openai() -> None:
    """Test that official OpenAI API is not detected as endpoint."""

    class MockChatOpenAI:
        openai_api_base = "https://api.openai.com/v1"

    MockChatOpenAI.__module__ = "langchain_openai"
    mock_llm = MockChatOpenAI()
    assert is_openai_endpoint_llm(mock_llm) is False


@pytest.mark.unit
def test_is_openai_endpoint_llm_no_base_url() -> None:
    """Test LLM without base URL attribute."""

    class MockChatOpenAI:
        pass

    MockChatOpenAI.__module__ = "langchain_openai"
    mock_llm = MockChatOpenAI()
    assert is_openai_endpoint_llm(mock_llm) is False


# =============================================================================
# parse_tool_output tests
# =============================================================================


@pytest.mark.unit
def test_parse_tool_output_search_result_items() -> None:
    """Test parsing of already-structured SearchResultItem list."""
    items = [
        SearchResultItem(title="Test", content="Content", url="https://example.com"),
    ]
    result = parse_tool_output(items)
    assert result == items


@pytest.mark.unit
def test_parse_tool_output_dict_list() -> None:
    """Test parsing of list of dicts."""
    input_data = [
        {"title": "Title 1", "content": "Content 1", "url": "https://example1.com"},
        {"title": "Title 2", "content": "Content 2", "url": "https://example2.com"},
    ]
    result = parse_tool_output(input_data)
    assert len(result) == 2
    assert result[0].title == "Title 1"
    assert result[0].content == "Content 1"
    assert result[1].title == "Title 2"


@pytest.mark.unit
def test_parse_tool_output_json_string() -> None:
    """Test parsing of JSON string."""
    input_data = '[{"title": "Test", "content": "Content", "url": "https://example.com"}]'
    result = parse_tool_output(input_data)
    assert len(result) == 1
    assert result[0].title == "Test"
    assert result[0].content == "Content"


@pytest.mark.unit
def test_parse_tool_output_plain_text() -> None:
    """Test parsing of plain text (fallback)."""
    input_data = "This is plain text search result."
    result = parse_tool_output(input_data)
    assert len(result) == 1
    assert result[0].title is None
    assert result[0].content == "This is plain text search result."


@pytest.mark.unit
def test_parse_tool_output_empty_list() -> None:
    """Test parsing of empty list."""
    result = parse_tool_output([])
    assert result == []


@pytest.mark.unit
def test_parse_tool_output_unknown_type() -> None:
    """Test parsing of unknown type returns empty list."""
    result = parse_tool_output(12345)
    assert result == []


@pytest.mark.unit
def test_parse_tool_output_skips_empty_content() -> None:
    """Test that items with empty content are skipped."""
    input_data = [
        {"title": "Title 1", "content": "Good content"},
        {"title": "Title 2", "content": ""},  # Empty content
        {"title": "Title 3", "content": "No content"},  # Default "No content"
    ]
    result = parse_tool_output(input_data)
    assert len(result) == 1
    assert result[0].title == "Title 1"
