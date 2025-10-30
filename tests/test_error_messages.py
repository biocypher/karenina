"""Test improved error messages in manual trace functionality."""

import pytest

from karenina.infrastructure.llm.manual_llm import ManualLLM, ManualTraceNotFoundError
from karenina.infrastructure.llm.manual_traces import (
    ManualTraceError,
    ManualTraceManager,
    clear_manual_traces,
    load_manual_traces,
)


def test_manual_trace_manager_helpful_error_messages() -> None:
    """Test that ManualTraceManager provides helpful error messages."""
    manager = ManualTraceManager()

    # Test non-dict data error
    with pytest.raises(ManualTraceError) as exc_info:
        manager.load_traces_from_json("not a dict")

    error_msg = str(exc_info.value)
    assert "Invalid trace data format" in error_msg
    assert "Expected a JSON object" in error_msg
    assert "Received str" in error_msg
    assert '{"hash1": "trace1", "hash2": "trace2"}' in error_msg

    # Test empty data error
    with pytest.raises(ManualTraceError) as exc_info:
        manager.load_traces_from_json({})

    error_msg = str(exc_info.value)
    assert "Empty trace data" in error_msg
    assert "No traces found" in error_msg
    assert "d41d8cd98f00b204e9800998ecf8427e" in error_msg

    # Test invalid hash format error
    with pytest.raises(ManualTraceError) as exc_info:
        manager.load_traces_from_json({"invalid-hash": "some trace"})

    error_msg = str(exc_info.value)
    assert "Invalid question hash format" in error_msg
    assert "invalid-hash" in error_msg
    assert "32-character hexadecimal MD5 hashes" in error_msg
    assert "question extraction" in error_msg
    assert "d41d8cd98f00b204e9800998ecf8427e" in error_msg
    assert "Download CSV mapper" in error_msg

    # Test invalid trace content error (non-string)
    with pytest.raises(ManualTraceError) as exc_info:
        manager.load_traces_from_json({"d41d8cd98f00b204e9800998ecf8427e": 123})

    error_msg = str(exc_info.value)
    assert "Invalid trace content" in error_msg
    assert "d41d8cd98f00b204e9800998ecf8427e" in error_msg
    assert "Expected a non-empty string" in error_msg
    assert "got int" in error_msg
    assert "precomputed answer text" in error_msg

    # Test invalid trace content error (empty string)
    with pytest.raises(ManualTraceError) as exc_info:
        manager.load_traces_from_json({"d41d8cd98f00b204e9800998ecf8427e": ""})

    error_msg = str(exc_info.value)
    assert "Invalid trace content" in error_msg
    assert "non-empty string" in error_msg


def test_manual_llm_helpful_error_messages() -> None:
    """Test that ManualLLM provides helpful error messages when traces not found."""
    # Clear any existing traces
    clear_manual_traces()

    # Test with no traces loaded
    manual_llm = ManualLLM("d41d8cd98f00b204e9800998ecf8427e")

    with pytest.raises(ManualTraceNotFoundError) as exc_info:
        manual_llm.invoke([])

    error_msg = str(exc_info.value)
    assert "No manual trace found" in error_msg
    assert "d41d8cd98f00b204e9800998ecf8427e" in error_msg
    assert "Currently loaded 0 trace(s)" in error_msg
    assert "Upload a JSON file" in error_msg
    assert "load_manual_traces()" in error_msg
    assert "Verify that the question hash matches" in error_msg

    # Test with some traces loaded but not the one we're looking for
    load_manual_traces({"c4ca4238a0b923820dcc509a6f75849b": "Trace 1", "c81e728d9d4c2f636f067f89cc14862c": "Trace 2"})

    with pytest.raises(ManualTraceNotFoundError) as exc_info:
        manual_llm.invoke([])

    error_msg = str(exc_info.value)
    assert "No manual trace found" in error_msg
    assert "d41d8cd98f00b204e9800998ecf8427e" in error_msg
    assert "Currently loaded 2 trace(s)" in error_msg
    assert "Upload a JSON file" in error_msg

    # Test content property as well
    with pytest.raises(ManualTraceNotFoundError) as exc_info:
        _ = manual_llm.content

    error_msg = str(exc_info.value)
    assert "No manual trace found" in error_msg
    assert "Currently loaded 2 trace(s)" in error_msg

    # Cleanup
    clear_manual_traces()


def test_error_message_consistency() -> None:
    """Test that error messages are consistent between different methods."""
    clear_manual_traces()

    manual_llm = ManualLLM("d41d8cd98f00b204e9800998ecf8427e")

    # Get error messages from both methods
    invoke_error = None
    content_error = None

    try:
        manual_llm.invoke([])
    except ManualTraceNotFoundError as e:
        invoke_error = str(e)

    try:
        _ = manual_llm.content
    except ManualTraceNotFoundError as e:
        content_error = str(e)

    # Both should have similar structure and helpfulness
    assert invoke_error is not None
    assert content_error is not None
    assert "No manual trace found" in invoke_error
    assert "No manual trace found" in content_error
    assert "Currently loaded 0 trace(s)" in invoke_error
    assert "Currently loaded 0 trace(s)" in content_error
    assert "Upload a JSON file" in invoke_error
    assert "Upload a JSON file" in content_error


def test_error_messages_include_actionable_guidance() -> None:
    """Test that error messages include actionable guidance for users."""
    manager = ManualTraceManager()

    # Test that hash validation error includes CSV mapper reference
    with pytest.raises(ManualTraceError) as exc_info:
        manager.load_traces_from_json({"bad-hash": "trace"})

    error_msg = str(exc_info.value)
    assert "Download CSV mapper" in error_msg
    assert "question extraction" in error_msg
    assert "32-character hexadecimal" in error_msg

    # Test that missing trace error includes multiple resolution options
    clear_manual_traces()
    manual_llm = ManualLLM("missing_hash_123456789012345678")

    with pytest.raises(ManualTraceNotFoundError) as exc_info:
        manual_llm.invoke([])

    error_msg = str(exc_info.value)
    resolution_options = ["Upload a JSON file", "load_manual_traces()", "Verify that the question hash matches"]

    for option in resolution_options:
        assert option in error_msg, f"Missing resolution option: {option}"

    # Test that trace count is included for context
    assert "Currently loaded" in error_msg
    assert "trace(s)" in error_msg


def test_error_messages_include_examples() -> None:
    """Test that error messages include helpful examples."""
    manager = ManualTraceManager()

    # Test format examples in various error messages
    test_cases = [
        # Empty data
        ({}, ["d41d8cd98f00b204e9800998ecf8427e", "Your answer trace here"]),
        # Invalid hash
        ({"bad": "trace"}, ["d41d8cd98f00b204e9800998ecf8427e"]),
        # Invalid trace content (number)
        ({"d41d8cd98f00b204e9800998ecf8427e": 123}, ["This is the answer"]),
    ]

    for invalid_data, expected_examples in test_cases:
        with pytest.raises(ManualTraceError) as exc_info:
            manager.load_traces_from_json(invalid_data)

        error_msg = str(exc_info.value)
        for example in expected_examples:
            assert example in error_msg, f"Missing example '{example}' in error: {error_msg}"


if __name__ == "__main__":
    pytest.main([__file__])
