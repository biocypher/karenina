"""Unit tests for LLM client utilities and infrastructure.

Tests cover:
- Exception classes (LLMError, LLMNotAvailableError, SessionError)
- ManualTraceManager class (trace storage, validation, cleanup)
- Manual trace utilities and helper functions
- ManualTraces class
- init_chat_model_unified function

Note: Tests for ChatOpenRouter and ChatOpenAIEndpoint are in test_langchain_adapter.py
since those classes live in karenina.adapters.langchain.models.

Note: Tests do NOT make actual API calls. All LLM interaction is mocked or uses fixture-backed implementations.

Note: ManualLLM class has been removed as dead code. Manual interface now uses
ManualAgentAdapter which reads traces directly from ManualTraceManager.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from karenina.adapters.langchain.initialization import init_chat_model_unified
from karenina.adapters.manual import (
    ManualTraceError,
    ManualTraceManager,
    ManualTraces,
    clear_manual_traces,
    convert_langchain_messages,
    extract_agent_metrics,
    get_manual_trace,
    get_manual_trace_count,
    get_manual_trace_with_metrics,
    get_memory_usage_info,
    harmonize_messages,
    has_manual_trace,
    is_langchain_message_list,
    is_port_message_list,
    load_manual_traces,
    preprocess_message_list,
    set_manual_trace,
)
from karenina.infrastructure.llm.exceptions import (
    LLMError,
    LLMNotAvailableError,
    SessionError,
)
from karenina.ports.messages import Message, Role, ToolUseContent

# =============================================================================
# Exception Classes Tests
# =============================================================================


@pytest.mark.unit
def test_llm_error_is_exception() -> None:
    """Test that LLMError is an Exception subclass."""
    assert issubclass(LLMError, Exception)
    error = LLMError("test error")
    assert str(error) == "test error"


@pytest.mark.unit
def test_llm_not_available_error_is_llm_error() -> None:
    """Test that LLMNotAvailableError inherits from LLMError."""
    assert issubclass(LLMNotAvailableError, LLMError)
    error = LLMNotAvailableError("not available")
    assert isinstance(error, LLMError)


@pytest.mark.unit
def test_session_error_is_llm_error() -> None:
    """Test that SessionError inherits from LLMError."""
    assert issubclass(SessionError, LLMError)
    error = SessionError("session error")
    assert isinstance(error, LLMError)


@pytest.mark.unit
def test_manual_trace_error_is_exception() -> None:
    """Test that ManualTraceError is an Exception subclass."""
    assert issubclass(ManualTraceError, Exception)
    error = ManualTraceError("trace error")
    assert str(error) == "trace error"


# =============================================================================
# ManualTraceManager Class Tests
# =============================================================================


@pytest.mark.unit
def test_manual_trace_manager_initialization() -> None:
    """Test ManualTraceManager initialization."""
    manager = ManualTraceManager()

    assert manager.get_trace_count() == 0
    assert manager.get_all_traces() == {}


@pytest.mark.unit
def test_manual_trace_manager_set_and_get_trace() -> None:
    """Test setting and getting a trace."""
    manager = ManualTraceManager()
    valid_hash = "d41d8cd98f00b204e9800998ecf8427e"

    manager.set_trace(valid_hash, "Test trace")

    assert manager.get_trace(valid_hash) == "Test trace"
    assert manager.has_trace(valid_hash) is True


@pytest.mark.unit
def test_manual_trace_manager_get_trace_not_found() -> None:
    """Test getting a non-existent trace returns None."""
    manager = ManualTraceManager()

    assert manager.get_trace("d41d8cd98f00b204e9800998ecf8427e") is None


@pytest.mark.unit
def test_manual_trace_manager_has_trace() -> None:
    """Test has_trace method."""
    manager = ManualTraceManager()
    valid_hash = "d41d8cd98f00b204e9800998ecf8427e"

    assert manager.has_trace(valid_hash) is False

    manager.set_trace(valid_hash, "trace")

    assert manager.has_trace(valid_hash) is True


@pytest.mark.unit
def test_manual_trace_manager_set_trace_with_agent_metrics() -> None:
    """Test setting trace with agent metrics."""
    manager = ManualTraceManager()
    valid_hash = "d41d8cd98f00b204e9800998ecf8427e"
    metrics = {"tool_calls": 3, "failures": 1}

    manager.set_trace(valid_hash, "Trace", agent_metrics=metrics)

    trace, retrieved_metrics = manager.get_trace_with_metrics(valid_hash)
    assert trace == "Trace"
    assert retrieved_metrics == metrics


@pytest.mark.unit
def test_manual_trace_manager_invalid_hash_raises_error() -> None:
    """Test that invalid hash format raises ManualTraceError."""
    manager = ManualTraceManager()

    with pytest.raises(ManualTraceError, match="Invalid question hash format"):
        manager.set_trace("invalid", "trace")


@pytest.mark.unit
def test_manual_trace_manager_get_all_traces() -> None:
    """Test getting all traces."""
    manager = ManualTraceManager()
    hash1 = "d41d8cd98f00b204e9800998ecf8427e"
    hash2 = "5d41402abc4b2a76b9719d911017c592"

    manager.set_trace(hash1, "trace1")
    manager.set_trace(hash2, "trace2")

    all_traces = manager.get_all_traces()

    assert all_traces == {hash1: "trace1", hash2: "trace2"}


@pytest.mark.unit
def test_manual_trace_manager_get_all_traces_returns_copy() -> None:
    """Test that get_all_traces returns a copy, not the internal dict."""
    manager = ManualTraceManager()
    hash1 = "d41d8cd98f00b204e9800998ecf8427e"
    hash2 = "5d41402abc4b2a76b9719d911017c592"
    manager.set_trace(hash1, "trace1")

    all_traces = manager.get_all_traces()
    all_traces[hash2] = "trace2"  # Modify the returned dict

    # Original should be unchanged
    assert manager.get_trace(hash2) is None


@pytest.mark.unit
def test_manual_trace_manager_clear_traces() -> None:
    """Test clearing all traces."""
    manager = ManualTraceManager()
    hash1 = "d41d8cd98f00b204e9800998ecf8427e"
    hash2 = "5d41402abc4b2a76b9719d911017c592"
    manager.set_trace(hash1, "trace1")
    manager.set_trace(hash2, "trace2")

    manager.clear_traces()

    assert manager.get_trace_count() == 0
    assert manager.get_all_traces() == {}


@pytest.mark.unit
def test_manual_trace_manager_get_memory_usage_info() -> None:
    """Test getting memory usage information."""
    manager = ManualTraceManager()
    hash1 = "d41d8cd98f00b204e9800998ecf8427e"
    hash2 = "5d41402abc4b2a76b9719d911017c592"

    manager.set_trace(hash1, "A" * 100)
    manager.set_trace(hash2, "B" * 50)

    info = manager.get_memory_usage_info()

    assert info["trace_count"] == 2
    assert info["total_characters"] == 150
    assert info["estimated_memory_bytes"] == 600  # 150 * 4
    assert "session_timeout_seconds" in info
    assert "last_access_timestamp" in info
    assert "seconds_since_last_access" in info


# =============================================================================
# Manual Trace Validation Tests
# =============================================================================


@pytest.mark.unit
def test_validate_trace_data_valid() -> None:
    """Test validation of valid trace data."""
    manager = ManualTraceManager()
    data = {
        "d41d8cd98f00b204e9800998ecf8427e": "Trace 1",
        "5d41402abc4b2a76b9719d911017c592": "Trace 2",
    }

    # Should not raise
    manager.load_traces_from_json(data)

    assert manager.get_trace_count() == 2


@pytest.mark.unit
def test_validate_trace_data_not_dict_raises_error() -> None:
    """Test that non-dict data raises ManualTraceError."""
    manager = ManualTraceManager()

    with pytest.raises(ManualTraceError, match="Invalid trace data format"):
        manager.load_traces_from_json("not a dict")


@pytest.mark.unit
def test_validate_trace_data_empty_raises_error() -> None:
    """Test that empty data raises ManualTraceError."""
    manager = ManualTraceManager()

    with pytest.raises(ManualTraceError, match="Empty trace data"):
        manager.load_traces_from_json({})


@pytest.mark.unit
def test_validate_trace_data_invalid_hash_raises_error() -> None:
    """Test that invalid hash format raises ManualTraceError."""
    manager = ManualTraceManager()

    with pytest.raises(ManualTraceError, match="Invalid question hash format"):
        manager.load_traces_from_json({"not-a-valid-hash": "trace"})


@pytest.mark.unit
def test_validate_trace_data_invalid_hash_length() -> None:
    """Test that wrong length hash raises error."""
    manager = ManualTraceManager()

    with pytest.raises(ManualTraceError, match="Invalid question hash format"):
        manager.load_traces_from_json({"abc123": "trace"})


@pytest.mark.unit
def test_validate_trace_data_invalid_hash_chars() -> None:
    """Test that hash with invalid chars raises error."""
    manager = ManualTraceManager()

    with pytest.raises(ManualTraceError, match="Invalid question hash format"):
        manager.load_traces_from_json({"g" * 32: "trace"})  # 'g' is not a valid hex char in this context


@pytest.mark.unit
def test_validate_trace_data_non_string_trace_raises_error() -> None:
    """Test that non-string trace raises ManualTraceError."""
    manager = ManualTraceManager()

    with pytest.raises(ManualTraceError, match="Invalid trace content"):
        manager.load_traces_from_json({"d41d8cd98f00b204e9800998ecf8427e": 123})


@pytest.mark.unit
def test_validate_trace_data_empty_trace_raises_error() -> None:
    """Test that empty trace raises ManualTraceError."""
    manager = ManualTraceManager()

    with pytest.raises(ManualTraceError, match="Invalid trace content"):
        manager.load_traces_from_json({"d41d8cd98f00b204e9800998ecf8427e": "   "})


@pytest.mark.unit
def test_is_valid_md5_hash() -> None:
    """Test MD5 hash validation."""
    manager = ManualTraceManager()

    # Valid hashes
    assert manager._is_valid_md5_hash("d41d8cd98f00b204e9800998ecf8427e") is True
    assert manager._is_valid_md5_hash("ABCDEF0123456789ABCDEF0123456789") is True
    assert manager._is_valid_md5_hash("00000000000000000000000000000000") is True

    # Invalid hashes
    assert manager._is_valid_md5_hash("") is False
    assert manager._is_valid_md5_hash("abc") is False
    assert manager._is_valid_md5_hash("g" * 32) is False  # 'g' is not a hex char
    assert manager._is_valid_md5_hash(None) is False
    assert manager._is_valid_md5_hash(123) is False


# =============================================================================
# Manual Trace Module Functions Tests
# =============================================================================


@pytest.mark.unit
def test_load_manual_traces() -> None:
    """Test loading manual traces from dict."""
    clear_manual_traces()

    data = {
        "d41d8cd98f00b204e9800998ecf8427e": "Trace 1",
        "5d41402abc4b2a76b9719d911017c592": "Trace 2",
    }
    load_manual_traces(data)

    assert get_manual_trace_count() == 2
    assert get_manual_trace("d41d8cd98f00b204e9800998ecf8427e") == "Trace 1"

    clear_manual_traces()


@pytest.mark.unit
def test_get_manual_trace() -> None:
    """Test getting a manual trace."""
    clear_manual_traces()
    valid_hash = "d41d8cd98f00b204e9800998ecf8427e"
    set_manual_trace(valid_hash, "Test trace")

    assert get_manual_trace(valid_hash) == "Test trace"
    assert get_manual_trace("5d41402abc4b2a76b9719d911017c592") is None

    clear_manual_traces()


@pytest.mark.unit
def test_has_manual_trace() -> None:
    """Test checking if a trace exists."""
    clear_manual_traces()
    valid_hash = "d41d8cd98f00b204e9800998ecf8427e"
    set_manual_trace(valid_hash, "trace")

    assert has_manual_trace(valid_hash) is True
    assert has_manual_trace("5d41402abc4b2a76b9719d911017c592") is False

    clear_manual_traces()


@pytest.mark.unit
def test_get_manual_trace_count() -> None:
    """Test getting trace count."""
    clear_manual_traces()

    assert get_manual_trace_count() == 0

    hash1 = "d41d8cd98f00b204e9800998ecf8427e"
    hash2 = "5d41402abc4b2a76b9719d911017c592"
    set_manual_trace(hash1, "trace1")
    set_manual_trace(hash2, "trace2")

    assert get_manual_trace_count() == 2

    clear_manual_traces()


@pytest.mark.unit
def test_get_manual_trace_with_metrics() -> None:
    """Test getting trace with metrics."""
    clear_manual_traces()

    valid_hash = "d41d8cd98f00b204e9800998ecf8427e"
    metrics = {"tool_calls": 5, "failures": 1}
    set_manual_trace(valid_hash, "trace", agent_metrics=metrics)

    trace, retrieved_metrics = get_manual_trace_with_metrics(valid_hash)

    assert trace == "trace"
    assert retrieved_metrics == metrics

    # Test not found
    trace, metrics = get_manual_trace_with_metrics("5d41402abc4b2a76b9719d911017c592")
    assert trace is None
    assert metrics is None

    clear_manual_traces()


@pytest.mark.unit
def test_get_memory_usage_info() -> None:
    """Test getting memory usage info."""
    clear_manual_traces()

    info = get_memory_usage_info()

    assert info["trace_count"] == 0
    assert info["total_characters"] == 0

    valid_hash = "d41d8cd98f00b204e9800998ecf8427e"
    set_manual_trace(valid_hash, "trace content")

    info = get_memory_usage_info()
    assert info["trace_count"] == 1

    clear_manual_traces()


@pytest.mark.unit
def test_set_manual_trace() -> None:
    """Test setting a manual trace."""
    clear_manual_traces()

    valid_hash = "d41d8cd98f00b204e9800998ecf8427e"
    set_manual_trace(valid_hash, "test trace")

    assert get_manual_trace(valid_hash) == "test trace"

    clear_manual_traces()


@pytest.mark.unit
def test_set_manual_trace_invalid_hash_raises_error() -> None:
    """Test that setting invalid hash raises error."""
    clear_manual_traces()

    with pytest.raises(ManualTraceError, match="Invalid question hash format"):
        set_manual_trace("invalid", "trace")


# =============================================================================
# init_chat_model_unified Tests
# =============================================================================


@pytest.mark.unit
def test_init_chat_model_openai_endpoint_requires_base_url() -> None:
    """Test that openai_endpoint interface requires endpoint_base_url."""
    with pytest.raises(ValueError, match="endpoint_base_url is required"):
        init_chat_model_unified("gpt-4", interface="openai_endpoint")


@pytest.mark.unit
def test_init_chat_model_openai_endpoint_requires_api_key() -> None:
    """Test that openai_endpoint interface requires endpoint_api_key."""
    with pytest.raises(ValueError, match="endpoint_api_key is required"):
        init_chat_model_unified(
            "gpt-4",
            interface="openai_endpoint",
            endpoint_base_url="http://localhost:8000",
        )


# =============================================================================
# ManualTraces Class Tests
# =============================================================================


@pytest.mark.unit
def test_manual_traces_initialization() -> None:
    """Test ManualTraces initialization."""
    mock_benchmark = MagicMock()
    mock_benchmark._questions_cache = {}

    traces = ManualTraces(mock_benchmark)

    assert traces._benchmark is mock_benchmark


@pytest.mark.unit
def test_manual_traces_register_trace_string() -> None:
    """Test registering a string trace."""
    clear_manual_traces()

    mock_benchmark = MagicMock()
    mock_benchmark._questions_cache = {}
    traces = ManualTraces(mock_benchmark)

    traces.register_trace("d41d8cd98f00b204e9800998ecf8427e", "Test trace")

    assert get_manual_trace("d41d8cd98f00b204e9800998ecf8427e") == "Test trace"

    clear_manual_traces()


@pytest.mark.unit
def test_manual_traces_register_trace_invalid_hash() -> None:
    """Test that registering with invalid hash raises error."""
    clear_manual_traces()

    mock_benchmark = MagicMock()
    mock_benchmark._questions_cache = {}
    traces = ManualTraces(mock_benchmark)

    with pytest.raises(ManualTraceError, match="Invalid question hash"):
        traces.register_trace("invalid", "trace")

    clear_manual_traces()


@pytest.mark.unit
def test_manual_traces_register_traces_batch() -> None:
    """Test batch registering traces."""
    clear_manual_traces()

    mock_benchmark = MagicMock()
    mock_benchmark._questions_cache = {}
    traces = ManualTraces(mock_benchmark)

    traces.register_traces(
        {
            "d41d8cd98f00b204e9800998ecf8427e": "Trace 1",
            "5d41402abc4b2a76b9719d911017c592": "Trace 2",
        }
    )

    assert get_manual_trace_count() == 2

    clear_manual_traces()


@pytest.mark.unit
def test_manual_traces_register_trace_message_list() -> None:
    """Test registering a LangChain message list as trace."""
    clear_manual_traces()

    mock_benchmark = MagicMock()
    mock_benchmark._questions_cache = {}
    traces = ManualTraces(mock_benchmark)

    # Create a simple message list
    messages = [
        HumanMessage(content="Question"),
        AIMessage(content="Answer"),
    ]

    # This should convert messages to string
    traces.register_trace("d41d8cd98f00b204e9800998ecf8427e", messages)

    result = get_manual_trace("d41d8cd98f00b204e9800998ecf8427e")
    assert isinstance(result, str)
    assert "Answer" in result

    clear_manual_traces()


# =============================================================================
# Message Utilities Tests (Port-based architecture)
# =============================================================================


@pytest.mark.unit
def test_is_port_message_list() -> None:
    """Test detection of port Message lists."""
    # Port messages
    port_messages = [Message.user("Hello"), Message.assistant("World")]
    assert is_port_message_list(port_messages) is True

    # Empty list
    assert is_port_message_list([]) is False

    # LangChain messages
    lc_messages = [HumanMessage(content="Hello")]
    assert is_port_message_list(lc_messages) is False

    # Other types
    assert is_port_message_list(["string"]) is False


@pytest.mark.unit
def test_is_langchain_message_list() -> None:
    """Test detection of LangChain message lists."""
    # LangChain messages
    lc_messages = [HumanMessage(content="Hello"), AIMessage(content="World")]
    assert is_langchain_message_list(lc_messages) is True

    # Empty list
    assert is_langchain_message_list([]) is False

    # Port messages
    port_messages = [Message.user("Hello")]
    assert is_langchain_message_list(port_messages) is False


@pytest.mark.unit
def test_convert_langchain_messages() -> None:
    """Test conversion from LangChain to port messages."""
    lc_messages = [
        HumanMessage(content="What is 2+2?"),
        AIMessage(content="The answer is 4."),
    ]

    port_messages = convert_langchain_messages(lc_messages)

    assert len(port_messages) == 2
    assert port_messages[0].role == Role.USER
    assert port_messages[0].text == "What is 2+2?"
    assert port_messages[1].role == Role.ASSISTANT
    assert port_messages[1].text == "The answer is 4."


@pytest.mark.unit
def test_harmonize_messages_basic() -> None:
    """Test basic message harmonization."""
    messages = [
        Message.user("Question?"),
        Message.assistant("Answer!"),
    ]

    trace = harmonize_messages(messages)

    # Should contain the assistant message but skip the first user message
    assert "AI Message" in trace
    assert "Answer!" in trace


@pytest.mark.unit
def test_harmonize_messages_with_tools() -> None:
    """Test message harmonization with tool calls and results."""
    messages = [
        Message.user("Calculate 2+2"),
        Message.assistant(
            "I'll calculate that.",
            tool_calls=[ToolUseContent(id="calc1", name="calculator", input={"expr": "2+2"})],
        ),
        Message.tool_result("calc1", "4"),
        Message.assistant("The answer is 4."),
    ]

    trace = harmonize_messages(messages)

    assert "Tool Calls:" in trace
    assert "calculator" in trace
    assert "Tool Message" in trace
    assert "The answer is 4." in trace


@pytest.mark.unit
def test_extract_agent_metrics_basic() -> None:
    """Test basic agent metrics extraction."""
    messages = [
        Message.user("Question?"),
        Message.assistant("Thinking..."),
        Message.assistant("Final answer."),
    ]

    metrics = extract_agent_metrics(messages)

    assert metrics["iterations"] == 2
    assert metrics["tool_calls"] == 0
    assert metrics["tools_used"] == []


@pytest.mark.unit
def test_extract_agent_metrics_with_tools() -> None:
    """Test agent metrics extraction with tool usage."""
    messages = [
        Message.user("Calculate 2+2"),
        Message.assistant(
            "",
            tool_calls=[
                ToolUseContent(id="calc1", name="calculator", input={}),
                ToolUseContent(id="search1", name="search", input={}),
            ],
        ),
        Message.tool_result("calc1", "4"),
        Message.tool_result("search1", "found it"),
        Message.assistant("Done."),
    ]

    metrics = extract_agent_metrics(messages)

    assert metrics["iterations"] == 2
    assert metrics["tool_calls"] == 2
    assert set(metrics["tools_used"]) == {"calculator", "search"}
    assert metrics["tool_call_counts"] == {"calculator": 1, "search": 1}


@pytest.mark.unit
def test_extract_agent_metrics_with_failures() -> None:
    """Test agent metrics extraction detects suspected failures."""
    messages = [
        Message.assistant("", tool_calls=[ToolUseContent(id="api1", name="api", input={})]),
        Message.tool_result("api1", "Error: Connection timeout"),
        Message.assistant("There was an error."),
    ]

    metrics = extract_agent_metrics(messages)

    assert metrics["suspect_failed_tool_calls"] == 1
    assert len(metrics["suspect_failed_tools"]) == 1


@pytest.mark.unit
def test_preprocess_message_list_port_format() -> None:
    """Test preprocessing port Message lists."""
    messages = [
        Message.user("Question?"),
        Message.assistant("Answer!"),
    ]

    trace, metrics = preprocess_message_list(messages)

    assert isinstance(trace, str)
    assert "Answer!" in trace
    assert metrics is not None
    assert metrics["iterations"] == 1


@pytest.mark.unit
def test_preprocess_message_list_langchain_format() -> None:
    """Test preprocessing LangChain message lists."""
    messages = [
        HumanMessage(content="Question?"),
        AIMessage(content="Answer!"),
    ]

    trace, metrics = preprocess_message_list(messages)

    assert isinstance(trace, str)
    assert "Answer!" in trace
    assert metrics is not None


@pytest.mark.unit
def test_manual_traces_register_port_messages() -> None:
    """Test registering port Message lists."""
    clear_manual_traces()

    mock_benchmark = MagicMock()
    mock_benchmark._questions_cache = {}
    traces = ManualTraces(mock_benchmark)

    # Create port message list
    messages = [
        Message.user("What is 2+2?"),
        Message.assistant("The answer is 4."),
    ]

    traces.register_trace("d41d8cd98f00b204e9800998ecf8427e", messages)

    result = get_manual_trace("d41d8cd98f00b204e9800998ecf8427e")
    assert isinstance(result, str)
    assert "answer is 4" in result

    clear_manual_traces()
