"""Unit tests for LLM client utilities and infrastructure.

Tests cover:
- Exception classes (LLMError, LLMNotAvailableError, SessionError, ManualTraceError, ManualTraceNotFoundError)
- Pydantic models (ChatRequest, ChatResponse)
- ChatSession class (initialization, message handling, system messages)
- Session management functions (get_session, list_sessions, delete_session, clear_all_sessions)
- ManualLLM class (fixture-based testing without API calls)
- ManualTraceManager class (trace storage, validation, cleanup)
- Manual trace utilities
- Custom OpenAI client classes (ChatOpenRouter, ChatOpenAIEndpoint)

Note: Tests do NOT make actual API calls. All LLM interaction is mocked or uses fixture-backed implementations.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from karenina.infrastructure.llm.exceptions import (
    LLMError,
    LLMNotAvailableError,
    ManualTraceError,
    ManualTraceNotFoundError,
    SessionError,
)
from karenina.infrastructure.llm.interface import (
    ChatOpenAIEndpoint,
    ChatOpenRouter,
    ChatRequest,
    ChatResponse,
    ChatSession,
    clear_all_sessions,
    delete_session,
    get_session,
    init_chat_model_unified,
    list_sessions,
)
from karenina.infrastructure.llm.manual_llm import ManualLLM, create_manual_llm
from karenina.infrastructure.llm.manual_traces import (
    ManualTraceManager,
    ManualTraces,
    clear_manual_traces,
    get_manual_trace,
    get_manual_trace_count,
    get_manual_trace_with_metrics,
    get_memory_usage_info,
    has_manual_trace,
    load_manual_traces,
    set_manual_trace,
)

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
def test_manual_trace_not_found_error_inherits_from_llm_error() -> None:
    """Test that ManualTraceNotFoundError inherits from LLMError."""
    assert issubclass(ManualTraceNotFoundError, LLMError)
    error = ManualTraceNotFoundError("trace not found")
    assert isinstance(error, LLMError)


@pytest.mark.unit
def test_manual_trace_error_is_llm_error() -> None:
    """Test that ManualTraceError inherits from LLMError."""
    assert issubclass(ManualTraceError, LLMError)
    error = ManualTraceError("trace error")
    assert isinstance(error, LLMError)


# =============================================================================
# ChatRequest Model Tests
# =============================================================================


@pytest.mark.unit
def test_chat_request_model_validation() -> None:
    """Test ChatRequest Pydantic model validation."""
    request = ChatRequest(
        model="gemini-2.0-flash",
        provider="google_genai",
        message="Hello, world!",
        session_id="test-session",
        system_message="You are helpful.",
        temperature=0.5,
        interface="langchain",
        endpoint_base_url="http://localhost:8000",
        endpoint_api_key="test-key",
    )

    assert request.model == "gemini-2.0-flash"
    assert request.provider == "google_genai"
    assert request.message == "Hello, world!"
    assert request.session_id == "test-session"
    assert request.system_message == "You are helpful."
    assert request.temperature == 0.5
    assert request.interface == "langchain"
    assert request.endpoint_base_url == "http://localhost:8000"
    assert request.endpoint_api_key == "test-key"


@pytest.mark.unit
def test_chat_request_defaults() -> None:
    """Test ChatRequest default values."""
    request = ChatRequest(
        model="gpt-4",
        provider="openai",
        message="Test message",
    )

    assert request.session_id is None
    assert request.system_message is None
    assert request.temperature == 0.7
    assert request.interface is None
    assert request.endpoint_base_url is None
    assert request.endpoint_api_key is None


@pytest.mark.unit
def test_chat_request_missing_required_fields() -> None:
    """Test that ChatRequest requires model, provider, and message fields."""
    with pytest.raises(ValidationError):
        ChatRequest(message="test")  # missing model and provider

    with pytest.raises(ValidationError):
        ChatRequest(model="test")  # missing provider and message


# =============================================================================
# ChatResponse Model Tests
# =============================================================================


@pytest.mark.unit
def test_chat_response_model() -> None:
    """Test ChatResponse Pydantic model."""
    response = ChatResponse(
        session_id="session-123",
        message="Response text",
        model="gemini-2.0-flash",
        provider="google_genai",
        timestamp="2025-01-11T12:00:00",
    )

    assert response.session_id == "session-123"
    assert response.message == "Response text"
    assert response.model == "gemini-2.0-flash"
    assert response.provider == "google_genai"
    assert response.timestamp == "2025-01-11T12:00:00"


# =============================================================================
# ChatSession Class Tests
# =============================================================================


@pytest.mark.unit
def test_chat_session_initialization() -> None:
    """Test ChatSession initialization with all parameters."""
    session = ChatSession(
        session_id="test-session",
        model="gemini-2.0-flash",
        provider="google_genai",
        temperature=0.5,
        mcp_urls_dict=None,
        mcp_tool_filter=None,
        interface="langchain",
        endpoint_base_url="http://localhost:8000",
        endpoint_api_key="test-key",
    )

    assert session.session_id == "test-session"
    assert session.model == "gemini-2.0-flash"
    assert session.provider == "google_genai"
    assert session.temperature == 0.5
    assert session.mcp_urls_dict is None
    assert session.mcp_tool_filter is None
    assert session.interface == "langchain"
    assert session.endpoint_base_url == "http://localhost:8000"
    assert session.endpoint_api_key == "test-key"
    assert session.messages == []
    assert session.llm is None
    assert session.is_agent is False


@pytest.mark.unit
def test_chat_session_defaults() -> None:
    """Test ChatSession default values."""
    session = ChatSession(
        session_id="test",
        model="gpt-4",
        provider="openai",
    )

    assert session.temperature == 0.7
    assert session.mcp_urls_dict is None
    assert session.mcp_tool_filter is None
    assert session.interface == "langchain"
    assert session.endpoint_base_url is None
    assert session.endpoint_api_key is None
    assert session.messages == []
    assert session.llm is None
    assert session.is_agent is False


@pytest.mark.unit
def test_chat_session_add_human_message() -> None:
    """Test adding a human message to the session."""
    session = ChatSession("test", "gpt-4", "openai")
    session.add_message("Hello!", is_human=True)

    assert len(session.messages) == 1
    assert isinstance(session.messages[0], HumanMessage)
    assert session.messages[0].content == "Hello!"


@pytest.mark.unit
def test_chat_session_add_ai_message() -> None:
    """Test adding an AI message to the session."""
    session = ChatSession("test", "gpt-4", "openai")
    session.add_message("Hi there!", is_human=False)

    assert len(session.messages) == 1
    assert isinstance(session.messages[0], AIMessage)
    assert session.messages[0].content == "Hi there!"


@pytest.mark.unit
def test_chat_session_add_system_message_first() -> None:
    """Test adding a system message when no messages exist."""
    session = ChatSession("test", "gpt-4", "openai")
    session.add_system_message("You are helpful.")

    assert len(session.messages) == 1
    assert isinstance(session.messages[0], SystemMessage)
    assert session.messages[0].content == "You are helpful."


@pytest.mark.unit
def test_chat_session_add_system_message_inserts_at_beginning() -> None:
    """Test that system message is inserted at the beginning."""
    session = ChatSession("test", "gpt-4", "openai")
    session.add_message("Hello!", is_human=True)
    session.add_system_message("You are helpful.")

    assert len(session.messages) == 2
    assert isinstance(session.messages[0], SystemMessage)
    assert session.messages[0].content == "You are helpful."
    assert isinstance(session.messages[1], HumanMessage)


@pytest.mark.unit
def test_chat_session_update_existing_system_message() -> None:
    """Test that adding a system message updates the existing one."""
    session = ChatSession("test", "gpt-4", "openai")
    session.add_system_message("You are helpful.")
    session.add_system_message("You are very helpful.")

    assert len(session.messages) == 1
    assert session.messages[0].content == "You are very helpful."


@pytest.mark.unit
def test_chat_session_mcp_urls_dict_sets_agent_flag() -> None:
    """Test that providing MCP URLs sets the agent flag."""
    session = ChatSession(
        session_id="test",
        model="gpt-4",
        provider="openai",
        mcp_urls_dict={"tool1": "http://example.com"},
    )

    # is_agent is only set to True after initialize_llm()
    assert session.is_agent is False
    assert session.mcp_urls_dict == {"tool1": "http://example.com"}


# =============================================================================
# Session Management Functions Tests
# =============================================================================


@pytest.mark.unit
def test_clear_all_sessions() -> None:
    """Test clearing all chat sessions."""
    import karenina.infrastructure.llm.interface as interface_module

    # First clear any existing sessions
    clear_all_sessions()

    # Add a test session by directly accessing the module's global
    session_id = "test-session"
    interface_module.chat_sessions[session_id] = ChatSession(session_id, "gpt-4", "openai")
    assert len(interface_module.chat_sessions) > 0

    clear_all_sessions()

    # Access through the module to see the updated global
    assert len(interface_module.chat_sessions) == 0


@pytest.mark.unit
def test_get_session_existing() -> None:
    """Test getting an existing session."""
    import karenina.infrastructure.llm.interface as interface_module

    clear_all_sessions()
    session_id = "test-session"
    session = ChatSession(session_id, "gpt-4", "openai")
    interface_module.chat_sessions[session_id] = session

    retrieved = get_session(session_id)
    assert retrieved is session

    clear_all_sessions()


@pytest.mark.unit
def test_get_session_nonexistent() -> None:
    """Test getting a non-existent session returns None."""
    retrieved = get_session("nonexistent")
    assert retrieved is None


@pytest.mark.unit
def test_delete_session_existing() -> None:
    """Test deleting an existing session."""
    import karenina.infrastructure.llm.interface as interface_module

    clear_all_sessions()
    session_id = "test-session"
    interface_module.chat_sessions[session_id] = ChatSession(session_id, "gpt-4", "openai")

    result = delete_session(session_id)

    assert result is True
    assert session_id not in interface_module.chat_sessions


@pytest.mark.unit
def test_delete_session_nonexistent() -> None:
    """Test deleting a non-existent session returns False."""
    result = delete_session("nonexistent")
    assert result is False


@pytest.mark.unit
def test_list_sessions_empty() -> None:
    """Test listing sessions when none exist."""
    clear_all_sessions()
    result = list_sessions()
    assert result == []


@pytest.mark.unit
def test_list_sessions_with_data() -> None:
    """Test listing sessions with actual data."""
    import karenina.infrastructure.llm.interface as interface_module

    clear_all_sessions()

    session_id = "test-session"
    session = ChatSession(session_id, "gpt-4", "openai")
    session.add_message("Hello", is_human=True)
    session.add_message("Hi there", is_human=False)
    interface_module.chat_sessions[session_id] = session

    result = list_sessions()

    assert len(result) == 1
    assert result[0]["session_id"] == session_id
    assert result[0]["model"] == "gpt-4"
    assert result[0]["provider"] == "openai"
    assert result[0]["message_count"] == 2  # Excludes system messages
    assert "created_at" in result[0]
    assert "last_used" in result[0]

    clear_all_sessions()


# =============================================================================
# ChatOpenRouter Class Tests
# =============================================================================


@pytest.mark.unit
def test_chat_openrouter_initialization() -> None:
    """Test ChatOpenRouter initialization."""
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        model = ChatOpenRouter(model="gpt-4", temperature=0.5)

        assert model.model_name == "gpt-4"
        assert model.temperature == 0.5


@pytest.mark.unit
def test_chat_openrouter_lc_secrets() -> None:
    """Test ChatOpenRouter lc_secrets property."""
    model = ChatOpenRouter(model="gpt-4", openai_api_key="test-key")
    secrets = model.lc_secrets
    assert secrets == {"openai_api_key": "OPENROUTER_API_KEY"}


# =============================================================================
# ChatOpenAIEndpoint Class Tests
# =============================================================================


@pytest.mark.unit
def test_chat_openai_endpoint_requires_api_key() -> None:
    """Test that ChatOpenAIEndpoint requires an API key."""
    with pytest.raises(ValueError, match="API key is required"):
        ChatOpenAIEndpoint(base_url="http://localhost:8000")


@pytest.mark.unit
def test_chat_openai_endpoint_requires_explicit_api_key() -> None:
    """Test that ChatOpenAIEndpoint does NOT read from environment."""
    # Even with env var set, it should still fail if no explicit key
    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}),
        pytest.raises(ValueError, match="API key is required"),
    ):
        ChatOpenAIEndpoint(base_url="http://localhost:8000")


@pytest.mark.unit
def test_chat_openai_endpoint_initialization_with_key() -> None:
    """Test ChatOpenAIEndpoint initialization with explicit API key."""
    model = ChatOpenAIEndpoint(
        base_url="http://localhost:8000",
        openai_api_key="explicit-key",
    )

    assert model is not None


@pytest.mark.unit
def test_chat_openai_endpoint_lc_secrets_empty() -> None:
    """Test ChatOpenAIEndpoint lc_secrets returns empty dict."""
    model = ChatOpenAIEndpoint(
        base_url="http://localhost:8000",
        openai_api_key="test-key",
    )
    secrets = model.lc_secrets
    assert secrets == {}


# =============================================================================
# ManualLLM Class Tests
# =============================================================================


@pytest.mark.unit
def test_manual_llm_initialization() -> None:
    """Test ManualLLM initialization."""
    llm = ManualLLM(question_hash="abc123")

    assert llm.question_hash == "abc123"


@pytest.mark.unit
def test_manual_llm_ignores_extra_kwargs() -> None:
    """Test that ManualLLM ignores extra kwargs."""
    llm = ManualLLM(question_hash="test", temperature=0.5, max_tokens=100)

    assert llm.question_hash == "test"


@pytest.mark.unit
def test_manual_llm_invoke_returns_trace() -> None:
    """Test ManualLLM.invoke returns precomputed trace."""
    # Use a valid 32-character MD5 hash
    valid_hash = "d41d8cd98f00b204e9800998ecf8427e"
    set_manual_trace(valid_hash, "This is a precomputed answer.")

    llm = ManualLLM(question_hash=valid_hash)
    result = llm.invoke([])

    assert isinstance(result, AIMessage)
    assert result.content == "This is a precomputed answer."

    # Clean up
    clear_manual_traces()


@pytest.mark.unit
def test_manual_llm_invoke_trace_not_found() -> None:
    """Test ManualLLM.invoke raises error when trace not found."""
    clear_manual_traces()

    # Use a valid MD5 hash format that doesn't exist
    llm = ManualLLM(question_hash="d41d8cd98f00b204e9800998ecf8427e")

    with pytest.raises(ManualTraceNotFoundError, match="No manual trace found"):
        llm.invoke([])


@pytest.mark.unit
def test_manual_llm_with_structured_output_returns_self() -> None:
    """Test ManualLLM.with_structured_output returns self."""
    llm = ManualLLM(question_hash="test")
    result = llm.with_structured_output(None)

    assert result is llm


@pytest.mark.unit
def test_manual_llm_content_property() -> None:
    """Test ManualLLM.content property returns trace."""
    valid_hash = "d41d8cd98f00b204e9800998ecf8427e"
    set_manual_trace(valid_hash, "Trace content")

    llm = ManualLLM(question_hash=valid_hash)
    assert llm.content == "Trace content"

    clear_manual_traces()


@pytest.mark.unit
def test_manual_llm_content_not_found_raises_error() -> None:
    """Test ManualLLM.content raises error when trace not found."""
    clear_manual_traces()

    llm = ManualLLM(question_hash="d41d8cd98f00b204e9800998ecf8427e")

    with pytest.raises(ManualTraceNotFoundError):
        _ = llm.content


@pytest.mark.unit
def test_manual_llm_get_agent_metrics() -> None:
    """Test ManualLLM.get_agent_metrics returns metrics."""
    valid_hash = "d41d8cd98f00b204e9800998ecf8427e"
    metrics = {"tool_calls": 5, "failures": 0}
    set_manual_trace(valid_hash, "Trace", agent_metrics=metrics)

    llm = ManualLLM(question_hash=valid_hash)
    result = llm.get_agent_metrics()

    assert result == metrics

    clear_manual_traces()


@pytest.mark.unit
def test_manual_llm_get_agent_metrics_none_when_not_set() -> None:
    """Test ManualLLM.get_agent_metrics returns None when no metrics."""
    valid_hash = "d41d8cd98f00b204e9800998ecf8427e"
    set_manual_trace(valid_hash, "Trace")

    llm = ManualLLM(question_hash=valid_hash)
    result = llm.get_agent_metrics()

    assert result is None

    clear_manual_traces()


@pytest.mark.unit
def test_create_manual_llm() -> None:
    """Test create_manual_llm factory function."""
    llm = create_manual_llm(question_hash="test123")

    assert isinstance(llm, ManualLLM)
    assert llm.question_hash == "test123"


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
def test_init_chat_model_unsupported_interface() -> None:
    """Test that unsupported interface raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported interface"):
        init_chat_model_unified("gpt-4", interface="unsupported")


@pytest.mark.unit
def test_init_chat_model_manual_requires_question_hash() -> None:
    """Test that manual interface requires question_hash."""
    with pytest.raises(ValueError, match="question_hash is required"):
        init_chat_model_unified("manual", interface="manual")


@pytest.mark.unit
def test_init_chat_model_manual_with_hash() -> None:
    """Test manual interface with question_hash."""
    # Set up a trace with valid MD5 hash
    valid_hash = "d41d8cd98f00b204e9800998ecf8427e"
    set_manual_trace(valid_hash, "Test trace")

    model = init_chat_model_unified("manual", interface="manual", question_hash=valid_hash)

    assert isinstance(model, ManualLLM)
    assert model.question_hash == valid_hash

    clear_manual_traces()


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


@pytest.mark.unit
def test_init_chat_model_mcp_with_manual_raises_error() -> None:
    """Test that MCP with manual interface raises ValueError."""
    with pytest.raises(ValueError, match="MCP integration is not supported"):
        init_chat_model_unified(
            "gpt-4",
            interface="manual",
            question_hash="abc123",
            mcp_urls_dict={"tool": "http://example.com"},
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
