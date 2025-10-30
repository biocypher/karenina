"""Tests for manual trace functionality."""

import pytest

from karenina.llm.interface import init_chat_model_unified
from karenina.llm.manual_llm import ManualLLM, ManualTraceNotFoundError, create_manual_llm
from karenina.llm.manual_traces import (
    ManualTraceError,
    ManualTraceManager,
    clear_manual_traces,
    get_manual_trace,
    get_manual_trace_count,
    has_manual_trace,
    load_manual_traces,
)
from karenina.schemas import ModelConfig


class TestManualTraceManager:
    """Test cases for ManualTraceManager."""

    def test_init(self) -> None:
        """Test manager initialization."""
        manager = ManualTraceManager()
        assert manager.get_trace_count() == 0
        assert manager.get_all_traces() == {}

    def test_load_valid_traces(self) -> None:
        """Test loading valid trace data."""
        manager = ManualTraceManager()
        trace_data = {
            "d41d8cd98f00b204e9800998ecf8427e": "Answer trace 1",
            "5d41402abc4b2a76b9719d911017c592": "Answer trace 2",
        }

        manager.load_traces_from_json(trace_data)

        assert manager.get_trace_count() == 2
        assert manager.get_trace("d41d8cd98f00b204e9800998ecf8427e") == "Answer trace 1"
        assert manager.has_trace("d41d8cd98f00b204e9800998ecf8427e")
        assert not manager.has_trace("nonexistent_hash")

    def test_load_invalid_hash_format(self) -> None:
        """Test loading traces with invalid hash format."""
        manager = ManualTraceManager()

        # Too short hash
        with pytest.raises(ManualTraceError, match="Invalid question hash format"):
            manager.load_traces_from_json({"short": "trace"})

        # Invalid characters
        with pytest.raises(ManualTraceError, match="Invalid question hash format"):
            manager.load_traces_from_json({"invalid_hash_with_special_chars!": "trace"})

        # Too long hash
        with pytest.raises(ManualTraceError, match="Invalid question hash format"):
            manager.load_traces_from_json({"d41d8cd98f00b204e9800998ecf8427eextra": "trace"})

    def test_load_invalid_trace_content(self) -> None:
        """Test loading traces with invalid content."""
        manager = ManualTraceManager()
        valid_hash = "d41d8cd98f00b204e9800998ecf8427e"

        # Empty string
        with pytest.raises(ManualTraceError, match="Invalid trace content"):
            manager.load_traces_from_json({valid_hash: ""})

        # Whitespace only
        with pytest.raises(ManualTraceError, match="Invalid trace content"):
            manager.load_traces_from_json({valid_hash: "   "})

        # Non-string content
        with pytest.raises(ManualTraceError, match="Invalid trace content"):
            manager.load_traces_from_json({valid_hash: 123})

    def test_load_empty_data(self) -> None:
        """Test loading empty trace data."""
        manager = ManualTraceManager()

        with pytest.raises(ManualTraceError, match="Empty trace data"):
            manager.load_traces_from_json({})

    def test_load_non_dict_data(self) -> None:
        """Test loading non-dictionary data."""
        manager = ManualTraceManager()

        with pytest.raises(ManualTraceError, match="Invalid trace data format"):
            manager.load_traces_from_json(["not", "a", "dict"])

    def test_clear_traces(self) -> None:
        """Test clearing traces."""
        manager = ManualTraceManager()
        manager.load_traces_from_json({"d41d8cd98f00b204e9800998ecf8427e": "trace"})

        assert manager.get_trace_count() == 1
        manager.clear_traces()
        assert manager.get_trace_count() == 0


class TestManualLLM:
    """Test cases for ManualLLM."""

    def setup_method(self) -> None:
        """Set up test traces before each test."""
        clear_manual_traces()
        load_manual_traces(
            {
                "d41d8cd98f00b204e9800998ecf8427e": "Test answer trace 1",
                "5d41402abc4b2a76b9719d911017c592": "Test answer trace 2",
            }
        )

    def teardown_method(self) -> None:
        """Clean up traces after each test."""
        clear_manual_traces()

    def test_manual_llm_creation(self) -> None:
        """Test creating ManualLLM instance."""
        llm = create_manual_llm("d41d8cd98f00b204e9800998ecf8427e")
        assert isinstance(llm, ManualLLM)
        assert llm.question_hash == "d41d8cd98f00b204e9800998ecf8427e"

    def test_manual_llm_invoke_success(self) -> None:
        """Test successful LLM invocation."""
        llm = ManualLLM("d41d8cd98f00b204e9800998ecf8427e")
        response = llm.invoke([])  # Messages ignored for manual traces

        assert response.content == "Test answer trace 1"

    def test_manual_llm_invoke_not_found(self) -> None:
        """Test LLM invocation with missing trace."""
        llm = ManualLLM("nonexistent_hash")

        with pytest.raises(ManualTraceNotFoundError, match="No manual trace found"):
            llm.invoke([])

    def test_manual_llm_structured_output(self) -> None:
        """Test structured output compatibility."""
        llm = ManualLLM("d41d8cd98f00b204e9800998ecf8427e")
        structured_llm = llm.with_structured_output(dict)

        assert structured_llm is llm  # Should return self

    def test_manual_llm_content_property(self) -> None:
        """Test content property access."""
        llm = ManualLLM("d41d8cd98f00b204e9800998ecf8427e")
        assert llm.content == "Test answer trace 1"

    def test_manual_llm_content_not_found(self) -> None:
        """Test content property with missing trace."""
        llm = ManualLLM("nonexistent_hash")

        with pytest.raises(ManualTraceNotFoundError):
            _ = llm.content


class TestGlobalTraceFunctions:
    """Test global trace management functions."""

    def teardown_method(self) -> None:
        """Clean up traces after each test."""
        clear_manual_traces()

    def test_global_functions(self) -> None:
        """Test global trace management functions."""
        # Initially empty
        assert get_manual_trace_count() == 0
        assert not has_manual_trace("test_hash")
        assert get_manual_trace("test_hash") is None

        # Load traces
        test_hash = "d41d8cd98f00b204e9800998ecf8427e"
        load_manual_traces({test_hash: "Test trace"})

        # Check functions work
        assert get_manual_trace_count() == 1
        assert has_manual_trace(test_hash)
        assert get_manual_trace(test_hash) == "Test trace"

        # Clear traces
        clear_manual_traces()
        assert get_manual_trace_count() == 0


class TestIntegrationWithLLMInterface:
    """Test integration with the unified LLM interface."""

    def setup_method(self) -> None:
        """Set up test traces before each test."""
        clear_manual_traces()
        load_manual_traces({"d41d8cd98f00b204e9800998ecf8427e": "Manual trace response"})

    def teardown_method(self) -> None:
        """Clean up traces after each test."""
        clear_manual_traces()

    def test_init_chat_model_unified_manual(self) -> None:
        """Test initializing manual interface through unified function."""
        llm = init_chat_model_unified(
            model="manual", interface="manual", question_hash="d41d8cd98f00b204e9800998ecf8427e"
        )

        assert isinstance(llm, ManualLLM)
        response = llm.invoke([])
        assert response.content == "Manual trace response"

    def test_init_chat_model_unified_missing_hash(self) -> None:
        """Test manual interface without question_hash."""
        with pytest.raises(ValueError, match="question_hash is required for manual interface"):
            init_chat_model_unified(model="manual", interface="manual")


class TestModelConfiguration:
    """Test ModelConfiguration with manual interface."""

    def test_model_config_manual_interface(self) -> None:
        """Test creating ModelConfiguration with manual interface."""
        config = ModelConfig(
            id="test-manual",
            model_provider="manual",
            model_name="manual",
            interface="manual",
            system_prompt="Test prompt",
        )

        assert config.interface == "manual"
        assert config.model_provider == "manual"
        assert config.model_name == "manual"
