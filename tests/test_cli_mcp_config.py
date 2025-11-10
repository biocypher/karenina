"""Test MCP configuration in interactive CLI."""

from unittest.mock import patch

import pytest

from karenina.cli.interactive import _prompt_for_model


@pytest.fixture
def mock_console_and_prompts():
    """Mock console and prompt functions."""
    with (
        patch("karenina.cli.interactive.console") as mock_console,
        patch("karenina.cli.interactive.Prompt") as mock_prompt,
        patch("karenina.cli.interactive.Confirm") as mock_confirm,
    ):
        yield mock_console, mock_prompt, mock_confirm


def test_mcp_config_for_answering_model_advanced_mode(mock_console_and_prompts):
    """Test that MCP configuration is prompted for answering models in advanced mode."""
    mock_console, mock_prompt, mock_confirm = mock_console_and_prompts

    # Mock user inputs (note: no system prompt text when use_custom_prompt=False)
    mock_prompt.ask.side_effect = [
        "langchain",  # interface
        "test-answering",  # model_id
        "gpt-4.1-mini",  # model_name
        "openai",  # model_provider
        "0.1",  # temperature
        "2",  # max_retries
        # No system_prompt text since use_custom_prompt=False
        '{"brave-search": "http://localhost:3001"}',  # mcp_urls_dict
        "brave_web_search",  # mcp_tool_filter
    ]

    # Mock Confirm responses
    mock_confirm.ask.side_effect = [
        False,  # Use custom system prompt? No
        True,  # Configure MCP tools? Yes
        True,  # Filter specific MCP tools? Yes
    ]

    # Call the function with advanced mode and answering model type
    model_config = _prompt_for_model(model_type="answering", mode="advanced")

    # Verify the model config has MCP settings
    assert model_config.id == "test-answering"
    assert model_config.model_name == "gpt-4.1-mini"
    assert model_config.model_provider == "openai"
    assert model_config.mcp_urls_dict == {"brave-search": "http://localhost:3001"}
    assert model_config.mcp_tool_filter == ["brave_web_search"]

    # Verify that the MCP configuration prompt was shown
    assert any("MCP Tools Configuration" in str(call) for call in mock_console.print.call_args_list)


def test_mcp_config_skipped_for_parsing_model_advanced_mode(mock_console_and_prompts):
    """Test that MCP configuration is NOT prompted for parsing models."""
    mock_console, mock_prompt, mock_confirm = mock_console_and_prompts

    # Mock user inputs (no system prompt text when use_custom_prompt=False)
    mock_prompt.ask.side_effect = [
        "langchain",  # interface
        "test-parsing",  # model_id
        "gpt-4.1-mini",  # model_name
        "openai",  # model_provider
        "0.1",  # temperature
        "2",  # max_retries
    ]

    # Mock Confirm responses
    mock_confirm.ask.side_effect = [
        False,  # Use custom system prompt? No
        # No MCP prompts should be shown for parsing models
    ]

    # Call the function with advanced mode and parsing model type
    model_config = _prompt_for_model(model_type="parsing", mode="advanced")

    # Verify the model config does NOT have MCP settings
    assert model_config.id == "test-parsing"
    assert model_config.model_name == "gpt-4.1-mini"
    assert model_config.model_provider == "openai"
    assert model_config.mcp_urls_dict is None
    assert model_config.mcp_tool_filter is None

    # Verify that the MCP configuration prompt was NOT shown
    assert not any("MCP Tools Configuration" in str(call) for call in mock_console.print.call_args_list)


def test_mcp_config_skipped_in_basic_mode(mock_console_and_prompts):
    """Test that MCP configuration is not prompted in basic mode."""
    mock_console, mock_prompt, mock_confirm = mock_console_and_prompts

    # Mock user inputs (no system prompt text when use_custom_prompt=False)
    mock_prompt.ask.side_effect = [
        "langchain",  # interface
        "test-basic",  # model_id
        "gpt-4.1-mini",  # model_name
        "openai",  # model_provider
        "0.1",  # temperature
        "2",  # max_retries
    ]

    # Mock Confirm responses
    mock_confirm.ask.side_effect = [
        False,  # Use custom system prompt? No
        # No MCP prompts in basic mode
    ]

    # Call the function with basic mode (even for answering model)
    model_config = _prompt_for_model(model_type="answering", mode="basic")

    # Verify the model config does NOT have MCP settings
    assert model_config.mcp_urls_dict is None
    assert model_config.mcp_tool_filter is None

    # Verify that the MCP configuration prompt was NOT shown
    assert not any("MCP Tools Configuration" in str(call) for call in mock_console.print.call_args_list)


def test_mcp_config_declined(mock_console_and_prompts):
    """Test that MCP configuration can be declined."""
    mock_console, mock_prompt, mock_confirm = mock_console_and_prompts

    # Mock user inputs (no system prompt text when use_custom_prompt=False)
    mock_prompt.ask.side_effect = [
        "langchain",  # interface
        "test-decline",  # model_id
        "gpt-4.1-mini",  # model_name
        "openai",  # model_provider
        "0.1",  # temperature
        "2",  # max_retries
    ]

    # Mock Confirm responses
    mock_confirm.ask.side_effect = [
        False,  # Use custom system prompt? No
        False,  # Configure MCP tools? No
    ]

    # Call the function with advanced mode and answering model type
    model_config = _prompt_for_model(model_type="answering", mode="advanced")

    # Verify the model config does NOT have MCP settings when declined
    assert model_config.mcp_urls_dict is None
    assert model_config.mcp_tool_filter is None


def test_mcp_config_invalid_json_handled(mock_console_and_prompts):
    """Test that invalid JSON for MCP URLs is handled gracefully."""
    mock_console, mock_prompt, mock_confirm = mock_console_and_prompts

    # Mock user inputs with invalid JSON (no system prompt text when use_custom_prompt=False)
    mock_prompt.ask.side_effect = [
        "langchain",  # interface
        "test-invalid",  # model_id
        "gpt-4.1-mini",  # model_name
        "openai",  # model_provider
        "0.1",  # temperature
        "2",  # max_retries
        "not valid json",  # invalid mcp_urls_dict
    ]

    # Mock Confirm responses
    mock_confirm.ask.side_effect = [
        False,  # Use custom system prompt? No
        True,  # Configure MCP tools? Yes
    ]

    # Call the function - should handle invalid JSON gracefully
    model_config = _prompt_for_model(model_type="answering", mode="advanced")

    # Verify MCP config is None after invalid JSON
    assert model_config.mcp_urls_dict is None
    assert model_config.mcp_tool_filter is None

    # Verify error message was shown
    assert any("Invalid JSON" in str(call) for call in mock_console.print.call_args_list)
