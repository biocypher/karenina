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
        "brave-search",  # MCP server name
        "http://localhost:3001",  # MCP server URL
        "brave_web_search",  # mcp_tool_filter
    ]

    # Mock Confirm responses
    mock_confirm.ask.side_effect = [
        False,  # Use custom system prompt? No
        True,  # Configure MCP tools? Yes
        False,  # Add another MCP server? No
        True,  # Filter specific MCP tools? Yes
    ]

    # Mock MCP validation to return tools
    with patch("karenina.infrastructure.llm.mcp_utils.sync_create_mcp_client_and_tools") as mock_mcp:
        # Create mock tool objects
        class MockTool:
            def __init__(self, name: str, description: str):
                self.name = name
                self.description = description

        mock_tool = MockTool("brave_web_search", "Search the web using Brave Search")
        mock_mcp.return_value = (None, [mock_tool])  # (client, tools)

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


def test_mcp_config_server_validation_failure(mock_console_and_prompts):
    """Test that MCP server validation failures are handled gracefully."""
    mock_console, mock_prompt, mock_confirm = mock_console_and_prompts

    # Mock user inputs (no system prompt text when use_custom_prompt=False)
    mock_prompt.ask.side_effect = [
        "langchain",  # interface
        "test-invalid",  # model_id
        "gpt-4.1-mini",  # model_name
        "openai",  # model_provider
        "0.1",  # temperature
        "2",  # max_retries
        "invalid-server",  # MCP server name
        "http://invalid-url",  # MCP server URL
    ]

    # Mock Confirm responses
    mock_confirm.ask.side_effect = [
        False,  # Use custom system prompt? No
        True,  # Configure MCP tools? Yes
        False,  # Continue with this MCP server anyway? No (after validation fails)
        False,  # Add another MCP server? No
    ]

    # Mock MCP validation to raise an exception
    with patch("karenina.infrastructure.llm.mcp_utils.sync_create_mcp_client_and_tools") as mock_mcp:
        mock_mcp.side_effect = Exception("Connection refused")

        # Call the function - should handle validation failure gracefully
        model_config = _prompt_for_model(model_type="answering", mode="advanced")

    # Verify MCP config is None after validation failure and user declining
    assert model_config.mcp_urls_dict is None
    assert model_config.mcp_tool_filter is None

    # Verify error message was shown
    assert any("Failed to validate MCP server" in str(call) for call in mock_console.print.call_args_list)
