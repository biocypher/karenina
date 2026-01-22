"""Tests for tool schema conversion utilities in claude_tool adapter.

Tests create_mcp_tool_function, wrap_mcp_tool, wrap_static_tool,
wrap_tool_with_executor, and apply_cache_control_to_tool.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from karenina.adapters.claude_tool.tools import (
    apply_cache_control_to_tool,
    create_mcp_tool_function,
    wrap_static_tool,
    wrap_tool_with_executor,
)


class TestCreateMcpToolFunction:
    """Tests for create_mcp_tool_function."""

    @pytest.mark.asyncio
    async def test_creates_callable_function(self) -> None:
        """Test creates a callable async function."""
        mock_session = AsyncMock()
        mock_tool = MagicMock()
        mock_tool.name = "search"

        tool_fn = create_mcp_tool_function(mock_session, mock_tool, "test_server")

        assert callable(tool_fn)

    @pytest.mark.asyncio
    async def test_calls_session_call_tool(self) -> None:
        """Test function calls session.call_tool with correct args."""
        mock_session = AsyncMock()

        # Mock the result
        mock_content = MagicMock()
        mock_content.text = "Search results"
        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_session.call_tool.return_value = mock_result

        mock_tool = MagicMock()
        mock_tool.name = "search"

        tool_fn = create_mcp_tool_function(mock_session, mock_tool, "test_server")
        result = await tool_fn(query="test query")

        mock_session.call_tool.assert_called_once_with("search", {"query": "test query"})
        assert result == "Search results"

    @pytest.mark.asyncio
    async def test_handles_multiple_content_blocks(self) -> None:
        """Test function handles multiple content blocks in result."""
        mock_session = AsyncMock()

        mock_content1 = MagicMock()
        mock_content1.text = "Part 1"
        mock_content2 = MagicMock()
        mock_content2.text = "Part 2"
        mock_result = MagicMock()
        mock_result.content = [mock_content1, mock_content2]
        mock_session.call_tool.return_value = mock_result

        mock_tool = MagicMock()
        mock_tool.name = "search"

        tool_fn = create_mcp_tool_function(mock_session, mock_tool, "test_server")
        result = await tool_fn()

        assert result == "Part 1\nPart 2"

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(self) -> None:
        """Test function returns error message on exception."""
        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = Exception("Connection lost")

        mock_tool = MagicMock()
        mock_tool.name = "search"

        tool_fn = create_mcp_tool_function(mock_session, mock_tool, "test_server")
        result = await tool_fn()

        assert "Error calling MCP tool 'search'" in result
        assert "Connection lost" in result


class TestWrapStaticTool:
    """Tests for wrap_static_tool function."""

    def test_wraps_tool_with_decorator(self) -> None:
        """Test wraps tool with beta_async_tool decorator."""
        from karenina.ports import Tool

        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
        )

        wrapped = wrap_static_tool(tool)

        # SDK returns BetaAsyncFunctionTool with to_dict method
        assert hasattr(wrapped, "to_dict")

    def test_wrapped_tool_has_correct_attributes(self) -> None:
        """Test wrapped tool has correct name and description."""
        from karenina.ports import Tool

        tool = Tool(
            name="my_tool",
            description="Description here",
            input_schema={"type": "object", "properties": {"arg": {"type": "string"}}},
        )

        wrapped = wrap_static_tool(tool)

        # BetaAsyncFunctionTool has these as direct attributes
        assert wrapped.name == "my_tool"
        assert wrapped.description == "Description here"

    @pytest.mark.asyncio
    async def test_wrapped_tool_returns_placeholder(self) -> None:
        """Test wrapped static tool returns placeholder response."""
        from karenina.ports import Tool

        tool = Tool(
            name="placeholder_tool",
            description="A placeholder",
            input_schema={"type": "object", "properties": {}},
        )

        wrapped = wrap_static_tool(tool)

        # The wrapped function is the callable itself
        # Call it to get the result
        result = await wrapped(arg1="test", arg2=123)

        assert "placeholder_tool" in result
        assert "arg1" in result or "test" in result


class TestWrapToolWithExecutor:
    """Tests for wrap_tool_with_executor function."""

    @pytest.mark.asyncio
    async def test_uses_provided_executor(self) -> None:
        """Test uses the provided executor function."""
        from karenina.ports import Tool

        tool = Tool(
            name="custom_tool",
            description="Custom tool",
            input_schema={"type": "object", "properties": {}},
        )

        async def executor(**kwargs: Any) -> str:
            return f"Executed with: {kwargs}"

        wrapped = wrap_tool_with_executor(tool, executor)
        result = await wrapped(key="value")

        assert "Executed with:" in result
        assert "key" in result

    def test_wrapped_executor_has_correct_attributes(self) -> None:
        """Test wrapped executor has correct name and description."""
        from karenina.ports import Tool

        tool = Tool(
            name="executor_tool",
            description="Tool with executor",
            input_schema={"type": "object", "properties": {}},
        )

        async def executor(**kwargs: Any) -> str:
            return "result"

        wrapped = wrap_tool_with_executor(tool, executor)

        assert wrapped.name == "executor_tool"
        assert wrapped.description == "Tool with executor"


class TestApplyCacheControlToTool:
    """Tests for apply_cache_control_to_tool function."""

    def test_returns_same_tool_without_to_params(self) -> None:
        """Test returns same tool if it doesn't have to_params method.

        Note: The Anthropic SDK's BetaAsyncFunctionTool uses to_dict, not to_params.
        This test verifies the function gracefully handles tools without to_params.
        """
        mock_tool = MagicMock(spec=["to_dict"])  # Has to_dict but not to_params

        result = apply_cache_control_to_tool(mock_tool)

        # Should return unchanged since there's no to_params
        assert result is mock_tool

    def test_patches_to_params_when_present(self) -> None:
        """Test patches to_params method when present."""
        # Create a mock with to_params
        mock_tool = MagicMock()
        mock_tool.to_params.return_value = {"name": "test", "description": "desc"}

        original_to_params = mock_tool.to_params
        cached = apply_cache_control_to_tool(mock_tool)

        # to_params should be patched
        assert cached.to_params is not original_to_params
        params = cached.to_params()
        assert "cache_control" in params
        assert params["cache_control"] == {"type": "ephemeral"}

    def test_preserves_original_params_in_patch(self) -> None:
        """Test original params are preserved after patching."""
        mock_tool = MagicMock()
        mock_tool.to_params.return_value = {
            "name": "my_tool",
            "description": "My description",
            "input_schema": {"type": "object"},
        }

        cached = apply_cache_control_to_tool(mock_tool)
        params = cached.to_params()

        assert params["name"] == "my_tool"
        assert params["description"] == "My description"
        assert params["input_schema"] == {"type": "object"}
        assert params["cache_control"] == {"type": "ephemeral"}


class TestWrapMcpTool:
    """Tests for wrap_mcp_tool function."""

    def test_wraps_mcp_tool_with_decorator(self) -> None:
        """Test wraps MCP tool with beta_async_tool decorator."""
        from karenina.adapters.claude_tool.tools import wrap_mcp_tool

        mock_session = AsyncMock()
        mock_tool = MagicMock()
        mock_tool.name = "mcp_search"
        mock_tool.description = "Search via MCP"
        mock_tool.inputSchema = {"type": "object", "properties": {}}

        wrapped = wrap_mcp_tool(mock_session, mock_tool, "server1")

        # BetaAsyncFunctionTool has these attributes
        assert wrapped.name == "mcp_search"
        assert wrapped.description == "Search via MCP"

    def test_uses_server_name_in_default_description(self) -> None:
        """Test uses server name when tool has no description."""
        from karenina.adapters.claude_tool.tools import wrap_mcp_tool

        mock_session = AsyncMock()
        mock_tool = MagicMock()
        mock_tool.name = "unnamed_tool"
        mock_tool.description = None
        mock_tool.inputSchema = {"type": "object", "properties": {}}

        wrapped = wrap_mcp_tool(mock_session, mock_tool, "my_server")

        assert "my_server" in wrapped.description
