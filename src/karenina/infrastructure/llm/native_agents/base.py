"""Base classes for native tool-calling agents.

This module provides the abstract base class and response dataclass for native
OpenAI and Anthropic tool-calling agents that bypass LangChain/LangGraph.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NativeAgentResponse:
    """Response from native agent execution.

    Attributes:
        content: Final response text from the agent
        messages: Full message history including tool calls (for trace)
        usage: Full token usage dict with input_tokens, output_tokens, total_tokens,
               input_token_details, and output_token_details
        agent_metrics: Metrics dict with iterations, tool_calls, tools_used, etc.
        recursion_limit_reached: Whether the agent hit max iterations
    """

    content: str
    messages: list[dict[str, Any]]
    usage: dict[str, Any]  # Full usage including token details
    agent_metrics: dict[str, Any] = field(default_factory=dict)
    recursion_limit_reached: bool = False


class NativeAgentBase(ABC):
    """Abstract base class for native tool-calling agents.

    This class provides the common interface and utilities for native OpenAI
    and Anthropic agents that use their respective SDKs directly instead of
    going through LangChain/LangGraph.

    Attributes:
        model: The model name/identifier
        tools: List of tool schemas in native format
        tool_map: Dict mapping tool names to original LangChain tools for execution
        system_prompt: Optional system prompt
        temperature: Model temperature (default 0.0)
        max_iterations: Maximum agent loop iterations (default 10)
        api_key: Optional explicit API key (falls back to env vars)
        extra_kwargs: Additional kwargs passed to the model
    """

    def __init__(
        self,
        model: str,
        tools: list[dict[str, Any]],
        tool_map: dict[str, Any],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_iterations: int = 10,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the native agent.

        Args:
            model: Model name/identifier (e.g., "gpt-4.1-mini", "claude-sonnet-4-20250514")
            tools: List of tool schemas in native format (OpenAI or Anthropic)
            tool_map: Dict mapping tool names to original LangChain tools
            system_prompt: Optional system prompt
            temperature: Model temperature
            max_iterations: Maximum agent loop iterations
            api_key: Optional explicit API key (falls back to env vars)
            **kwargs: Additional kwargs passed to the underlying model
        """
        self.model = model
        self.tools = tools
        self.tool_map = tool_map
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.api_key = api_key
        self.extra_kwargs = kwargs

    @abstractmethod
    async def ainvoke(self, messages: list[dict[str, Any]]) -> NativeAgentResponse:
        """Async invoke the agent with messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            NativeAgentResponse with content, messages, usage, and metrics
        """
        pass

    def invoke(self, messages: list[dict[str, Any]]) -> NativeAgentResponse:
        """Sync invoke - runs async in thread pool if needed.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            NativeAgentResponse with content, messages, usage, and metrics
        """
        try:
            asyncio.get_running_loop()
            # Already in async context, use thread pool to avoid nested event loops
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: asyncio.run(self.ainvoke(messages)))
                return future.result(timeout=120)
        except RuntimeError:
            # No event loop running, safe to use asyncio.run directly
            return asyncio.run(self.ainvoke(messages))

    @abstractmethod
    async def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Execute a tool call and return the result.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments to pass to the tool

        Returns:
            Tool execution result as string
        """
        pass

    def _is_suspect_failure(self, result: str) -> bool:
        """Detect suspected tool failure from result text.

        Uses pattern matching to identify common error indicators in tool results.

        Args:
            result: Tool execution result string

        Returns:
            True if the result appears to indicate a failure
        """
        failure_patterns = [
            "error",
            "failed",
            "exception",
            "traceback",
            "404",
            "500",
            "502",
            "503",
            "timeout",
            "invalid",
            "unauthorized",
            "forbidden",
        ]
        result_lower = result.lower()
        return any(pattern in result_lower for pattern in failure_patterns)

    async def _execute_tool_with_langchain(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Execute a tool using the original LangChain tool from tool_map.

        This method handles the execution of MCP tools that were fetched via
        langchain-mcp-adapters and converted to native format.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments to pass to the tool

        Returns:
            Tool execution result as string
        """
        tool = self.tool_map.get(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found"

        # Try to map argument names if schema has aliases
        mapped_args = self._map_tool_args(tool, tool_args)

        try:
            # MCP tools from langchain-mcp-adapters support ainvoke
            if hasattr(tool, "ainvoke"):
                result = await tool.ainvoke(mapped_args)
            elif hasattr(tool, "invoke"):
                result = tool.invoke(mapped_args)
            elif hasattr(tool, "run"):
                result = tool.run(mapped_args)
            elif callable(tool):
                result = tool(mapped_args)
            else:
                return f"Error: Tool '{tool_name}' is not callable"

            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e!s}"

    def _map_tool_args(self, tool: Any, tool_args: dict[str, Any]) -> dict[str, Any]:
        """Map tool arguments from JSON schema names to Pydantic field names.

        MCP tools may have JSON schema property names that differ from Pydantic
        field names (due to aliases). This method attempts to map the argument
        names correctly.

        Args:
            tool: The LangChain tool with args_schema
            tool_args: Arguments with JSON schema property names

        Returns:
            Arguments with Pydantic field names
        """
        if not hasattr(tool, "args_schema") or tool.args_schema is None:
            return tool_args

        try:
            args_schema = tool.args_schema

            # Build mapping from JSON property name (alias) to field name
            alias_to_field: dict[str, str] = {}

            # Pydantic v2: Check model_fields for alias information
            if hasattr(args_schema, "model_fields"):
                for field_name, field_info in args_schema.model_fields.items():
                    # Get the alias if set, otherwise use field name
                    alias = getattr(field_info, "alias", None)
                    if alias and alias != field_name:
                        alias_to_field[alias] = field_name
                    # Also check serialization_alias and validation_alias
                    ser_alias = getattr(field_info, "serialization_alias", None)
                    val_alias = getattr(field_info, "validation_alias", None)
                    if ser_alias and ser_alias != field_name:
                        alias_to_field[ser_alias] = field_name
                    if val_alias and val_alias != field_name:
                        alias_to_field[val_alias] = field_name

            # If we found alias mappings, apply them
            if alias_to_field:
                mapped_args = {}
                for key, value in tool_args.items():
                    # Use field name if key is an alias, otherwise keep original
                    mapped_key = alias_to_field.get(key, key)
                    mapped_args[mapped_key] = value
                return mapped_args

            # No aliases found, return original args
            return tool_args

        except Exception:
            # If mapping fails, return original args
            return tool_args
