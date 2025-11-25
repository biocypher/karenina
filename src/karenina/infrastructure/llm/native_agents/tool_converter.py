"""Tool schema conversion for native tool-calling agents.

This module provides utilities to convert LangChain MCP tools to native
OpenAI and Anthropic tool schema formats.
"""

from __future__ import annotations

import contextlib
from typing import Any


class NativeToolConverter:
    """Converts LangChain MCP tools to native provider formats.

    LangChain tools from langchain-mcp-adapters have a specific structure:
    - tool.name: str - The tool name
    - tool.description: str - Tool description
    - tool.args_schema: Pydantic model - Input schema (optional)

    This class converts them to the format expected by OpenAI and Anthropic APIs.
    """

    @staticmethod
    def to_openai(tools: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Convert LangChain tools to OpenAI tool format.

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "Tool description",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }

        Args:
            tools: List of LangChain tools (from MCP adapters)

        Returns:
            Tuple of (openai_tools, tool_map) where:
            - openai_tools: List of tools in OpenAI format
            - tool_map: Dict mapping tool name to original LangChain tool
        """
        openai_tools: list[dict[str, Any]] = []
        tool_map: dict[str, Any] = {}

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            tool_map[tool_name] = tool

            # Extract JSON schema from Pydantic args_schema
            parameters: dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }

            if hasattr(tool, "args_schema") and tool.args_schema is not None:
                args_schema = tool.args_schema
                schema: dict[str, Any] | None = None

                # Check if args_schema is already a dict (raw JSON schema from MCP)
                if isinstance(args_schema, dict):
                    schema = args_schema
                # Check if it's a Pydantic model
                elif hasattr(args_schema, "model_json_schema"):
                    with contextlib.suppress(Exception):
                        # Use by_alias=False to ensure property names match field names
                        # This prevents mismatch between schema (aliases) and validation (field names)
                        schema = args_schema.model_json_schema(by_alias=False)

                if schema:
                    parameters = {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", []),
                    }

                    # Handle $defs if present (nested schemas)
                    if "$defs" in schema:
                        parameters["$defs"] = schema["$defs"]

            description = getattr(tool, "description", "") or ""

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
            )

        return openai_tools, tool_map

    @staticmethod
    def to_anthropic(tools: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Convert LangChain tools to Anthropic tool format.

        Anthropic format:
        {
            "name": "tool_name",
            "description": "Tool description",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }

        Args:
            tools: List of LangChain tools (from MCP adapters)

        Returns:
            Tuple of (anthropic_tools, tool_map) where:
            - anthropic_tools: List of tools in Anthropic format
            - tool_map: Dict mapping tool name to original LangChain tool
        """
        anthropic_tools: list[dict[str, Any]] = []
        tool_map: dict[str, Any] = {}

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            tool_map[tool_name] = tool

            # Extract JSON schema from Pydantic args_schema
            input_schema: dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }

            if hasattr(tool, "args_schema") and tool.args_schema is not None:
                args_schema = tool.args_schema
                schema: dict[str, Any] | None = None

                # Check if args_schema is already a dict (raw JSON schema from MCP)
                if isinstance(args_schema, dict):
                    schema = args_schema
                # Check if it's a Pydantic model
                elif hasattr(args_schema, "model_json_schema"):
                    with contextlib.suppress(Exception):
                        # Use by_alias=False to ensure property names match field names
                        # This prevents mismatch between schema (aliases) and validation (field names)
                        schema = args_schema.model_json_schema(by_alias=False)

                if schema:
                    input_schema = {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", []),
                    }

                    # Handle $defs if present (nested schemas)
                    # Anthropic expects these to be inlined or referenced properly
                    if "$defs" in schema:
                        input_schema["$defs"] = schema["$defs"]

            description = getattr(tool, "description", "") or ""

            anthropic_tools.append(
                {
                    "name": tool_name,
                    "description": description,
                    "input_schema": input_schema,
                }
            )

        return anthropic_tools, tool_map

    @staticmethod
    def get_tool_names(tools: list[Any]) -> list[str]:
        """Extract tool names from a list of LangChain tools.

        Args:
            tools: List of LangChain tools

        Returns:
            List of tool names
        """
        return [getattr(tool, "name", str(tool)) for tool in tools]
