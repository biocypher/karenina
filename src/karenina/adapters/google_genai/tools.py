"""Tool conversion and execution for the google_genai agent loop.

Static ``Tool`` definitions and MCP tools become Gemini
``FunctionDeclaration``s. Gemini only accepts names matching
``^[A-Za-z_][A-Za-z0-9_.-]{0,63}$``, so names are sanitised and a name map
turns the model's calls back into the original tool. The adapter runs every
call itself (automatic function calling is off) and never lets a tool
failure escape the loop: errors go back to the model as error results.

Built-in server-side tools (Google Search, URL reading) are separate
``types.Tool`` objects, chosen per turn so the search budget can drop
Google Search once it is spent (design D5).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from karenina.ports import AgentExecutionError

from .settings import BUILTIN_GOOGLE_SEARCH, BUILTIN_URL_CONTEXT

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from karenina.ports.agent import Tool

logger = logging.getLogger(__name__)

_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]")
_MAX_NAME_LENGTH = 64
_EMPTY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def sanitize_tool_name(name: str) -> str:
    """Make a tool name acceptable to Gemini.

    Invalid characters become underscores, a leading character that is not
    a letter or underscore gets an underscore prefix, and the result is cut
    to 64 characters. Valid names are returned unchanged.

    Args:
        name: The original tool name.

    Returns:
        A name matching ``^[A-Za-z_][A-Za-z0-9_.-]{0,63}$``.
    """
    cleaned = _INVALID_NAME_CHARS.sub("_", name)
    if not cleaned or not (cleaned[0].isascii() and (cleaned[0].isalpha() or cleaned[0] == "_")):
        cleaned = f"_{cleaned}"
    return cleaned[:_MAX_NAME_LENGTH]


def truncate_tool_result(text: str, max_chars: int) -> str:
    """Cut a tool result to ``max_chars`` characters with a visible marker.

    Args:
        text: The tool result text.
        max_chars: The maximum number of characters kept.

    Returns:
        The text unchanged when short enough, otherwise its first
        ``max_chars`` characters followed by a truncation marker.
    """
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[... truncated {len(text) - max_chars} characters]"


def builtin_tools(names: Iterable[str]) -> list[Any]:
    """Build Gemini built-in tool objects.

    Args:
        names: Built-in tool names (``google_search``, ``url_context``).

    Returns:
        A list of ``types.Tool`` objects, one per name.
    """
    from google.genai import types

    result: list[Any] = []
    for name in names:
        if name == BUILTIN_GOOGLE_SEARCH:
            result.append(types.Tool(google_search=types.GoogleSearch()))
        elif name == BUILTIN_URL_CONTEXT:
            result.append(types.Tool(url_context=types.UrlContext()))
        else:
            raise ValueError(f"Unknown built-in tool: {name!r}")
    return result


@dataclass(frozen=True)
class ToolBinding:
    """One function tool exposed to the model.

    Attributes:
        original_name: The tool's real name (used for the call and the trace).
        sanitized_name: The name declared to Gemini.
        description: The description declared to Gemini.
        parameters_json_schema: The JSON Schema of the arguments.
        session: The MCP ClientSession for MCP tools, None for static tools.
        server_name: The MCP server name for MCP tools.
    """

    original_name: str
    sanitized_name: str
    description: str
    parameters_json_schema: dict[str, Any]
    session: Any = None
    server_name: str | None = None


class Toolset:
    """The function tools of one agent run, keyed by sanitised name."""

    def __init__(self, bindings: list[ToolBinding]) -> None:
        """Create a toolset.

        Args:
            bindings: The tool bindings.

        Raises:
            AgentExecutionError: If two tools share a sanitised name.
        """
        self._bindings: dict[str, ToolBinding] = {}
        for binding in bindings:
            existing = self._bindings.get(binding.sanitized_name)
            if existing is not None:
                raise AgentExecutionError(
                    f"Tools {existing.original_name!r} and {binding.original_name!r} both map to the Gemini "
                    f"function name {binding.sanitized_name!r}. Rename one of them or filter it out"
                )
            self._bindings[binding.sanitized_name] = binding

    def __bool__(self) -> bool:
        return bool(self._bindings)

    def __len__(self) -> int:
        return len(self._bindings)

    def original_name(self, sanitized_name: str) -> str:
        """Return the original tool name for a declared name (unchanged if unknown)."""
        binding = self._bindings.get(sanitized_name)
        return binding.original_name if binding is not None else sanitized_name

    def function_declaration_tool(self) -> Any | None:
        """Return one ``types.Tool`` holding every function declaration, or None."""
        if not self._bindings:
            return None
        from google.genai import types

        declarations = [
            types.FunctionDeclaration(
                name=b.sanitized_name,
                description=b.description,
                parameters_json_schema=b.parameters_json_schema,
            )
            for b in self._bindings.values()
        ]
        return types.Tool(function_declarations=declarations)

    async def execute(self, sanitized_name: str, args: Mapping[str, Any] | None) -> tuple[str, bool]:
        """Run one function call.

        Exceptions and MCP ``isError`` results become error results, never
        exceptions, so one failing tool cannot end the run.

        Args:
            sanitized_name: The function name the model called.
            args: The call arguments.

        Returns:
            Tuple of (result text, is_error).
        """
        binding = self._bindings.get(sanitized_name)
        call_args = dict(args or {})
        if binding is None:
            return f"Error: unknown tool {sanitized_name!r}. Use only the tools provided.", True
        if binding.session is None:
            # Static tools carry no executor (same behaviour as claude_tool).
            return f"Tool '{binding.original_name}' was called with arguments: {call_args}", False
        try:
            result = await binding.session.call_tool(binding.original_name, call_args)
        except Exception as e:
            logger.warning("MCP tool '%s' on server '%s' failed: %s", binding.original_name, binding.server_name, e)
            return f"Error calling MCP tool '{binding.original_name}': {type(e).__name__}: {e}", True
        return _mcp_result_text(result), bool(getattr(result, "isError", False))


def _mcp_result_text(result: Any) -> str:
    """Join the text blocks of an MCP CallToolResult."""
    texts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            texts.append(text)
        else:
            texts.append(f"[{getattr(block, 'type', 'non-text')} content omitted]")
    if not texts:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return json.dumps(structured, default=str)
    return "\n".join(texts)


def static_tool_bindings(tools: Iterable[Tool]) -> list[ToolBinding]:
    """Build bindings for static ``Tool`` definitions.

    Args:
        tools: Static tool definitions from the caller.

    Returns:
        One binding per tool.
    """
    return [
        ToolBinding(
            original_name=tool.name,
            sanitized_name=sanitize_tool_name(tool.name),
            description=tool.description,
            parameters_json_schema=dict(tool.input_schema) if tool.input_schema else dict(_EMPTY_OBJECT_SCHEMA),
        )
        for tool in tools
    ]


def mcp_tool_bindings(
    mcp_tools: Iterable[tuple[str, Any, Any]],
    *,
    tool_filter: Iterable[str] | None = None,
    description_overrides: Mapping[str, str] | None = None,
) -> list[ToolBinding]:
    """Build bindings for MCP tools.

    Args:
        mcp_tools: ``(server_name, session, mcp_tool)`` tuples from
            ``get_all_mcp_tools``.
        tool_filter: Tool names to keep, or None to keep every tool.
        description_overrides: Tool descriptions to replace, keyed by the
            original tool name.

    Returns:
        One binding per kept tool.
    """
    allowed = set(tool_filter) if tool_filter else None
    overrides = description_overrides or {}
    bindings: list[ToolBinding] = []
    for server_name, session, mcp_tool in mcp_tools:
        name = mcp_tool.name
        if allowed is not None and name not in allowed:
            logger.debug("Skipping MCP tool '%s' (not in tool filter)", name)
            continue
        schema = getattr(mcp_tool, "inputSchema", None)
        bindings.append(
            ToolBinding(
                original_name=name,
                sanitized_name=sanitize_tool_name(name),
                description=overrides.get(name) or mcp_tool.description or f"Tool from {server_name}",
                parameters_json_schema=dict(schema) if schema else dict(_EMPTY_OBJECT_SCHEMA),
                session=session,
                server_name=server_name,
            )
        )
    return bindings
