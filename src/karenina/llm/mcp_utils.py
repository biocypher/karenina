"""MCP (Model Context Protocol) utilities for Karenina.

This module provides utilities for integrating MCP servers with Karenina's
LLM interface, including agent response harmonization and MCP client management.
"""

import asyncio
from typing import Any


def harmonize_agent_response(response: Any) -> str:
    """
    Harmonize agent response messages into a single string with full trace.

    LangGraph agents return multiple messages instead of a single response.
    This function extracts AI and Tool messages (excluding system prompts and human messages)
    and returns the complete agent execution trace using pretty_print() formatting.

    Args:
        response: Response from a LangGraph agent, which may be a single message,
                 list of messages, or agent state with messages

    Returns:
        A single string containing the complete agent trace with reasoning and tool usage

    Examples:
        >>> from langchain_core.messages import AIMessage, ToolMessage
        >>> messages = [
        ...     AIMessage(content="I need to search for information."),
        ...     ToolMessage(content="Search results: ...", tool_call_id="call_123"),
        ...     AIMessage(content="Based on the search, the answer is 42.")
        ... ]
        >>> trace = harmonize_agent_response({"messages": messages})
        >>> print("Full agent trace with reasoning and tool usage")
    """
    if response is None:
        return ""

    # Handle single message with content attribute
    if hasattr(response, "content"):
        return str(response.content)

    # Handle agent state dict with 'messages' key
    if isinstance(response, dict) and "messages" in response:
        messages = response["messages"]
        return _extract_agent_trace(messages)

    # Handle list of messages directly
    if isinstance(response, list):
        return _extract_agent_trace(response)

    # Fallback: convert to string
    return str(response)


def _format_message_for_trace(msg: Any) -> str:
    """
    Format a single message for trace output with Excel-friendly separators.

    Uses simple dashes instead of equal signs to avoid Excel formula confusion.

    Args:
        msg: A LangChain message object (AIMessage, ToolMessage, or HumanMessage)

    Returns:
        Formatted message string with header and content
    """
    try:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    except ImportError:
        # Fallback for non-LangChain messages
        msg_type = type(msg).__name__
        content = str(getattr(msg, "content", msg))
        return f"--- {msg_type} ---\n{content}"

    # Format based on message type
    if isinstance(msg, AIMessage):
        header = "--- AI Message ---"
        content_parts = []

        # Add main content if present
        if msg.content:
            content_parts.append(str(msg.content))

        # Add tool calls if present
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            content_parts.append("\nTool Calls:")
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "unknown")
                tool_id = tool_call.get("id", "unknown")
                tool_args = tool_call.get("args", {})
                content_parts.append(f"  {tool_name} (call_{tool_id})")
                content_parts.append(f"   Call ID: {tool_id}")
                if tool_args:
                    content_parts.append(f"   Args: {tool_args}")

        content = "\n".join(content_parts) if content_parts else ""

    elif isinstance(msg, ToolMessage):
        tool_call_id = getattr(msg, "tool_call_id", "unknown")
        header = f"--- Tool Message (call_id: {tool_call_id}) ---"
        content = str(msg.content) if msg.content else ""

    elif isinstance(msg, HumanMessage):
        header = "--- Human Message ---"
        content = str(msg.content) if msg.content else ""

    else:
        # Fallback for unknown message types
        msg_type = type(msg).__name__
        header = f"--- {msg_type} ---"
        content = str(getattr(msg, "content", msg))

    return f"{header}\n{content}" if content else header


def _extract_agent_trace(messages: list[Any]) -> str:
    """
    Extract the complete agent trace from a list of messages.

    Filters out system messages and the first human message (initial user question),
    but preserves subsequent human messages that may contain agent reasoning steps.

    Args:
        messages: List of LangChain messages

    Returns:
        The formatted trace of AI, Tool, and intermediate Human messages (excluding first), empty string if none found
    """
    # Import here to avoid circular imports
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    except ImportError:
        # Fallback: look for messages with type or role indicating AI, Tool, or Human messages
        trace_parts = []
        first_human_found = False

        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                # Check message type
                msg_type = type(msg).__name__.lower()
                role = getattr(msg, "role", "").lower() if hasattr(msg, "role") else ""

                # Skip system messages
                if "system" in msg_type or role == "system":
                    continue

                # Skip first human message (initial user question)
                if ("human" in msg_type or role == "user" or role == "human") and not first_human_found:
                    first_human_found = True
                    continue  # Skip the first human message
                # Continue processing subsequent human messages

                # Include AI, Tool, and subsequent Human messages
                if (
                    "ai" in msg_type
                    or role == "assistant"
                    or "tool" in msg_type
                    or role == "tool"
                    or "human" in msg_type
                    or role == "user"
                    or role == "human"
                ):
                    formatted_msg = _format_message_for_trace(msg)
                    if formatted_msg.strip():
                        trace_parts.append(formatted_msg.strip())

        return "\n\n".join(trace_parts) if trace_parts else ""

    # Extract AI, Tool, and intermediate Human messages (skip first Human and all System messages)
    trace_parts = []
    first_human_found = False

    for msg in messages:
        # Skip system messages
        if isinstance(msg, SystemMessage):
            continue

        # Skip first human message (initial user question)
        if isinstance(msg, HumanMessage) and not first_human_found:
            first_human_found = True
            continue  # Skip the first human message
        # Continue processing subsequent human messages

        # Include AI, Tool, and subsequent Human messages
        if isinstance(msg, AIMessage | ToolMessage | HumanMessage):
            formatted_msg = _format_message_for_trace(msg)
            if formatted_msg.strip():
                trace_parts.append(formatted_msg.strip())

    return "\n\n".join(trace_parts) if trace_parts else ""


async def create_mcp_client_and_tools(
    mcp_urls_dict: dict[str, str], tool_filter: list[str] | None = None
) -> tuple[Any, list[Any]]:
    """
    Create an MCP client and fetch tools from the specified servers.

    Args:
        mcp_urls_dict: Dictionary mapping tool names to MCP server URLs.
                      Keys are tool names, values are server URLs.
        tool_filter: Optional list of tool names to include. If provided,
                    only tools with names in this list will be returned.
                    Converted to set internally for efficiency.

    Returns:
        Tuple of (client, tools) where:
        - client: MultiServerMCPClient instance
        - tools: List of LangChain-compatible tools fetched from servers,
                 optionally filtered by tool_filter

    Raises:
        ImportError: If langchain-mcp-adapters is not installed
        Exception: If MCP client creation or tool fetching fails

    Examples:
        >>> mcp_urls = {"biocontext": "https://mcp.biocontext.ai/mcp/"}
        >>> client, tools = await create_mcp_client_and_tools(mcp_urls)
        >>> print(f"Loaded {len(tools)} tools")

        >>> # Filter to specific tools
        >>> client, tools = await create_mcp_client_and_tools(mcp_urls, ["search_proteins", "get_interactions"])
        >>> print(f"Loaded {len(tools)} filtered tools")
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as e:
        raise ImportError(
            "langchain-mcp-adapters package is required for MCP support. Install it with: uv add langchain-mcp-adapters"
        ) from e

    # Configure servers with streamable_http transport
    server_config = {}
    for tool_name, url in mcp_urls_dict.items():
        server_config[tool_name] = {"transport": "streamable_http", "url": url}

    try:
        # Create client and fetch tools with timeout
        client = MultiServerMCPClient(server_config)

        # Add timeout to prevent hanging
        tools = await asyncio.wait_for(client.get_tools(), timeout=30.0)

        # Filter tools if tool_filter is provided
        if tool_filter is not None:
            allowed_tools = set(tool_filter)
            filtered_tools = []
            for tool in tools:
                # Check if tool has a name attribute and if it's in the allowed set
                current_tool_name: str | None = getattr(tool, "name", None)
                if current_tool_name and current_tool_name in allowed_tools:
                    filtered_tools.append(tool)
            tools = filtered_tools

        return client, tools

    except TimeoutError as e:
        raise Exception(
            "MCP server connection timed out after 30 seconds. Check server availability and network connection."
        ) from e
    except Exception as e:
        raise Exception(f"Failed to create MCP client or fetch tools: {e}") from e


def sync_create_mcp_client_and_tools(
    mcp_urls_dict: dict[str, str], tool_filter: list[str] | None = None
) -> tuple[Any, list[Any]]:
    """
    Synchronous wrapper for creating MCP client and fetching tools.

    This function runs the async MCP client creation in a new event loop,
    making it usable from synchronous code.

    Args:
        mcp_urls_dict: Dictionary mapping tool names to MCP server URLs
        tool_filter: Optional list of tool names to include. If provided,
                    only tools with names in this list will be returned.

    Returns:
        Tuple of (client, tools) as in create_mcp_client_and_tools

    Raises:
        Same exceptions as create_mcp_client_and_tools
    """
    import threading

    # Check if we're already in an async context
    try:
        current_loop = asyncio.get_running_loop()
        if current_loop:
            # We're in an async context, need to run in a separate thread
            def run_in_thread() -> tuple[Any, list[Any]]:
                return asyncio.run(create_mcp_client_and_tools(mcp_urls_dict, tool_filter))

            result: list[tuple[Any, list[Any]] | None] = [None]
            exception: list[Exception | None] = [None]

            def thread_target() -> None:
                try:
                    result[0] = run_in_thread()
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=thread_target)
            thread.start()
            thread.join(timeout=45.0)  # 45 second total timeout

            if thread.is_alive():
                raise Exception("MCP client creation timed out after 45 seconds")

            if exception[0]:
                raise exception[0]

            if result[0] is None:
                raise Exception("MCP client creation failed without exception")

            return result[0]
    except RuntimeError:
        # No event loop running, safe to create new one
        pass

    # Create new event loop and run the async function
    return asyncio.run(create_mcp_client_and_tools(mcp_urls_dict, tool_filter))
