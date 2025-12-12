"""MCP (Model Context Protocol) utilities for Karenina.

This module provides utilities for integrating MCP servers with Karenina's
LLM interface, including agent response harmonization and MCP client management.
"""

import asyncio
from typing import Any


def harmonize_agent_response(response: Any, original_question: str | None = None) -> str:
    """
    Harmonize agent response messages into a single string with full trace.

    LangGraph agents return multiple messages instead of a single response.
    This function extracts AI and Tool messages (excluding system prompts and human messages)
    and returns the complete agent execution trace using pretty_print() formatting.

    Args:
        response: Response from a LangGraph agent, which may be a single message,
                 list of messages, or agent state with messages
        original_question: The original user question. If provided, enables reliable
                          detection of summary messages from SummarizationMiddleware.
                          If the first HumanMessage differs from this, it's treated
                          as a summary and included in the trace.

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

    # Handle nested agent state dict (from astream): {'agent': {'messages': [...]}}
    # This is the structure returned by LangGraph's astream method
    if isinstance(response, dict):
        # Check for nested agent state first
        if "agent" in response and isinstance(response["agent"], dict) and "messages" in response["agent"]:
            messages = response["agent"]["messages"]
            return _extract_agent_trace(messages, original_question)
        # Handle flat state dict with 'messages' key (from ainvoke)
        elif "messages" in response:
            messages = response["messages"]
            return _extract_agent_trace(messages, original_question)

    # Handle list of messages directly
    if isinstance(response, list):
        return _extract_agent_trace(response, original_question)

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


def _is_summary_message(msg: Any, original_question: str | None = None) -> bool:
    """
    Check if a message is a summarization middleware summary message.

    Detection methods (in order of reliability):
    1. If original_question is provided, check if the message content differs from it
    2. Fall back to pattern matching on known summary prefixes

    Args:
        msg: A message object to check
        original_question: The original user question (if known). If the HumanMessage
                          content doesn't match this, it's likely a summary.

    Returns:
        True if the message appears to be a summary message
    """
    if not hasattr(msg, "content") or not msg.content:
        return False

    content = str(msg.content).strip()

    # Method 1: Compare with original question (most reliable)
    if original_question is not None:
        # If content differs significantly from original question, it's a summary
        # Use strip and normalize for comparison
        original_normalized = original_question.strip()
        return content != original_normalized

    # Method 2: Fall back to pattern matching (less reliable but works without context)
    content_lower = content.lower()
    summary_markers = [
        "here is a summary of the conversation",
        "summary of the conversation to date",
        "previous conversation was too long to summarize",
        "conversation summary:",
        "summary of previous conversation",  # Our custom middleware format
    ]
    return any(marker in content_lower for marker in summary_markers)


def _extract_agent_trace(messages: list[Any], original_question: str | None = None) -> str:
    """
    Extract the complete agent trace from a list of messages.

    Filters out system messages and the first human message (initial user question),
    but preserves:
    - Subsequent human messages that may contain agent reasoning steps
    - Summary messages from SummarizationMiddleware (detected by comparing with original_question)

    Args:
        messages: List of LangChain messages
        original_question: The original user question. If provided, enables reliable
                          detection of summary messages (if first HumanMessage differs
                          from this, it's treated as a summary and included in trace).

    Returns:
        The formatted trace of AI, Tool, and intermediate Human messages (excluding first unless it's a summary), empty string if none found
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

                # Handle human messages
                if "human" in msg_type or role == "user" or role == "human":
                    # Check if this is a summary message (not the original question)
                    if _is_summary_message(msg, original_question):
                        formatted_msg = _format_message_for_trace(msg)
                        if formatted_msg.strip():
                            trace_parts.append(formatted_msg.strip())
                        first_human_found = True
                        continue

                    # Skip first human message (initial user question)
                    if not first_human_found:
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

        # Handle human messages
        if isinstance(msg, HumanMessage):
            # Check if this is a summary message (not the original question)
            if _is_summary_message(msg, original_question):
                formatted_msg = _format_message_for_trace(msg)
                if formatted_msg.strip():
                    trace_parts.append(formatted_msg.strip())
                first_human_found = True
                continue

            # Skip first human message (initial user question)
            if not first_human_found:
                first_human_found = True
                continue  # Skip the first human message

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
        client = MultiServerMCPClient(server_config)  # type: ignore[arg-type]

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


def extract_final_ai_message_from_response(response: Any) -> tuple[str | None, str | None]:
    """
    Extract only the final AI text response from an agent response (messages or dict).

    This function works with the original agent response before harmonization,
    checking message types directly rather than parsing strings.

    Args:
        response: Response from a LangGraph agent (messages list, dict with 'messages', or state dict)

    Returns:
        Tuple of (extracted_message, error_message) where:
        - extracted_message: The final AI text content, or None if extraction failed
        - error_message: Error description if extraction failed, or None if successful

    Error cases:
        - Empty or no messages
        - Last message is not an AIMessage
        - Final AIMessage has no text content (only tool calls)

    Examples:
        >>> from langchain_core.messages import AIMessage
        >>> messages = [AIMessage(content="Final answer")]
        >>> message, error = extract_final_ai_message_from_response(messages)
        >>> print(message)  # "Final answer"
    """
    try:
        from langchain_core.messages import AIMessage
    except ImportError:
        return None, "langchain_core not available"

    # Extract messages list from various response formats
    messages = None

    if response is None:
        return None, "Empty response"

    # Handle nested agent state dict: {'agent': {'messages': [...]}}
    if isinstance(response, dict):
        if "agent" in response and isinstance(response["agent"], dict) and "messages" in response["agent"]:
            messages = response["agent"]["messages"]
        elif "messages" in response:
            messages = response["messages"]
    # Handle list of messages directly
    elif isinstance(response, list):
        messages = response
    # Handle single message
    elif isinstance(response, AIMessage):
        messages = [response]

    if not messages or len(messages) == 0:
        return None, "No messages found in response"

    # Get the last message
    last_message = messages[-1]

    # Check if it's an AIMessage
    if not isinstance(last_message, AIMessage):
        return None, "Last message is not an AIMessage"

    # Extract content
    content = last_message.content if last_message.content else ""

    # Handle list content (convert to string)
    if isinstance(content, list):
        # Try to extract text from list of content blocks
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))
        content = " ".join(text_parts) if text_parts else ""

    # Check if content is empty
    if not content or not content.strip():
        # Check if there are tool calls but no text content
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return None, "Final AI message has no text content (only tool calls)"
        return None, "Final AI message has no content"

    return content.strip(), None


def extract_final_ai_message(harmonized_trace: str) -> tuple[str | None, str | None]:
    """
    Extract only the final AI text response from a harmonized agent trace string.

    This is a fallback for when only the harmonized string is available.
    Prefer using extract_final_ai_message_from_response() with the original messages.

    Args:
        harmonized_trace: The full agent trace string produced by harmonize_agent_response()

    Returns:
        Tuple of (extracted_message, error_message) where:
        - extracted_message: The final AI text content, or None if extraction failed
        - error_message: Error description if extraction failed, or None if successful
    """
    # Check for empty or whitespace-only trace
    if not harmonized_trace or not harmonized_trace.strip():
        return None, "Empty or whitespace-only trace"

    trace = harmonized_trace.strip()

    # Split trace into message blocks based on the separator pattern
    # Message blocks are separated by "--- <Type> Message ---" headers
    message_blocks = []
    current_block_type = None
    current_block_content: list[str] = []

    lines = trace.split("\n")

    for line in lines:
        # Check if this is a message header
        if line.startswith("--- ") and line.endswith(" ---"):
            # Save previous block if exists
            if current_block_type is not None:
                message_blocks.append({"type": current_block_type, "content": "\n".join(current_block_content).strip()})

            # Start new block
            current_block_type = line
            current_block_content = []
        else:
            # Add line to current block content
            if current_block_type is not None:
                current_block_content.append(line)

    # Save last block
    if current_block_type is not None:
        message_blocks.append({"type": current_block_type, "content": "\n".join(current_block_content).strip()})

    # Check if we found any message blocks
    if not message_blocks:
        return None, "Malformed trace: no message blocks found"

    # Get the last message block
    last_block = message_blocks[-1]

    # Check if the last block is an AI message
    if not last_block["type"].startswith("--- AI Message"):
        return None, "Last message in trace is not an AI message"

    # Extract content, excluding tool call information if present
    # Tool calls in AI messages appear after the main content with "Tool Calls:" header
    content = last_block["content"]

    # If there are tool calls in the message, extract only the text before them
    if "\nTool Calls:" in content:
        content = content.split("\nTool Calls:")[0].strip()
    # Also check if content starts with "Tool Calls:" (no text before)
    elif content.startswith("Tool Calls:"):
        content = ""

    # Final check: ensure we have non-empty content
    if not content:
        return None, "Final AI message has no text content (only tool calls)"

    return content, None


def cleanup_mcp_client(client: Any) -> None:
    """
    Clean up MCP client connections gracefully.

    This function attempts to close MCP client connections that might keep
    background threads or event loops alive after usage is complete.

    Args:
        client: MCP client instance (MultiServerMCPClient or similar)

    Examples:
        >>> client, tools = sync_create_mcp_client_and_tools(mcp_urls)
        >>> # ... use client ...
        >>> cleanup_mcp_client(client)
    """
    import asyncio
    import logging

    logger = logging.getLogger(__name__)

    if client is None:
        return

    try:
        # Try synchronous close first
        if hasattr(client, "close"):
            client.close()
            logger.debug("MCP client closed (sync)")
            return

        # Try async close
        if hasattr(client, "aclose"):
            try:
                # Check if we're in an async context
                loop = asyncio.get_running_loop()
                # We're in an async context - schedule close
                loop.create_task(client.aclose())
                logger.debug("MCP client close scheduled (async)")
            except RuntimeError:
                # No event loop running - run async close in new loop
                asyncio.run(client.aclose())
                logger.debug("MCP client closed (async new loop)")
            return

        # Try __exit__ if it's a context manager
        if hasattr(client, "__exit__"):
            client.__exit__(None, None, None)
            logger.debug("MCP client exited (context manager)")
            return

    except Exception as e:
        logger.warning(f"Failed to cleanup MCP client: {e}")

    # If no cleanup method found, log a warning
    logger.debug("No cleanup method found for MCP client")
