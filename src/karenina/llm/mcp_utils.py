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


def _extract_agent_trace(messages: list[Any]) -> str:
    """
    Extract the complete agent trace from a list of messages.

    Args:
        messages: List of LangChain messages

    Returns:
        The pretty-printed trace of all AI and Tool messages, empty string if none found
    """
    # Import here to avoid circular imports
    try:
        from langchain_core.messages import AIMessage, ToolMessage
    except ImportError:
        # Fallback: look for messages with type or role indicating AI or Tool messages
        trace_parts = []
        for msg in messages:
            if hasattr(msg, "content") and msg.content:
                # Check if this looks like an AI or Tool message by type name
                msg_type = type(msg).__name__.lower()
                role = getattr(msg, "role", "").lower() if hasattr(msg, "role") else ""

                if "ai" in msg_type or role == "assistant" or "tool" in msg_type or role == "tool":
                    # Try to use pretty_print if available, otherwise use content
                    if hasattr(msg, "pretty_print"):
                        try:
                            import io
                            import sys

                            # Capture stdout for pretty_print
                            old_stdout = sys.stdout
                            sys.stdout = captured_output = io.StringIO()

                            msg.pretty_print()

                            pretty_output = captured_output.getvalue()
                            sys.stdout = old_stdout

                            if pretty_output.strip():
                                trace_parts.append(pretty_output.strip())

                        except Exception:
                            trace_parts.append(str(msg.content))
                        finally:
                            sys.stdout = old_stdout
                    else:
                        trace_parts.append(str(msg.content))

        return "\n".join(trace_parts) if trace_parts else ""

    # Extract AI and Tool messages and pretty print them
    trace_parts = []
    for msg in messages:
        if isinstance(msg, AIMessage | ToolMessage):
            try:
                # Capture pretty_print output using StringIO
                import io
                import sys

                # Capture stdout
                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()

                # Call pretty_print which prints to stdout
                msg.pretty_print()

                # Get the captured output
                pretty_output = captured_output.getvalue()

                # Restore stdout
                sys.stdout = old_stdout

                if pretty_output.strip():
                    trace_parts.append(pretty_output.strip())

            except Exception:
                # Fallback to content if pretty_print fails
                if hasattr(msg, "content") and msg.content:
                    trace_parts.append(str(msg.content))
            finally:
                # Ensure stdout is always restored
                sys.stdout = old_stdout

    return "\n".join(trace_parts) if trace_parts else ""


async def create_mcp_client_and_tools(mcp_urls_dict: dict[str, str]) -> tuple[Any, list[Any]]:
    """
    Create an MCP client and fetch tools from the specified servers.

    Args:
        mcp_urls_dict: Dictionary mapping tool names to MCP server URLs.
                      Keys are tool names, values are server URLs.

    Returns:
        Tuple of (client, tools) where:
        - client: MultiServerMCPClient instance
        - tools: List of LangChain-compatible tools fetched from servers

    Raises:
        ImportError: If langchain-mcp-adapters is not installed
        Exception: If MCP client creation or tool fetching fails

    Examples:
        >>> mcp_urls = {"biocontext": "https://mcp.biocontext.ai/mcp/"}
        >>> client, tools = await create_mcp_client_and_tools(mcp_urls)
        >>> print(f"Loaded {len(tools)} tools")
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

        return client, tools

    except TimeoutError as e:
        raise Exception(
            "MCP server connection timed out after 30 seconds. Check server availability and network connection."
        ) from e
    except Exception as e:
        raise Exception(f"Failed to create MCP client or fetch tools: {e}") from e


def sync_create_mcp_client_and_tools(mcp_urls_dict: dict[str, str]) -> tuple[Any, list[Any]]:
    """
    Synchronous wrapper for creating MCP client and fetching tools.

    This function runs the async MCP client creation in a new event loop,
    making it usable from synchronous code.

    Args:
        mcp_urls_dict: Dictionary mapping tool names to MCP server URLs

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
                return asyncio.run(create_mcp_client_and_tools(mcp_urls_dict))

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
    return asyncio.run(create_mcp_client_and_tools(mcp_urls_dict))
