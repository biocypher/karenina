"""LLM interface and session management functionality.

This module provides a unified interface for calling language models,
managing conversation sessions, and handling LLM-related operations.
"""

import os
import uuid
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from .manual_llm import create_manual_llm

load_dotenv()


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class LLMNotAvailableError(LLMError):
    """Raised when LangChain is not available."""

    pass


class SessionError(LLMError):
    """Raised when there's an error with session management."""

    pass


class ChatRequest(BaseModel):
    """Request model for chat API."""

    model: str
    provider: str
    message: str
    session_id: str | None = None
    system_message: str | None = None
    temperature: float | None = 0.7


class ChatResponse(BaseModel):
    """Response model for chat API."""

    session_id: str
    message: str
    model: str
    provider: str
    timestamp: str


class ChatSession:
    """Manages a conversation session with an LLM."""

    def __init__(
        self,
        session_id: str,
        model: str,
        provider: str,
        temperature: float = 0.7,
        mcp_urls_dict: dict[str, str] | None = None,
        mcp_tool_filter: list[str] | None = None,
    ):
        self.session_id = session_id
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.mcp_urls_dict = mcp_urls_dict
        self.mcp_tool_filter = mcp_tool_filter
        self.messages: list[BaseMessage] = []
        self.llm = None
        self.is_agent = False  # Track if LLM is actually a LangGraph agent
        self.created_at = datetime.now()
        self.last_used = datetime.now()

    def initialize_llm(self) -> None:
        """Initialize the LLM if not already done."""
        if self.llm is None:
            self.llm = init_chat_model_unified(
                model=self.model,
                provider=self.provider,
                interface="langchain",
                temperature=self.temperature,
                mcp_urls_dict=self.mcp_urls_dict,
                mcp_tool_filter=self.mcp_tool_filter,
            )
            # Check if we got an agent by looking for 'invoke' vs 'stream' methods
            # Agents typically have additional methods like 'stream' for state management
            self.is_agent = self.mcp_urls_dict is not None

    def add_message(self, message: str, is_human: bool = True) -> None:
        """Add a message to the conversation history."""
        if is_human:
            self.messages.append(HumanMessage(content=message))
        else:
            self.messages.append(AIMessage(content=message))
        self.last_used = datetime.now()

    def add_system_message(self, message: str) -> None:
        """Add a system message to the conversation."""
        # Insert system message at the beginning if it doesn't exist
        if not self.messages or not isinstance(self.messages[0], SystemMessage):
            self.messages.insert(0, SystemMessage(content=message))
        else:
            # Update existing system message
            self.messages[0] = SystemMessage(content=message)


# Global chat session storage
chat_sessions: dict[str, ChatSession] = {}


class ChatOpenRouter(ChatOpenAI):
    openai_api_key: SecretStr | None = Field(alias="api_key", default=None)

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "OPENROUTER_API_KEY"}

    def __init__(self, openai_api_key: str | None = None, **kwargs: Any) -> None:
        openai_api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(openai_api_key) if openai_api_key else None,
            **kwargs,
        )


def init_chat_model_unified(
    model: str,
    provider: str | None = None,
    interface: str = "langchain",
    question_hash: str | None = None,
    mcp_urls_dict: dict[str, str] | None = None,
    mcp_tool_filter: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Initialize a chat model using the unified interface.

    This function provides a unified way to initialize different chat models
    across various interfaces (LangChain, OpenRouter, Manual) with consistent
    parameter handling. When MCP URLs are provided, creates a LangGraph agent
    with tools from MCP servers.

    Args:
        model: The model name (e.g., "gemini-2.0-flash", "gpt-4.1-mini", "claude-3-sonnet")
        provider: The model provider (e.g., "google_genai", "openai", "anthropic").
                 Optional for OpenRouter and Manual interfaces.
        interface: The interface to use for model initialization.
                  Supported values: "langchain", "openrouter", "manual"
        question_hash: The MD5 hash of the question (required for manual interface)
        mcp_urls_dict: Dictionary mapping tool names to MCP server URLs.
                      When provided, creates a LangGraph agent with MCP tools.
                      Keys are tool names, values are server URLs.
                      Not supported with manual interface.
        mcp_tool_filter: Optional list of tool names to include from MCP servers.
                        If provided, only tools with names in this list will be used.
                        Ignored if mcp_urls_dict is None.
        **kwargs: Additional keyword arguments passed to the underlying model
                 initialization (e.g., temperature, max_tokens, api_key)

    Returns:
        An initialized model instance or LangGraph agent ready for inference

    Raises:
        ValueError: If an unsupported interface is specified or required args missing
        ImportError: If langchain-mcp-adapters is not installed when MCP URLs provided
        Exception: If MCP client creation or agent initialization fails

    Examples:
        Initialize a Google Gemini model via LangChain:
        >>> model = init_chat_model_unified("gemini-2.0-flash", "google_genai")

        Initialize an OpenAI model via OpenRouter:
        >>> model = init_chat_model_unified("gpt-4.1-mini", interface="openrouter")

        Initialize with MCP tools:
        >>> mcp_urls = {"biocontext": "https://mcp.biocontext.ai/mcp/"}
        >>> agent = init_chat_model_unified("gpt-4.1-mini", "openai", mcp_urls_dict=mcp_urls)

        Initialize with filtered MCP tools:
        >>> mcp_urls = {"biocontext": "https://mcp.biocontext.ai/mcp/"}
        >>> tools_filter = ["search_proteins", "get_interactions"]
        >>> agent = init_chat_model_unified("gpt-4.1-mini", "openai", mcp_urls_dict=mcp_urls, mcp_tool_filter=tools_filter)

        Initialize with custom temperature:
        >>> model = init_chat_model_unified("claude-3-sonnet", "anthropic", temperature=0.2)

        Initialize manual traces:
        >>> model = init_chat_model_unified("manual", interface="manual", question_hash="abc123...")
    """
    # Check for MCP with manual interface (not supported)
    if mcp_urls_dict is not None and interface == "manual":
        raise ValueError("MCP integration is not supported with manual interface")

    # Initialize base model first
    if interface == "langchain":
        base_model = init_chat_model(model=model, model_provider=provider, **kwargs)
    elif interface == "openrouter":
        base_model = ChatOpenRouter(model=model, **kwargs)
    elif interface == "manual":
        if question_hash is None:
            raise ValueError("question_hash is required for manual interface")
        return create_manual_llm(question_hash=question_hash, **kwargs)
    else:
        raise ValueError(f"Unsupported interface: {interface}")

    # If no MCP URLs provided, return base model
    if mcp_urls_dict is None:
        return base_model

    # Create LangGraph agent with MCP tools
    try:
        from langgraph.prebuilt import create_react_agent

        from .mcp_utils import sync_create_mcp_client_and_tools
    except ImportError as e:
        raise ImportError(
            "langgraph and langchain-mcp-adapters are required for MCP support. "
            "Install with: uv add langgraph langchain-mcp-adapters"
        ) from e

    try:
        # Get MCP client and tools
        _, tools = sync_create_mcp_client_and_tools(mcp_urls_dict, mcp_tool_filter)

        # Create React agent with base model and MCP tools
        agent = create_react_agent(base_model, tools)

        return agent

    except Exception as e:
        raise Exception(f"Failed to create MCP-enabled agent: {e}") from e


def call_model(
    model: str,
    provider: str,
    message: str,
    session_id: str | None = None,
    system_message: str | None = None,
    temperature: float = 0.7,
    mcp_urls_dict: dict[str, str] | None = None,
    mcp_tool_filter: list[str] | None = None,
) -> ChatResponse:
    """
    Call a language model and return the response, supporting conversational context.

    Args:
        model: The model name (e.g., "gemini-2.0-flash", "gpt-4.1-mini")
        provider: The model provider (e.g., "google_genai", "openai")
        message: The user message to send
        session_id: Optional session ID for continuing a conversation
        system_message: Optional system message to set context
        temperature: Model temperature for response generation
        mcp_urls_dict: Optional dictionary mapping tool names to MCP server URLs
        mcp_tool_filter: Optional list of tool names to include from MCP servers

    Returns:
        ChatResponse with the model's response and session information

    Raises:
        SessionError: If there's an error with session management
        LLMError: For other LLM-related errors
    """

    # Create new session or get existing one
    if session_id is None or session_id not in chat_sessions:
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = ChatSession(
            session_id, model, provider, temperature, mcp_urls_dict, mcp_tool_filter
        )

    session = chat_sessions[session_id]

    # Validate that model and provider match for existing sessions
    if session.model != model or session.provider != provider:
        raise SessionError(
            f"Session {session_id} is configured for {session.provider}:{session.model}, "
            f"but request is for {provider}:{model}"
        )

    try:
        # Initialize LLM if needed
        session.initialize_llm()

        # Add system message if provided
        if system_message:
            session.add_system_message(system_message)

        # Add user message to conversation
        session.add_message(message, is_human=True)

        # Get response from model
        if session.llm is None:
            raise ValueError("LLM not initialized")

        # Handle agent vs regular LLM invocation
        if session.is_agent:
            # LangGraph agents with MCP tools need async invocation
            import asyncio

            recursion_limit_reached = False

            async def invoke_agent_async():
                nonlocal recursion_limit_reached
                try:
                    return await session.llm.ainvoke({"messages": session.messages})
                except Exception as e:
                    # Check if this is a GraphRecursionError
                    if "GraphRecursionError" in str(type(e).__name__) or "recursion_limit" in str(e).lower():
                        recursion_limit_reached = True
                        # Try to extract partial state from the agent
                        try:
                            agent_state = session.llm.get_state({"messages": session.messages})
                            return agent_state
                        except Exception:
                            # If we can't get state, return the messages we have so far
                            return {"messages": session.messages}
                    else:
                        raise e

            # Run the async invocation in the event loop
            try:
                asyncio.get_running_loop()
                # We're in an async context, use ThreadPoolExecutor
                import concurrent.futures

                def run_in_thread():
                    return asyncio.run(invoke_agent_async())

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    response = future.result(timeout=60)  # 60 second timeout

            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                response = asyncio.run(invoke_agent_async())

            from .mcp_utils import harmonize_agent_response

            response_content = harmonize_agent_response(response)

            # Add note if recursion limit was reached
            if recursion_limit_reached:
                response_content += "\n\n[Note: Recursion limit reached - partial response shown]"
        else:
            # Regular LLMs expect the messages list directly
            response = session.llm.invoke(session.messages)
            response_content = response.content if hasattr(response, "content") else str(response)

        # Add AI response to conversation
        session.add_message(response_content, is_human=False)

        return ChatResponse(
            session_id=session_id,
            message=response_content,
            model=model,
            provider=provider,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise LLMError(f"Error calling model {provider}:{model}: {e!s}") from e


def get_session(session_id: str) -> ChatSession | None:
    """Get a chat session by ID."""
    return chat_sessions.get(session_id)


def list_sessions() -> list[dict[str, Any]]:
    """List all active chat sessions."""
    return [
        {
            "session_id": session.session_id,
            "model": session.model,
            "provider": session.provider,
            "created_at": session.created_at.isoformat(),
            "last_used": session.last_used.isoformat(),
            "message_count": len([msg for msg in session.messages if not isinstance(msg, SystemMessage)]),
        }
        for session in chat_sessions.values()
    ]


def delete_session(session_id: str) -> bool:
    """Delete a chat session."""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return True
    return False


def clear_all_sessions() -> None:
    """Clear all chat sessions."""
    global chat_sessions
    chat_sessions = {}
