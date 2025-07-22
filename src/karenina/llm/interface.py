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

    def __init__(self, session_id: str, model: str, provider: str, temperature: float = 0.7):
        self.session_id = session_id
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.messages: list[BaseMessage] = []
        self.llm = None
        self.created_at = datetime.now()
        self.last_used = datetime.now()

    def initialize_llm(self) -> None:
        """Initialize the LLM if not already done."""
        if self.llm is None:
            self.llm = init_chat_model_unified(
                model=self.model, provider=self.provider, interface="langchain", temperature=self.temperature
            )

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


class ChatOpenRouter(ChatOpenAI):  # type: ignore[misc]
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
    **kwargs: Any,
) -> Any:
    """Initialize a chat model using the unified interface.

    This function provides a unified way to initialize different chat models
    across various interfaces (LangChain, OpenRouter, Manual) with consistent
    parameter handling.

    Args:
        model: The model name (e.g., "gemini-2.0-flash", "gpt-4", "claude-3-sonnet")
        provider: The model provider (e.g., "google_genai", "openai", "anthropic").
                 Optional for OpenRouter and Manual interfaces.
        interface: The interface to use for model initialization.
                  Supported values: "langchain", "openrouter", "manual"
        question_hash: The MD5 hash of the question (required for manual interface)
        **kwargs: Additional keyword arguments passed to the underlying model
                 initialization (e.g., temperature, max_tokens, api_key)

    Returns:
        An initialized model instance ready for inference

    Raises:
        ValueError: If an unsupported interface is specified or required args missing

    Examples:
        Initialize a Google Gemini model via LangChain:
        >>> model = init_chat_model_unified("gemini-2.0-flash", "google_genai")

        Initialize an OpenAI model via OpenRouter:
        >>> model = init_chat_model_unified("gpt-4", interface="openrouter")

        Initialize with custom temperature:
        >>> model = init_chat_model_unified("claude-3-sonnet", "anthropic", temperature=0.2)

        Initialize manual traces:
        >>> model = init_chat_model_unified("manual", interface="manual", question_hash="abc123...")
    """
    if interface == "langchain":
        return init_chat_model(model=model, model_provider=provider, **kwargs)
    elif interface == "openrouter":
        return ChatOpenRouter(model=model, **kwargs)
    elif interface == "manual":
        if question_hash is None:
            raise ValueError("question_hash is required for manual interface")
        return create_manual_llm(question_hash=question_hash, **kwargs)
    else:
        raise ValueError(f"Unsupported interface: {interface}")


def call_model(
    model: str,
    provider: str,
    message: str,
    session_id: str | None = None,
    system_message: str | None = None,
    temperature: float = 0.7,
) -> ChatResponse:
    """
    Call a language model and return the response, supporting conversational context.

    Args:
        model: The model name (e.g., "gemini-2.0-flash", "gpt-4")
        provider: The model provider (e.g., "google_genai", "openai")
        message: The user message to send
        session_id: Optional session ID for continuing a conversation
        system_message: Optional system message to set context
        temperature: Model temperature for response generation

    Returns:
        ChatResponse with the model's response and session information

    Raises:
        SessionError: If there's an error with session management
        LLMError: For other LLM-related errors
    """

    # Create new session or get existing one
    if session_id is None or session_id not in chat_sessions:
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = ChatSession(session_id, model, provider, temperature)

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
        response = session.llm.invoke(session.messages)
        response_content = response.content

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
