"""LLM interface and session management functionality.

This module provides a unified interface for calling language models,
managing conversation sessions, and handling LLM-related operations.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from .manual_llm import create_manual_llm

if TYPE_CHECKING:
    from karenina.schemas.workflow.models import AgentMiddlewareConfig

load_dotenv()

logger = logging.getLogger(__name__)


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
    interface: str | None = None
    endpoint_base_url: str | None = None
    endpoint_api_key: str | None = None


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
        interface: str = "langchain",
        endpoint_base_url: str | None = None,
        endpoint_api_key: str | None = None,
    ):
        self.session_id = session_id
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.mcp_urls_dict = mcp_urls_dict
        self.mcp_tool_filter = mcp_tool_filter
        self.interface = interface
        self.endpoint_base_url = endpoint_base_url
        self.endpoint_api_key = endpoint_api_key
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
                interface=self.interface,
                temperature=self.temperature,
                mcp_urls_dict=self.mcp_urls_dict,
                mcp_tool_filter=self.mcp_tool_filter,
                endpoint_base_url=self.endpoint_base_url,
                endpoint_api_key=self.endpoint_api_key,
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


class ChatOpenAIEndpoint(ChatOpenAI):
    """ChatOpenAI wrapper for user-provided custom endpoints.

    Unlike ChatOpenRouter, this does NOT automatically read from environment.
    API key must be explicitly provided by the user.
    """

    openai_api_key: SecretStr | None = Field(alias="api_key", default=None)

    @property
    def lc_secrets(self) -> dict[str, str]:
        # Return empty dict - we don't want LangChain trying to read from env
        return {}

    def __init__(
        self,
        base_url: str,
        openai_api_key: str | SecretStr | None = None,
        **kwargs: Any,
    ) -> None:
        # Do NOT fallback to environment - require explicit API key
        if openai_api_key is None:
            raise ValueError(
                "API key is required for openai_endpoint interface. "
                "This interface does not automatically read from environment variables."
            )

        if isinstance(openai_api_key, str):
            openai_api_key = SecretStr(openai_api_key)

        super().__init__(
            base_url=base_url,
            api_key=openai_api_key,
            **kwargs,
        )


def _create_invoke_summarization_middleware(
    model: Any,
    trigger_tokens: int,
    keep_messages: int,
) -> Any:
    """
    Create a custom middleware that summarizes conversation WITHIN a single invoke.

    Unlike built-in SummarizationMiddleware which only triggers between invokes,
    this uses the before_model hook to check and summarize before each model call.

    Uses LangGraph's RemoveMessage to properly remove old messages from state,
    since the default message reducer appends rather than replaces.

    Args:
        model: The LLM to use for generating summaries
        trigger_tokens: Number of approximate tokens to trigger summarization
        keep_messages: Number of recent messages to keep after summarization

    Returns:
        An InvokeSummarizationMiddleware instance
    """
    # Import middleware classes inside function to avoid module-level import issues
    try:
        from langchain.agents import AgentState
        from langchain.agents.middleware import AgentMiddleware
        from langchain_core.messages import RemoveMessage
    except ImportError as e:
        raise ImportError(
            "langchain>=1.1.0 and langgraph are required for middleware support. "
            "Install with: uv add 'langchain>=1.1.0' langgraph"
        ) from e

    class InvokeSummarizationMiddleware(AgentMiddleware):
        """
        Custom middleware that summarizes conversation WITHIN a single invoke.

        Unlike built-in SummarizationMiddleware which only triggers between invokes,
        this uses the before_model hook to check and summarize before each model call.

        Uses RemoveMessage operations to properly remove summarized messages from
        LangGraph state, since the default message reducer appends rather than replaces.
        """

        def __init__(
            self,
            summarization_model: Any,
            trigger_token_count: int = 4000,
            keep_message_count: int = 5,
        ):
            super().__init__()
            self.model = summarization_model
            self.trigger_tokens = trigger_token_count
            self.keep_messages = keep_message_count

        def _count_tokens(self, messages: list[Any]) -> int:
            """Approximate token count (4 chars ≈ 1 token)."""
            total_chars = sum(len(str(m.content)) for m in messages if hasattr(m, "content"))
            return total_chars // 4

        def _extract_original_question(self, messages: list[Any]) -> str | None:
            """Extract the original user question from messages."""
            from langchain_core.messages import HumanMessage as LCHumanMessage
            from langchain_core.messages import SystemMessage

            for msg in messages:
                # Skip system messages
                if isinstance(msg, SystemMessage):
                    continue
                # First non-system message with content is likely the question
                if isinstance(msg, LCHumanMessage) and hasattr(msg, "content") and msg.content:
                    return str(msg.content)
            return None

        def _summarize_messages(self, messages: list[Any], original_question: str | None = None) -> str:
            """Use the model to summarize older messages, preserving critical context."""
            # Format messages with more detail for tool messages
            formatted_parts = []
            for msg in messages:
                msg_type = type(msg).__name__
                content = str(getattr(msg, "content", ""))

                # For tool messages, try to preserve key data points
                if "Tool" in msg_type:
                    # Truncate but keep more for tool responses (they contain important data)
                    formatted_parts.append(f"{msg_type}: {content[:1000]}")
                else:
                    formatted_parts.append(f"{msg_type}: {content[:500]}")

            messages_text = "\n\n".join(formatted_parts)

            # Build context-aware prompt
            question_context = ""
            if original_question:
                question_context = f"""
ORIGINAL QUESTION: {original_question}

"""

            summary_prompt = f"""You are summarizing a conversation to preserve context for an ongoing task.
{question_context}CONVERSATION TO SUMMARIZE:
{messages_text}

INSTRUCTIONS:
Create a concise but information-rich summary that preserves:
1. The original question/goal being addressed
2. Key data and results from any tool calls (specific values, IDs, names)
3. Important reasoning steps and conclusions reached
4. Any errors encountered and how they were handled
5. The current state of progress toward answering the question

Keep the summary focused and factual. Do not include unnecessary pleasantries or meta-commentary.

SUMMARY:"""

            response = self.model.invoke([HumanMessage(content=summary_prompt)])
            return str(response.content)

        def before_model(self, state: AgentState[Any], runtime: Any) -> dict[str, Any] | None:  # noqa: ARG002
            """Check token count before each model call and summarize if needed.

            Uses RemoveMessage to properly delete old messages from LangGraph state,
            then adds a summary message. This ensures the trace only contains the
            summarized version, not the original long conversation.
            """
            messages = state.get("messages", [])

            current_tokens = self._count_tokens(messages)
            logger.debug(f"[InvokeSummarization] Token count: ~{current_tokens}, threshold: {self.trigger_tokens}")

            if current_tokens >= self.trigger_tokens and len(messages) > self.keep_messages:
                logger.info(
                    f"[InvokeSummarization] TRIGGERING SUMMARIZATION! "
                    f"Tokens: ~{current_tokens} >= {self.trigger_tokens}"
                )

                messages_to_summarize = messages[: -self.keep_messages]
                messages_to_keep = messages[-self.keep_messages :]

                # Extract original question for context-aware summarization
                original_question = self._extract_original_question(messages)

                summary_text = self._summarize_messages(messages_to_summarize, original_question)
                summary_message = HumanMessage(content=f"Summary of previous conversation:\n{summary_text}")

                # Create RemoveMessage operations for ALL messages, then re-add in correct order
                # This ensures summary comes FIRST, followed by kept messages
                # LangGraph's reducer appends, so we need to remove everything and rebuild
                remove_ops = []
                msgs_without_ids = 0

                # Remove ALL messages (both summarized and kept ones)
                for msg in messages:
                    msg_id = getattr(msg, "id", None)
                    if msg_id:
                        remove_ops.append(RemoveMessage(id=msg_id))
                    else:
                        msgs_without_ids += 1
                        logger.warning(
                            f"[InvokeSummarization] Message without ID cannot be removed: {type(msg).__name__}"
                        )

                if msgs_without_ids > 0:
                    logger.warning(
                        f"[InvokeSummarization] {msgs_without_ids}/{len(messages)} "
                        f"messages lack IDs and cannot be removed from state"
                    )

                # Return: remove ALL messages, then add [summary, kept_messages] in correct order
                # This ensures trace starts with summary
                state_updates = remove_ops + [summary_message] + list(messages_to_keep)
                logger.info(
                    f"[InvokeSummarization] Summarized {len(messages_to_summarize)} messages, "
                    f"keeping {len(messages_to_keep)} recent messages, "
                    f"created {len(remove_ops)} RemoveMessage ops"
                )

                return {"messages": state_updates}

            return None

    return InvokeSummarizationMiddleware(
        summarization_model=model,
        trigger_token_count=trigger_tokens,
        keep_message_count=keep_messages,
    )


def _build_agent_middleware(
    config: AgentMiddlewareConfig | None,
    max_context_tokens: int | None = None,
    interface: str = "langchain",
    base_model: Any = None,
    provider: str | None = None,
) -> list[Any]:
    """
    Build middleware list from configuration.

    Args:
        config: Middleware configuration (uses defaults if None)
        max_context_tokens: Maximum context tokens for the model.
            For langchain interface, fraction-based triggering is used (auto-detected from model).
            For openrouter/openai_endpoint, uses absolute token count (defaults to 100000).
        interface: The interface type (langchain, openrouter, openai_endpoint)
        base_model: The base LLM model instance to use for summarization when no explicit
            summarization model is specified. If None, falls back to gpt-4o-mini.
        provider: The model provider (e.g., "anthropic", "openai"). Used to add
            provider-specific middleware like Anthropic prompt caching.

    Returns:
        List of configured middleware instances
    """
    from langchain.agents.middleware import (
        ModelCallLimitMiddleware,
        ModelRetryMiddleware,
        ToolCallLimitMiddleware,
        ToolRetryMiddleware,
    )

    from karenina.schemas.workflow.models import AgentMiddlewareConfig

    if config is None:
        config = AgentMiddlewareConfig()

    middleware: list[Any] = []

    # 0. Anthropic Prompt Caching (if using Anthropic provider with langchain interface)
    # Added first so it can cache the system prompt, tools, and conversation history
    # See: https://docs.langchain.com/oss/python/integrations/middleware/anthropic#prompt-caching
    is_anthropic = provider is not None and provider.lower() == "anthropic"
    if is_anthropic and interface == "langchain" and config.prompt_caching.enabled:
        try:
            from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

            middleware.append(
                AnthropicPromptCachingMiddleware(
                    ttl=config.prompt_caching.ttl,
                    min_messages_to_cache=config.prompt_caching.min_messages_to_cache,
                    unsupported_model_behavior=config.prompt_caching.unsupported_model_behavior,
                )
            )
            logger.info(
                f"Anthropic prompt caching enabled: ttl={config.prompt_caching.ttl}, "
                f"min_messages_to_cache={config.prompt_caching.min_messages_to_cache}"
            )
        except ImportError:
            logger.warning(
                "langchain-anthropic not installed or AnthropicPromptCachingMiddleware not available. "
                "Prompt caching disabled. Install with: uv add langchain-anthropic"
            )

    # 1. Model call limit (always enabled for safety)
    middleware.append(
        ModelCallLimitMiddleware(
            run_limit=config.limits.model_call_limit,
            exit_behavior=config.limits.exit_behavior,  # type: ignore[arg-type]
        )
    )

    # 2. Tool call limit (always enabled for safety)
    middleware.append(
        ToolCallLimitMiddleware(
            run_limit=config.limits.tool_call_limit,
            exit_behavior="continue" if config.limits.exit_behavior == "end" else "error",
        )
    )

    # 3. Model retry middleware (replaces tenacity for agent path)
    middleware.append(
        ModelRetryMiddleware(
            max_retries=config.model_retry.max_retries,
            backoff_factor=config.model_retry.backoff_factor,
            initial_delay=config.model_retry.initial_delay,
            max_delay=config.model_retry.max_delay,
            jitter=config.model_retry.jitter,
            on_failure=config.model_retry.on_failure,  # type: ignore[arg-type]
        )
    )

    # 4. Tool retry middleware (new capability)
    middleware.append(
        ToolRetryMiddleware(
            max_retries=config.tool_retry.max_retries,
            backoff_factor=config.tool_retry.backoff_factor,
            initial_delay=config.tool_retry.initial_delay,
            on_failure=config.tool_retry.on_failure,  # type: ignore[arg-type]
        )
    )

    # 5. Summarization middleware (optional, token-based trigger)
    # Uses custom InvokeSummarizationMiddleware that triggers WITHIN a single invoke,
    # unlike built-in SummarizationMiddleware which only triggers between invokes.
    if config.summarization.enabled:
        # Use explicit model if specified, otherwise use the same model as the agent
        if config.summarization.model:
            summarization_model = config.summarization.model
            model_info = config.summarization.model
        elif base_model is not None:
            summarization_model = base_model
            model_info = "same as answering model"
        else:
            raise ValueError(
                "Summarization is enabled but no model is available. "
                "Either provide summarization.model in agent_middleware config, "
                "or ensure base_model is passed to _build_agent_middleware."
            )

        # Determine trigger token count based on configuration priority:
        # 1. config.summarization.trigger_tokens (explicit token threshold in config)
        # 2. max_context_tokens (model's context window, used directly as trigger)
        # 3. Default fallback based on interface type
        if config.summarization.trigger_tokens is not None:
            # Explicit trigger_tokens in config - highest priority
            trigger_tokens = config.summarization.trigger_tokens
            trigger_info = f"trigger_tokens={trigger_tokens} (from config)"
        elif max_context_tokens is not None:
            # max_context_tokens provided - use directly as trigger threshold
            trigger_tokens = max_context_tokens
            trigger_info = f"trigger_tokens={trigger_tokens} (from max_context_tokens)"
        else:
            # Default fallback: use fraction of default context size
            context_tokens = 100000
            trigger_tokens = int(context_tokens * config.summarization.trigger_fraction)
            trigger_info = (
                f"trigger_tokens={trigger_tokens} "
                f"({config.summarization.trigger_fraction * 100:.0f}% of {context_tokens} default)"
            )

        # Create custom middleware that summarizes WITHIN a single invoke
        summarization_mw = _create_invoke_summarization_middleware(
            model=summarization_model,
            trigger_tokens=trigger_tokens,
            keep_messages=config.summarization.keep_messages,
        )
        middleware.append(summarization_mw)
        logger.info(
            f"Summarization middleware enabled (InvokeSummarizationMiddleware): model={model_info}, "
            f"{trigger_info}, keep_messages={config.summarization.keep_messages}"
        )

    return middleware


def init_chat_model_unified(
    model: str,
    provider: str | None = None,
    interface: str = "langchain",
    question_hash: str | None = None,
    mcp_urls_dict: dict[str, str] | None = None,
    mcp_tool_filter: list[str] | None = None,
    endpoint_base_url: str | None = None,
    endpoint_api_key: str | SecretStr | None = None,
    agent_middleware_config: AgentMiddlewareConfig | None = None,
    max_context_tokens: int | None = None,
    **kwargs: Any,
) -> Any:
    """Initialize a chat model using the unified interface.

    This function provides a unified way to initialize different chat models
    across various interfaces (LangChain, OpenRouter, OpenAI Endpoint, Manual) with consistent
    parameter handling. When MCP URLs are provided, creates a LangGraph agent
    with tools from MCP servers.

    Args:
        model: The model name (e.g., "gemini-2.0-flash", "gpt-4.1-mini", "claude-3-sonnet")
        provider: The model provider (e.g., "google_genai", "openai", "anthropic").
                 Optional for OpenRouter, OpenAI Endpoint, and Manual interfaces.
        interface: The interface to use for model initialization.
                  Supported values: "langchain", "openrouter", "openai_endpoint", "manual"
        question_hash: The MD5 hash of the question (required for manual interface)
        mcp_urls_dict: Dictionary mapping tool names to MCP server URLs.
                      When provided, creates a LangGraph agent with MCP tools.
                      Keys are tool names, values are server URLs.
                      Not supported with manual interface.
        mcp_tool_filter: Optional list of tool names to include from MCP servers.
                        If provided, only tools with names in this list will be used.
                        Ignored if mcp_urls_dict is None.
        endpoint_base_url: Custom base URL for openai_endpoint interface.
                          Required for openai_endpoint interface.
                          Used to connect to OpenAI-compatible endpoints (vLLM, Ollama, etc.)
        endpoint_api_key: API key for openai_endpoint interface.
                         Required for openai_endpoint interface.
                         Must be explicitly provided - does NOT read from environment.
        agent_middleware_config: Optional middleware configuration for MCP-enabled agents.
                                Controls retry behavior, execution limits, and summarization.
                                Only used when mcp_urls_dict is provided.
        max_context_tokens: Maximum context tokens for the model.
                           For langchain interface, this is auto-detected from model profiles.
                           For openrouter/openai_endpoint, defaults to 100000 if not specified.
                           Used by SummarizationMiddleware to determine trigger threshold.
        **kwargs: Additional keyword arguments passed to the underlying model
                 initialization (e.g., temperature, max_tokens)

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

        Initialize OpenAI-compatible endpoint with explicit API key:
        >>> model = init_chat_model_unified(
        ...     "llama2",
        ...     interface="openai_endpoint",
        ...     endpoint_base_url="http://localhost:11434/v1",
        ...     endpoint_api_key="your-api-key"
        ... )

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
    elif interface == "openai_endpoint":
        if endpoint_base_url is None:
            raise ValueError("endpoint_base_url is required for openai_endpoint interface")
        if endpoint_api_key is None:
            raise ValueError(
                "endpoint_api_key is required for openai_endpoint interface. "
                "Pass the API key explicitly - this interface does not read from environment."
            )
        base_model = ChatOpenAIEndpoint(
            base_url=endpoint_base_url,
            openai_api_key=endpoint_api_key,
            model=model,
            **kwargs,
        )
    elif interface == "manual":
        if question_hash is None:
            raise ValueError("question_hash is required for manual interface")
        return create_manual_llm(question_hash=question_hash, **kwargs)
    else:
        raise ValueError(f"Unsupported interface: {interface}")

    # If no MCP URLs provided, return base model
    if mcp_urls_dict is None:
        return base_model

    # Create LangGraph agent with MCP tools and middleware
    try:
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        from .mcp_utils import sync_create_mcp_client_and_tools
    except ImportError as e:
        raise ImportError(
            "langchain>=1.1.0, langgraph and langchain-mcp-adapters are required for MCP support. "
            "Install with: uv add 'langchain>=1.1.0' langgraph langchain-mcp-adapters"
        ) from e

    try:
        # Get MCP client and tools
        _, tools = sync_create_mcp_client_and_tools(mcp_urls_dict, mcp_tool_filter)

        # Build middleware list from configuration
        # Pass base_model so summarization uses the same model by default
        # Pass provider to enable provider-specific middleware (e.g., Anthropic prompt caching)
        middleware = _build_agent_middleware(
            config=agent_middleware_config,
            max_context_tokens=max_context_tokens,
            interface=interface,
            base_model=base_model,
            provider=provider,
        )

        # Create agent with middleware and checkpointer
        # InMemorySaver enables partial state recovery when limits are hit
        memory = InMemorySaver()
        agent: Any = create_agent(
            model=base_model,
            tools=tools,
            checkpointer=memory,
            middleware=middleware,
        )

        logger.info(f"Created MCP agent with {len(tools)} tools and {len(middleware)} middleware components")

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
    interface: str = "langchain",
    endpoint_base_url: str | None = None,
    endpoint_api_key: str | None = None,
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
        interface: The interface to use ("langchain", "openrouter", "openai_endpoint", "manual")
        endpoint_base_url: Custom base URL for openai_endpoint interface
        endpoint_api_key: API key for openai_endpoint interface

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
            session_id,
            model,
            provider,
            temperature,
            mcp_urls_dict,
            mcp_tool_filter,
            interface,
            endpoint_base_url,
            endpoint_api_key,
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

            # Config for checkpointer - use session_id as thread_id for state tracking
            # This is required for middleware (like SummarizationMiddleware) to work correctly
            agent_config = {"configurable": {"thread_id": session_id}}

            async def invoke_agent_async() -> tuple[Any, bool]:
                """
                Invoke agent and return (response, recursion_limit_reached).
                This avoids using 'nonlocal' which can cause issues with nested async functions.
                """
                local_recursion_limit_reached = False

                try:
                    # Use ainvoke - returns complete final state with all messages
                    # Pass config with thread_id for proper checkpointing/middleware support
                    response = await session.llm.ainvoke({"messages": session.messages}, config=agent_config)
                    return response, local_recursion_limit_reached

                except Exception as e:
                    # Check if this is a GraphRecursionError
                    if "GraphRecursionError" in str(type(e).__name__) or "recursion_limit" in str(e).lower():
                        local_recursion_limit_reached = True

                        # Try multiple methods to extract accumulated messages from the agent
                        # Method 1: Check if exception contains state information
                        if hasattr(e, "state") and e.state is not None:
                            return e.state, local_recursion_limit_reached

                        # Method 2: Try to get current graph state if checkpointer exists
                        if hasattr(session.llm, "checkpointer") and session.llm.checkpointer is not None:
                            try:
                                if hasattr(session.llm, "get_state"):
                                    # Use agent_config which has the correct thread_id (session_id)
                                    state = session.llm.get_state(agent_config)
                                    if state and hasattr(state, "values") and "messages" in state.values:
                                        return {"messages": state.values["messages"]}, local_recursion_limit_reached
                            except Exception:
                                pass

                        # Method 3: Check if exception has accumulated messages attribute
                        if hasattr(e, "messages"):
                            return {"messages": e.messages}, local_recursion_limit_reached

                        # FALLBACK: Return input messages with warning
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.warning(
                            "Could not extract partial agent state after recursion limit. "
                            "Returning input messages only. Accumulated trace may be lost."
                        )
                        return {"messages": session.messages}, local_recursion_limit_reached
                    else:
                        raise e

            # Run the async invocation using the shared portal if available,
            # otherwise fall back to asyncio.run()
            from karenina.benchmark.verification.batch_runner import get_async_portal

            portal = get_async_portal()

            if portal is not None:
                # Use the shared BlockingPortal for proper event loop management
                response, recursion_limit_reached = portal.call(invoke_agent_async)
            else:
                # No portal available - use asyncio.run()
                try:
                    asyncio.get_running_loop()
                    # We're in an async context, use ThreadPoolExecutor
                    import concurrent.futures

                    def run_in_thread():
                        return asyncio.run(invoke_agent_async())

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        result = future.result(timeout=60)  # 60 second timeout
                        response, recursion_limit_reached = result

                except RuntimeError:
                    # No event loop running, safe to use asyncio.run
                    response, recursion_limit_reached = asyncio.run(invoke_agent_async())

            from .mcp_utils import harmonize_agent_response

            response_content = harmonize_agent_response(response, original_question=message)

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
