"""LangChain agent middleware builders.

This module contains middleware factory functions for LangGraph agents,
including retry handling, execution limits, and conversation summarization.

These functions are adapter-internal and should not be imported directly.
Use the AgentPort interface via get_agent() instead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from .prompts import SUMMARIZATION, build_question_context

if TYPE_CHECKING:
    from karenina.schemas.workflow.models import AgentMiddlewareConfig

logger = logging.getLogger(__name__)


def fetch_openai_endpoint_context_size(
    base_url: str,
    api_key: str | SecretStr,
    model_name: str,
) -> int | None:
    """Fetch max context size from OpenAI-compatible endpoint's /v1/models API.

    Queries the /v1/models endpoint to discover the model's max_model_len,
    which is the context window size. This is supported by most OpenAI-compatible
    servers (vLLM, SGLang, Ollama, etc.).

    Args:
        base_url: The endpoint base URL (e.g., "http://localhost:8000/v1")
        api_key: The API key for authentication
        model_name: The model name to look up

    Returns:
        The max_model_len value if found, None otherwise.
        Returns None on any error (network, parsing, model not found).
    """
    # Normalize base_url (remove trailing /v1 if present, we'll add /v1/models)
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    url = f"{base_url}/v1/models"

    try:
        key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        response = httpx.get(
            url,
            headers={"Authorization": f"Bearer {key}"},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

        # Find the matching model
        for model_info in data.get("data", []):
            if model_info.get("id") == model_name:
                max_len = model_info.get("max_model_len")
                if max_len is not None:
                    logger.info(f"Auto-detected max_model_len={max_len} for model '{model_name}'")
                    return int(max_len)

        logger.debug(f"Model '{model_name}' not found in /v1/models response")
        return None

    except Exception as e:
        logger.debug(f"Failed to fetch context size from {url}: {e}")
        return None


class InvokeSummarizationMiddleware:
    """Custom middleware that summarizes conversation WITHIN a single invoke.

    Unlike built-in SummarizationMiddleware which only triggers between invokes,
    this uses the before_model hook to check and summarize before each model call.

    Uses RemoveMessage operations to properly remove summarized messages from
    LangGraph state, since the default message reducer appends rather than replaces.

    Note: This class lazily imports LangChain dependencies in __init__ to avoid
    import issues when langchain is not installed.
    """

    def __init__(
        self,
        summarization_model: Any,
        trigger_token_count: int = 4000,
        keep_message_count: int = 5,
    ):
        # Lazy import to avoid module-level import issues
        try:
            from langchain.agents.middleware import AgentMiddleware
        except ImportError as e:
            raise ImportError(
                "langchain>=1.1.0 and langgraph are required for middleware support. "
                "Install with: uv add 'langchain>=1.1.0' langgraph"
            ) from e

        # Store base class for isinstance checks if needed
        self._middleware_base = AgentMiddleware

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
        question_context = build_question_context(original_question)
        summary_prompt = SUMMARIZATION.format(
            question_context=question_context,
            messages_text=messages_text,
        )

        response = self.model.invoke([HumanMessage(content=summary_prompt)])
        return str(response.content)

    def before_model(self, state: Any, runtime: Any = None) -> dict[str, Any] | None:  # noqa: ARG002
        """Check token count before each model call and summarize if needed.

        Uses RemoveMessage to properly delete old messages from LangGraph state,
        then adds a summary message. This ensures the trace only contains the
        summarized version, not the original long conversation.
        """
        from langchain_core.messages import RemoveMessage

        messages = state.get("messages", [])

        current_tokens = self._count_tokens(messages)
        logger.debug(f"[InvokeSummarization] Token count: ~{current_tokens}, threshold: {self.trigger_tokens}")

        if current_tokens >= self.trigger_tokens and len(messages) > self.keep_messages:
            logger.info(
                f"[InvokeSummarization] TRIGGERING SUMMARIZATION! Tokens: ~{current_tokens} >= {self.trigger_tokens}"
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
                    logger.warning(f"[InvokeSummarization] Message without ID cannot be removed: {type(msg).__name__}")

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


def create_invoke_summarization_middleware(
    model: Any,
    trigger_tokens: int,
    keep_messages: int,
) -> InvokeSummarizationMiddleware:
    """Create a custom middleware that summarizes conversation WITHIN a single invoke.

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
    return InvokeSummarizationMiddleware(
        summarization_model=model,
        trigger_token_count=trigger_tokens,
        keep_message_count=keep_messages,
    )


def build_agent_middleware(
    config: AgentMiddlewareConfig | None,
    max_context_tokens: int | None = None,
    interface: str = "langchain",
    base_model: Any = None,
    provider: str | None = None,
) -> list[Any]:
    """Build middleware list from configuration.

    Args:
        config: Middleware configuration (uses defaults if None).
        max_context_tokens: Maximum context tokens for the model. Used as
            trigger threshold for summarization middleware when provided.
        interface: The interface type (langchain, openrouter, openai_endpoint).
        base_model: The base LLM model instance to use for summarization.
            Required if summarization is enabled and no explicit model is configured.
        provider: The model provider (e.g., "anthropic", "openai"). Used to add
            provider-specific middleware like Anthropic prompt caching.

    Returns:
        List of configured middleware instances.
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
                "or ensure base_model is passed to build_agent_middleware."
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
        summarization_mw = create_invoke_summarization_middleware(
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
