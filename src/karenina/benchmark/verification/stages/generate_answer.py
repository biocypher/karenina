"""Answer generation stage.

Generates LLM responses to questions using the configured answering model.
Uses a unified adapter path for ALL interfaces including manual.
"""

import hashlib
import logging
import traceback
from typing import Any

from ....adapters import get_agent
from ....ports import AgentConfig, AgentPort, Message
from ..stage import BaseVerificationStage, VerificationContext
from ..utils.llm_invocation import _construct_few_shot_prompt
from ..utils.trace_usage_tracker import UsageTracker

# Set up logger
logger = logging.getLogger(__name__)


class GenerateAnswerStage(BaseVerificationStage):
    """
    Generates LLM answer to the question using unified adapter path.

    This stage uses a single code path for ALL interfaces including manual:
    1. Computes question hash (for manual trace lookup, ignored by other adapters)
    2. Gets adapter via unified factory (get_agent)
    3. Constructs prompt messages (system prompt + few-shot examples)
    4. Invokes adapter with AgentConfig containing question_hash
    5. Handles recursion limit errors gracefully
    6. Extracts raw response text from AgentResult

    Requires:
        - "Answer": Validated Answer class (to check if we should proceed)

    Produces:
        - "raw_llm_response": Raw text response from LLM
        - "recursion_limit_reached": Whether agent hit recursion limit (bool)
        - "answering_model_str": Model string for result (provider/model)
        - "answering_mcp_servers": List of MCP server names (if using MCP)

    Error Handling:
        If adapter call fails with non-recoverable error, marks context.error
        and sets completed_without_errors=False. Recursion limit errors
        are handled gracefully and allow pipeline to continue.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "GenerateAnswer"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        # Answer artifact is optional - not needed in rubric_only mode
        return []

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            "raw_llm_response",
            "recursion_limit_reached",
            "answering_model_str",
            "answering_mcp_servers",
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run if either:
        - Answer class was successfully validated (template modes), OR
        - No Answer artifact exists (rubric_only mode - template validation skipped)

        Skip only if there's an error.
        """
        # In rubric_only mode, Answer artifact won't exist but we still need to generate response
        # In template modes, Answer artifact should exist
        return not context.error

    def execute(self, context: VerificationContext) -> None:
        """
        Generate answer using configured LLM or use cached answer.

        This stage checks if cached answer data is available in the context.
        If so, it injects the cached data and skips LLM invocation. This
        optimization allows multiple judges to share the same answering trace.

        Args:
            context: Verification context

        Side Effects:
            - Sets context.artifacts["raw_llm_response"]
            - Sets context.artifacts["recursion_limit_reached"]
            - Sets context.artifacts["answering_model_str"]
            - Sets context.artifacts["answering_mcp_servers"]
            - Sets context.error if LLM call fails fatally
        """
        # Check if cached answer data is available
        if context.cached_answer_data is not None:
            logger.info(f"Using cached answer for question {context.question_id}")

            # Extract cached data
            raw_llm_response = context.cached_answer_data.get("raw_llm_response", "")
            recursion_limit_reached = context.cached_answer_data.get("recursion_limit_reached", False)
            answering_mcp_servers = context.cached_answer_data.get("answering_mcp_servers")
            usage_metadata = context.cached_answer_data.get("usage_metadata")
            agent_metrics = context.cached_answer_data.get("agent_metrics")

            # Store cached data in context
            context.set_artifact("raw_llm_response", raw_llm_response)
            context.set_artifact("recursion_limit_reached", recursion_limit_reached)
            context.set_artifact("answering_mcp_servers", answering_mcp_servers)

            # Build model string for result (centralized via adapter registry)
            from karenina.adapters import format_model_string

            answering_model = context.answering_model
            answering_model_str = format_model_string(answering_model)
            context.set_artifact("answering_model_str", answering_model_str)

            # Set result fields
            context.set_result_field("raw_llm_response", raw_llm_response)
            context.set_result_field("recursion_limit_reached", recursion_limit_reached)
            context.set_result_field("answering_mcp_servers", answering_mcp_servers)

            # Handle usage tracking for cached answers
            usage_tracker = UsageTracker()
            if usage_metadata:
                usage_tracker.track_call("answer_generation", answering_model_str, usage_metadata)
            if agent_metrics:
                usage_tracker.set_agent_metrics(agent_metrics)
            context.set_artifact("usage_tracker", usage_tracker)

            return  # Skip LLM invocation

        # No cached answer - proceed with normal answer generation
        answering_model = context.answering_model

        # Build model string for result (centralized via adapter registry)
        from karenina.adapters import format_model_string

        answering_model_str = format_model_string(answering_model)
        context.set_artifact("answering_model_str", answering_model_str)

        # Extract MCP server names if configured
        answering_mcp_servers = list(answering_model.mcp_urls_dict.keys()) if answering_model.mcp_urls_dict else None
        context.set_artifact("answering_mcp_servers", answering_mcp_servers)

        # Log MCP configuration if present
        if answering_mcp_servers:
            logger.info(f"Answering model MCP servers: {answering_mcp_servers}")

        # Step 1: Compute question hash (needed for manual, ignored by other adapters)
        question_hash = hashlib.md5(context.question_text.encode("utf-8")).hexdigest()
        if answering_model.interface == "manual":
            logger.info(f"Manual interface: Using MD5 hash '{question_hash}' from question text for trace lookup")

        # Step 2: Initialize agent adapter (unified path for ALL interfaces)
        answering_agent: AgentPort | None = None

        try:
            # All interfaces use unified adapter factory including manual
            # ManualAgentAdapter reads from ManualTraceManager using question_hash
            answering_agent = get_agent(answering_model, auto_fallback=True)
            logger.info(f"Using {answering_model.interface} adapter for {answering_model_str}")
        except Exception as e:
            error_msg = f"Failed to initialize answering model: {type(e).__name__}: {e}"
            logger.error(error_msg)
            context.mark_error(error_msg)
            return

        # Step 3: Construct prompt text
        constructed_prompt = _construct_few_shot_prompt(
            context.question_text, context.few_shot_examples, context.few_shot_enabled
        )

        # Step 4: Invoke adapter
        recursion_limit_reached = False
        raw_llm_response = ""

        # Initialize usage tracker for this stage
        usage_tracker = UsageTracker()

        try:
            # Unified adapter path for ALL interfaces
            # (langchain, openrouter, openai_endpoint, claude_agent_sdk, manual)
            # Construct messages in unified Message format
            adapter_messages: list[Message] = []
            if answering_model.system_prompt:
                adapter_messages.append(Message.system(answering_model.system_prompt))
            adapter_messages.append(Message.user(constructed_prompt))

            # Build MCP server config if needed
            mcp_servers: dict[str, Any] | None = None
            if answering_model.mcp_urls_dict:
                # Convert URL-based config to SDK format
                # The adapter handles this conversion internally
                mcp_servers = {}
                for name, url in answering_model.mcp_urls_dict.items():
                    mcp_servers[name] = {"type": "http", "url": url}

            # Build agent config with question_hash (used by manual, ignored by others)
            agent_config = AgentConfig(
                max_turns=answering_model.agent_middleware.limits.model_call_limit
                if answering_model.agent_middleware
                else 25,
                timeout=120,
                question_hash=question_hash,
            )

            # Run the agent
            result = answering_agent.run_sync(
                messages=adapter_messages,
                mcp_servers=mcp_servers,
                config=agent_config,
            )

            # Extract results from AgentResult
            raw_llm_response = result.raw_trace
            recursion_limit_reached = result.limit_reached

            # Track usage metadata from adapter
            if result.usage:
                # Build inner usage dict
                inner_usage: dict[str, int | float] = {
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "total_tokens": result.usage.total_tokens,
                }
                if result.usage.cost_usd is not None:
                    inner_usage["cost_usd"] = result.usage.cost_usd
                if result.usage.cache_read_tokens is not None:
                    inner_usage["cache_read_input_tokens"] = result.usage.cache_read_tokens
                if result.usage.cache_creation_tokens is not None:
                    inner_usage["cache_creation_input_tokens"] = result.usage.cache_creation_tokens

                # Wrap in model-keyed format expected by UsageTracker
                usage_metadata = {answering_model_str: inner_usage}
                usage_tracker.track_call("answer_generation", answering_model_str, usage_metadata)

            # Track agent metrics
            agent_metrics = {
                "iterations": result.turns,
                "limit_reached": result.limit_reached,
            }
            usage_tracker.set_agent_metrics(agent_metrics)

            # Store trace_messages for future use (PR5a)
            if result.trace_messages:
                context.set_artifact("trace_messages", result.trace_messages)

        except Exception as e:
            # Adapters handle recursion limits internally and return AgentResult with
            # limit_reached=True. If we reach here, it's an unexpected error.
            # Log detailed error information for debugging
            error_details = traceback.format_exc()
            logger.error(
                f"Adapter call failed for question {context.question_id}\n"
                f"Exception type: {type(e).__name__}\n"
                f"Exception message: {e}\n"
                f"Full traceback:\n{error_details}"
            )

            # Mark as fatal error
            error_msg = f"Adapter call failed: {type(e).__name__}: {e}"
            context.mark_error(error_msg)
            context.set_artifact("raw_llm_response", "")
            context.set_artifact("recursion_limit_reached", False)
            return

        # Store results
        context.set_artifact("raw_llm_response", raw_llm_response)
        context.set_artifact("recursion_limit_reached", recursion_limit_reached)

        # Store usage tracker for next stages to continue tracking
        context.set_artifact("usage_tracker", usage_tracker)

        # Also store in result builder for easy access
        context.set_result_field("raw_llm_response", raw_llm_response)
        context.set_result_field("recursion_limit_reached", recursion_limit_reached)
        context.set_result_field("answering_mcp_servers", answering_mcp_servers)
