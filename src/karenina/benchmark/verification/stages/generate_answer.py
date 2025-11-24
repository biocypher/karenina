"""Answer generation stage.

Generates LLM responses to questions using the configured answering model.
"""

import logging
import traceback
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ....infrastructure.llm.interface import init_chat_model_unified
from ..stage import BaseVerificationStage, VerificationContext
from ..utils import UsageTracker
from ..verification_utils import _construct_few_shot_prompt, _invoke_llm_with_retry

# Set up logger
logger = logging.getLogger(__name__)


class GenerateAnswerStage(BaseVerificationStage):
    """
    Generates LLM answer to the question.

    This stage:
    1. Validates question_id format for manual interface
    2. Initializes the answering LLM (with MCP support if configured)
    3. Constructs prompt messages (system prompt + few-shot examples)
    4. Invokes LLM with automatic retry logic
    5. Handles MCP agents vs regular LLMs
    6. Handles recursion limit errors gracefully
    7. Extracts raw response text

    Requires:
        - "Answer": Validated Answer class (to check if we should proceed)

    Produces:
        - "raw_llm_response": Raw text response from LLM
        - "recursion_limit_reached": Whether agent hit recursion limit (bool)
        - "answering_model_str": Model string for result (provider/model)
        - "answering_mcp_servers": List of MCP server names (if using MCP)

    Error Handling:
        If LLM call fails with non-recoverable error, marks context.error
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

            # Build model string for result
            answering_model = context.answering_model
            # Note: model_name is guaranteed non-None by ModelConfig validator
            assert answering_model.model_name is not None, "model_name must not be None"

            if answering_model.interface == "openrouter":
                answering_model_str = answering_model.model_name
            elif answering_model.interface == "openai_endpoint":
                answering_model_str = f"endpoint/{answering_model.model_name}"
            else:
                answering_model_str = f"{answering_model.model_provider}/{answering_model.model_name}"
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
        # Note: model_name is guaranteed non-None by ModelConfig validator
        assert answering_model.model_name is not None, "model_name must not be None"

        # Build model string for result
        if answering_model.interface == "openrouter":
            answering_model_str = answering_model.model_name
        elif answering_model.interface == "openai_endpoint":
            answering_model_str = f"endpoint/{answering_model.model_name}"
        else:
            answering_model_str = f"{answering_model.model_provider}/{answering_model.model_name}"
        context.set_artifact("answering_model_str", answering_model_str)

        # Extract MCP server names if configured
        answering_mcp_servers = list(answering_model.mcp_urls_dict.keys()) if answering_model.mcp_urls_dict else None
        context.set_artifact("answering_mcp_servers", answering_mcp_servers)

        # Log MCP configuration if present
        if answering_mcp_servers:
            logger.info(f"Answering model MCP servers: {answering_mcp_servers}")

        # Step 1: For manual interface, compute MD5 hash from question text
        question_hash_for_manual = None
        if answering_model.interface == "manual":
            import hashlib

            # Convert question text to MD5 hash for manual trace lookup
            question_hash_for_manual = hashlib.md5(context.question_text.encode("utf-8")).hexdigest()
            logger.info(
                f"Manual interface: Using MD5 hash '{question_hash_for_manual}' from question text for trace lookup"
            )

        # Step 2: Initialize answering LLM
        try:
            # Build base kwargs for model initialization
            model_kwargs: dict[str, Any] = {
                "model": answering_model.model_name,
                "provider": answering_model.model_provider,
                "temperature": answering_model.temperature,
                "interface": answering_model.interface,
                "mcp_urls_dict": answering_model.mcp_urls_dict,
                "mcp_tool_filter": answering_model.mcp_tool_filter,
                # Native tool calling configuration (for MCP agents)
                "native_tool_calling": answering_model.native_tool_calling,
                "native_tool_calling_max_iterations": answering_model.native_tool_calling_max_iterations,
            }

            # Add interface-specific parameters
            if answering_model.interface == "manual":
                model_kwargs["question_hash"] = question_hash_for_manual
            elif answering_model.interface == "openai_endpoint":
                # Require endpoint configuration
                if not answering_model.endpoint_base_url:
                    raise ValueError("endpoint_base_url is required for openai_endpoint interface")
                if not answering_model.endpoint_api_key:
                    raise ValueError("endpoint_api_key is required for openai_endpoint interface")

                model_kwargs["endpoint_base_url"] = answering_model.endpoint_base_url
                model_kwargs["endpoint_api_key"] = answering_model.endpoint_api_key.get_secret_value()

            # Add any extra kwargs if provided (e.g., vendor-specific API keys)
            if answering_model.extra_kwargs:
                model_kwargs.update(answering_model.extra_kwargs)

            answering_llm = init_chat_model_unified(**model_kwargs)
        except Exception as e:
            error_msg = f"Failed to initialize answering model: {type(e).__name__}: {e}"
            logger.error(error_msg)
            context.mark_error(error_msg)
            return

        # Step 3: Construct prompt messages
        messages: list[BaseMessage] = []
        if answering_model.system_prompt:
            messages.append(SystemMessage(content=answering_model.system_prompt))

        # Construct prompt with optional few-shot examples
        constructed_prompt = _construct_few_shot_prompt(
            context.question_text, context.few_shot_examples, context.few_shot_enabled
        )
        messages.append(HumanMessage(content=constructed_prompt))

        # Step 4: Invoke LLM with retry logic
        recursion_limit_reached = False

        # Initialize usage tracker for this stage
        usage_tracker = UsageTracker()

        try:
            # Use retry-wrapped invocation
            is_agent = answering_model.mcp_urls_dict is not None
            response, recursion_limit_reached, usage_metadata, agent_metrics = _invoke_llm_with_retry(
                answering_llm, messages, is_agent
            )

            # Track usage metadata
            if usage_metadata:
                usage_tracker.track_call("answer_generation", answering_model_str, usage_metadata)

            # Track agent metrics if this was an MCP agent
            if is_agent and agent_metrics:
                # Store agent metrics in tracker for aggregation in final result
                usage_tracker.set_agent_metrics(agent_metrics)

            # Process response based on type
            if is_agent:
                raw_llm_response = response
                # Add note if recursion limit was reached
                if recursion_limit_reached:
                    raw_llm_response += "\n\n[Note: Recursion limit reached - partial response shown]"
            else:
                raw_llm_response = response.content if hasattr(response, "content") else str(response)

            # For manual interface, retrieve agent metrics if available
            if answering_model.interface == "manual" and hasattr(answering_llm, "get_agent_metrics"):
                manual_agent_metrics = answering_llm.get_agent_metrics()
                if manual_agent_metrics:
                    usage_tracker.set_agent_metrics(manual_agent_metrics)

        except Exception as e:
            # Check if this is a recursion limit error that wasn't caught earlier
            if "GraphRecursionError" in str(type(e).__name__) or "recursion_limit" in str(e).lower():
                recursion_limit_reached = True
                raw_llm_response = f"[Note: Recursion limit reached before completion. Error: {e}]"
                logger.warning(f"Recursion limit reached for question {context.question_id}")
                # Continue processing with this partial response
            else:
                # Log detailed error information for debugging
                error_details = traceback.format_exc()
                logger.error(
                    f"LLM call failed for question {context.question_id}\n"
                    f"Exception type: {type(e).__name__}\n"
                    f"Exception message: {e}\n"
                    f"Full traceback:\n{error_details}"
                )

                # Mark as fatal error
                error_msg = f"LLM call failed: {type(e).__name__}: {e}"
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
