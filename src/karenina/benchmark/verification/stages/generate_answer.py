"""Answer generation stage.

Generates LLM responses to questions using the configured answering model.
"""

import logging
import traceback

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ....llm.interface import init_chat_model_unified
from ..stage import BaseVerificationStage, VerificationContext
from ..verification_utils import _construct_few_shot_prompt, _invoke_llm_with_retry, _is_valid_md5_hash

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
        return ["Answer"]  # Needs validated Answer class

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
        """Run if Answer class was successfully validated."""
        return context.has_artifact("Answer") and not context.error

    def execute(self, context: VerificationContext) -> None:
        """
        Generate answer using configured LLM.

        Args:
            context: Verification context

        Side Effects:
            - Sets context.artifacts["raw_llm_response"]
            - Sets context.artifacts["recursion_limit_reached"]
            - Sets context.artifacts["answering_model_str"]
            - Sets context.artifacts["answering_mcp_servers"]
            - Sets context.error if LLM call fails fatally
        """
        answering_model = context.answering_model

        # Build model string for result
        if answering_model.interface == "openrouter":
            answering_model_str = answering_model.model_name
        else:
            answering_model_str = f"{answering_model.model_provider}/{answering_model.model_name}"
        context.set_artifact("answering_model_str", answering_model_str)

        # Extract MCP server names if configured
        answering_mcp_servers = list(answering_model.mcp_urls_dict.keys()) if answering_model.mcp_urls_dict else None
        context.set_artifact("answering_mcp_servers", answering_mcp_servers)

        # Log MCP configuration if present
        if answering_mcp_servers:
            logger.info(f"Answering model MCP servers: {answering_mcp_servers}")

        # Step 1: Validate question_id for manual interface
        if answering_model.interface == "manual" and not _is_valid_md5_hash(context.question_id):
            error_msg = (
                f"Invalid question_id format for manual interface: '{context.question_id}'. "
                "question_id must be a 32-character hexadecimal MD5 hash when using manual interface. "
                "This hash is typically generated during question extraction from the question text."
            )
            context.mark_error(error_msg)
            return

        # Step 2: Initialize answering LLM
        try:
            if answering_model.interface == "manual":
                answering_llm = init_chat_model_unified(
                    model=answering_model.model_name,
                    provider=answering_model.model_provider,
                    temperature=answering_model.temperature,
                    interface=answering_model.interface,
                    question_hash=context.question_id,
                    mcp_urls_dict=answering_model.mcp_urls_dict,
                    mcp_tool_filter=answering_model.mcp_tool_filter,
                )
            else:
                answering_llm = init_chat_model_unified(
                    model=answering_model.model_name,
                    provider=answering_model.model_provider,
                    temperature=answering_model.temperature,
                    interface=answering_model.interface,
                    mcp_urls_dict=answering_model.mcp_urls_dict,
                    mcp_tool_filter=answering_model.mcp_tool_filter,
                )
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

        try:
            # Use retry-wrapped invocation
            is_agent = answering_model.mcp_urls_dict is not None
            response, recursion_limit_reached = _invoke_llm_with_retry(answering_llm, messages, is_agent)

            # Process response based on type
            if is_agent:
                raw_llm_response = response
                # Add note if recursion limit was reached
                if recursion_limit_reached:
                    raw_llm_response += "\n\n[Note: Recursion limit reached - partial response shown]"
            else:
                raw_llm_response = response.content if hasattr(response, "content") else str(response)

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

        # Also store in result builder for easy access
        context.set_result_field("raw_llm_response", raw_llm_response)
        context.set_result_field("recursion_limit_reached", recursion_limit_reached)
        context.set_result_field("answering_mcp_servers", answering_mcp_servers)
