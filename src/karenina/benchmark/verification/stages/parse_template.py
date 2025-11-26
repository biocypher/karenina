"""Template parsing stage.

Parses LLM responses into Pydantic objects using standard or deep-judgment parsing.
"""

import logging
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from ....infrastructure.llm.interface import init_chat_model_unified
from ....infrastructure.llm.mcp_utils import extract_final_ai_message
from ....schemas.workflow import VerificationConfig
from ..evaluators.deep_judgment import deep_judgment_parse
from ..stage import BaseVerificationStage, VerificationContext
from ..utils import UsageTracker
from ..utils.parsing import _strip_markdown_fences
from ..verification_utils import (
    _retry_parse_with_null_feedback,
    _should_expose_ground_truth,
    _system_prompt_compose,
)

# Set up logger
logger = logging.getLogger(__name__)


class ParseTemplateStage(BaseVerificationStage):
    """
    Parses LLM response into Pydantic object.

    This stage:
    1. Initializes the parsing LLM
    2. Creates PydanticOutputParser for the Answer class
    3. Optionally extracts ground truth for semantic matching
    4. Creates parsing prompt with format instructions
    5. Chooses parsing strategy (standard vs deep-judgment)
    6. For standard: Single-stage parsing with PydanticOutputParser
    7. For deep-judgment: Multi-stage parsing with excerpt extraction
    8. Handles parsing errors gracefully

    Requires:
        - "RawAnswer": Validated Answer class (before question ID injection)
        - "Answer": Answer class with question ID injected
        - "raw_llm_response": Raw LLM response text

    Produces:
        - "parsed_answer": Parsed Pydantic object
        - "parsing_model_str": Model string for result
        - "deep_judgment_performed": Whether deep-judgment was used (bool)
        - "extracted_excerpts": Dict of excerpts per attribute (if deep-judgment)
        - "attribute_reasoning": Dict of reasoning per attribute (if deep-judgment)
        - "deep_judgment_stages_completed": List of completed stages (if deep-judgment)
        - "deep_judgment_model_calls": Number of LLM calls (if deep-judgment)
        - "deep_judgment_excerpt_retry_count": Retry count (if deep-judgment)
        - "attributes_without_excerpts": Attributes missing excerpts (if deep-judgment)
        - "hallucination_risk_assessment": Risk per attribute (if deep-judgment with search)

    Error Handling:
        If parsing fails (PydanticOutputParser creation or parsing),
        marks context.error and sets completed_without_errors=False.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "ParseTemplate"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        return ["RawAnswer", "Answer", "raw_llm_response"]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            "parsed_answer",
            "parsing_model_str",
            "deep_judgment_performed",
            "extracted_excerpts",
            "attribute_reasoning",
            "deep_judgment_stages_completed",
            "deep_judgment_model_calls",
            "deep_judgment_excerpt_retry_count",
            "attributes_without_excerpts",
            "hallucination_risk_assessment",
            "template_evaluation_input",
            "used_full_trace_for_template",
            "trace_extraction_error",
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """Run if we have raw LLM response, no errors, and no recursion limit."""
        # Skip parsing if recursion limit was reached (response is truncated/unreliable)
        if context.get_artifact("recursion_limit_reached", False):
            return False
        return context.has_artifact("raw_llm_response") and context.has_artifact("Answer") and not context.error

    def execute(self, context: VerificationContext) -> None:
        """
        Parse LLM response into Pydantic object.

        Args:
            context: Verification context

        Side Effects:
            - Sets context.artifacts["parsed_answer"]
            - Sets context.artifacts["parsing_model_str"]
            - Sets deep-judgment artifacts if enabled
            - Sets context.error if parsing fails
        """
        parsing_model = context.parsing_model
        raw_llm_response = context.get_artifact("raw_llm_response")
        Answer = context.get_artifact("Answer")
        RawAnswer = context.get_artifact("RawAnswer")

        # Retrieve usage tracker from previous stage or initialize new one
        usage_tracker = context.get_artifact("usage_tracker")
        if usage_tracker is None:
            usage_tracker = UsageTracker()
            logger.warning("No usage tracker found in context, initializing new one")

        # Determine what input to pass to template parsing based on config
        use_full_trace = context.use_full_trace_for_template
        trace_extraction_error = None
        template_evaluation_input = raw_llm_response  # Default to full trace

        if not use_full_trace:
            # Extract only the final AI message
            extracted_message, error = extract_final_ai_message(raw_llm_response)

            if error is not None:
                # Extraction failed - mark as error and stop
                error_msg = f"Failed to extract final AI message for template parsing: {error}"
                logger.error(error_msg)
                trace_extraction_error = error
                context.mark_error(error_msg)

                # Store metadata before returning
                context.set_artifact("used_full_trace_for_template", use_full_trace)
                context.set_artifact("trace_extraction_error", trace_extraction_error)
                context.set_artifact("template_evaluation_input", None)
                context.set_result_field("used_full_trace_for_template", use_full_trace)
                context.set_result_field("trace_extraction_error", trace_extraction_error)
                context.set_result_field("template_evaluation_input", None)
                return
            else:
                # Extraction successful - use extracted message
                template_evaluation_input = extracted_message
                logger.info("Using final AI message only for template parsing")

        # Store trace filtering metadata
        context.set_artifact("used_full_trace_for_template", use_full_trace)
        context.set_artifact("template_evaluation_input", template_evaluation_input)
        context.set_artifact("trace_extraction_error", trace_extraction_error)

        # Build model string for result
        if parsing_model.interface == "openrouter":
            parsing_model_str = parsing_model.model_name
        elif parsing_model.interface == "openai_endpoint":
            parsing_model_str = f"endpoint/{parsing_model.model_name}"
        else:
            parsing_model_str = f"{parsing_model.model_provider}/{parsing_model.model_name}"
        context.set_artifact("parsing_model_str", parsing_model_str)

        # Step 1: Initialize parsing LLM
        try:
            # Note: model_name is guaranteed non-None by ModelConfig validator
            assert parsing_model.model_name is not None, "model_name must not be None"

            # Build base kwargs for model initialization
            model_kwargs: dict[str, Any] = {
                "model": parsing_model.model_name,
                "provider": parsing_model.model_provider,
                "temperature": parsing_model.temperature,
                "interface": parsing_model.interface,
            }

            # Add interface-specific parameters
            if parsing_model.interface == "openai_endpoint":
                # Require endpoint configuration
                if not parsing_model.endpoint_base_url:
                    raise ValueError("endpoint_base_url is required for openai_endpoint interface")
                if not parsing_model.endpoint_api_key:
                    raise ValueError("endpoint_api_key is required for openai_endpoint interface")

                model_kwargs["endpoint_base_url"] = parsing_model.endpoint_base_url
                model_kwargs["endpoint_api_key"] = parsing_model.endpoint_api_key.get_secret_value()

            # Add any extra kwargs if provided (e.g., vendor-specific API keys)
            if parsing_model.extra_kwargs:
                model_kwargs.update(parsing_model.extra_kwargs)

            parsing_llm = init_chat_model_unified(**model_kwargs)
        except Exception as e:
            error_msg = f"Failed to initialize parsing model: {type(e).__name__}: {e}"
            logger.error(error_msg)
            context.mark_error(error_msg)
            return

        # Step 2: Create PydanticOutputParser
        try:
            parser: Any = PydanticOutputParser(pydantic_object=Answer)
        except Exception as e:
            error_msg = f"Failed to create PydanticOutputParser: {e}"
            logger.error(error_msg)
            context.mark_error(error_msg)
            return

        # Step 3: Extract ground truth if enabled
        ground_truth = None
        if _should_expose_ground_truth():
            try:
                from ..utils.parsing import create_test_instance_from_answer_class

                # Create test instance and extract ground truth
                _, ground_truth = create_test_instance_from_answer_class(RawAnswer)
            except Exception as e:
                # If we can't extract ground truth, continue without it
                logger.warning(f"Could not extract ground truth for question {context.question_id}: {e}")

        # Step 4: Create parsing prompt
        format_instructions = parser.get_format_instructions()
        combined_system_prompt = _system_prompt_compose(parsing_model.system_prompt, format_instructions, ground_truth)

        # Construct the parsing prompt (user message) with question context
        # Use template_evaluation_input which may be full trace or final AI message
        parsing_prompt = f"""<original_question>
Your task is to parse an answer given to the question reported in this section. Use the question to contextualize the info from the schema fields below:

Original Question: {context.question_text}
</original_question>

<response_to_parse>
{template_evaluation_input}
</response_to_parse>"""

        parsing_messages: list[BaseMessage] = []
        if combined_system_prompt:
            parsing_messages.append(SystemMessage(content=combined_system_prompt))
        parsing_messages.append(HumanMessage(content=parsing_prompt))

        # Initialize deep-judgment metadata variables
        deep_judgment_performed = False
        extracted_excerpts = None
        attribute_reasoning = None
        deep_judgment_stages_completed = None
        deep_judgment_model_calls = 0
        deep_judgment_excerpt_retry_count = 0
        attributes_without_excerpts = None
        hallucination_risk_assessment = None

        # Step 5: Choose parsing strategy and execute
        try:
            if context.deep_judgment_enabled:
                # Create minimal config for deep-judgment
                dj_config = VerificationConfig(
                    answering_models=[],
                    parsing_models=[parsing_model],
                    parsing_only=True,
                    deep_judgment_enabled=True,
                    deep_judgment_max_excerpts_per_attribute=context.deep_judgment_max_excerpts_per_attribute,
                    deep_judgment_fuzzy_match_threshold=context.deep_judgment_fuzzy_match_threshold,
                    deep_judgment_excerpt_retry_attempts=context.deep_judgment_excerpt_retry_attempts,
                    deep_judgment_search_enabled=context.deep_judgment_search_enabled,
                    deep_judgment_search_tool=context.deep_judgment_search_tool,
                )

                # Deep-judgment multi-stage parsing
                # Use template_evaluation_input which may be full trace or final AI message
                parsed_answer, extracted_excerpts, attribute_reasoning, dj_metadata = deep_judgment_parse(
                    raw_llm_response=template_evaluation_input,
                    RawAnswer=RawAnswer,
                    parsing_model=parsing_model,
                    parsing_llm=parsing_llm,
                    question_text=context.question_text,
                    config=dj_config,
                    format_instructions=format_instructions,
                    combined_system_prompt=combined_system_prompt,
                    usage_tracker=usage_tracker,
                    parsing_model_str=parsing_model_str,
                )
                deep_judgment_performed = True
                deep_judgment_stages_completed = dj_metadata.get("stages_completed", [])
                deep_judgment_model_calls = dj_metadata.get("model_calls", 0)
                deep_judgment_excerpt_retry_count = dj_metadata.get("excerpt_retry_count", 0)
                attributes_without_excerpts = dj_metadata.get("attributes_without_excerpts", None)
                hallucination_risk_assessment = dj_metadata.get("hallucination_risk", None)
            else:
                # Standard single-stage parsing with usage tracking
                with get_usage_metadata_callback() as cb:
                    parsing_response = parsing_llm.invoke(parsing_messages)

                # Track the parsing call
                usage_metadata = dict(cb.usage_metadata) if cb.usage_metadata else {}
                if usage_metadata:
                    usage_tracker.track_call("parsing", parsing_model_str, usage_metadata)

                raw_parsing_response = (
                    parsing_response.content if hasattr(parsing_response, "content") else str(parsing_response)
                )

                # Strip markdown fences and parse with PydanticOutputParser
                cleaned_response = _strip_markdown_fences(raw_parsing_response)
                if cleaned_response is None:
                    raise ValueError("Empty response from parsing model after markdown fence removal")

                # Try parsing, with null-value retry on failure
                try:
                    parsed_answer = parser.parse(cleaned_response)
                except Exception as parse_error:
                    # Try to recover with null-value feedback
                    logger.warning(f"Initial parsing failed: {parse_error}")
                    retried_answer, retry_usage = _retry_parse_with_null_feedback(
                        parsing_llm=parsing_llm,
                        parser=parser,
                        original_messages=parsing_messages,
                        failed_response=cleaned_response,
                        error=parse_error,
                        usage_tracker=usage_tracker,
                        model_str=parsing_model_str,
                    )

                    if retried_answer is not None:
                        parsed_answer = retried_answer
                    else:
                        # Retry failed, re-raise original error
                        raise parse_error

        except Exception as e:
            error_msg = f"Parsing failed: {e}"
            logger.error(error_msg)
            context.mark_error(error_msg)
            return

        # Store results
        context.set_artifact("parsed_answer", parsed_answer)
        context.set_artifact("deep_judgment_performed", deep_judgment_performed)

        # Store updated usage tracker for next stages
        context.set_artifact("usage_tracker", usage_tracker)
        context.set_artifact("extracted_excerpts", extracted_excerpts)
        context.set_artifact("attribute_reasoning", attribute_reasoning)
        context.set_artifact("deep_judgment_stages_completed", deep_judgment_stages_completed)
        context.set_artifact("deep_judgment_model_calls", deep_judgment_model_calls)
        context.set_artifact("deep_judgment_excerpt_retry_count", deep_judgment_excerpt_retry_count)
        context.set_artifact("attributes_without_excerpts", attributes_without_excerpts)
        context.set_artifact("hallucination_risk_assessment", hallucination_risk_assessment)

        # Also store in result builder
        context.set_result_field("deep_judgment_enabled", context.deep_judgment_enabled)
        context.set_result_field("deep_judgment_performed", deep_judgment_performed)
        context.set_result_field("extracted_excerpts", extracted_excerpts)
        context.set_result_field("attribute_reasoning", attribute_reasoning)
        context.set_result_field("deep_judgment_stages_completed", deep_judgment_stages_completed)
        context.set_result_field("deep_judgment_model_calls", deep_judgment_model_calls)
        context.set_result_field("deep_judgment_excerpt_retry_count", deep_judgment_excerpt_retry_count)
        context.set_result_field("attributes_without_excerpts", attributes_without_excerpts)
        context.set_result_field("deep_judgment_search_enabled", context.deep_judgment_search_enabled)
        context.set_result_field("hallucination_risk_assessment", hallucination_risk_assessment)

        # Store trace filtering metadata in result builder
        context.set_result_field("used_full_trace_for_template", use_full_trace)
        context.set_result_field("template_evaluation_input", template_evaluation_input)
        context.set_result_field("trace_extraction_error", trace_extraction_error)
