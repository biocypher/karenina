"""Abstention detection for identifying when models refuse to answer questions."""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from karenina.adapters import get_llm
from karenina.adapters.registry import close_adapter
from karenina.benchmark.verification.prompts.assembler import PromptAssembler
from karenina.benchmark.verification.prompts.task_types import PromptTask
from karenina.benchmark.verification.prompts.trace.abstention import ABSTENTION_DETECTION_SYS, ABSTENTION_DETECTION_USER
from karenina.benchmark.verification.utils.llm_judge_helpers import extract_judge_result, fallback_json_parse
from karenina.ports import LLMResponse
from karenina.ports.capabilities import PortCapabilities
from karenina.schemas.config import ModelConfig
from karenina.schemas.verification.prompt_config import PromptConfig

# Set up logger
logger = logging.getLogger(__name__)


class AbstentionResult(BaseModel):
    """Result of abstention detection by LLM judge."""

    reasoning: str = Field(description="Brief explanation of why this was classified as abstention or genuine attempt")
    abstention_detected: bool = Field(
        description="True if the model refused to answer or abstained, False if genuine answer attempt"
    )


def detect_abstention(
    raw_llm_response: str,
    parsing_model: ModelConfig,
    question_text: str,
    prompt_config: PromptConfig | None = None,
) -> tuple[bool, bool, str | None, dict[str, Any]]:
    """
    Detect if the model refused to answer or abstained from answering.

    This function uses an LLM to analyze the response and determine if it contains
    patterns indicating refusal, abstention, or evasion. The adapter handles retries
    for transient errors internally via RetryExecutor.

    Args:
        raw_llm_response: The raw response text from the answering model
        parsing_model: Configuration for the model to use for abstention detection
        question_text: The original question that was asked

    Returns:
        Tuple of (abstention_detected, check_performed, reasoning, usage_metadata):
        - abstention_detected: True if model refused/abstained, False if genuine attempt
        - check_performed: True if check completed successfully, False if check failed
        - reasoning: The LLM's explanation for its determination (None if check failed)
        - usage_metadata: Token usage metadata from the LLM invocation

    Examples:
        >>> config = ModelConfig(id="parser", model_provider="openai", ...)
        >>> detected, performed, reasoning, metadata = detect_abstention("I cannot answer this", config, "What is X?")
        >>> print(detected, performed, reasoning)
        True, True, "Response contains explicit refusal pattern"
    """

    usage_metadata: dict[str, Any] = {}
    llm: Any | None = None
    structured_llm: Any | None = None

    try:
        # Create config copy with temperature=0 for consistent detection
        # Note: LangChain adapter respects temperature; Claude SDK adapter ignores it
        detection_config = parsing_model.model_copy(update={"temperature": 0.0})

        # Get LLM via adapter factory (adapter handles retries internally)
        llm = get_llm(detection_config)

        # Configure for structured output
        structured_llm = llm.with_structured_output(AbstentionResult)

        # Build messages using PromptAssembler (tri-section pattern)
        user_prompt = ABSTENTION_DETECTION_USER.format(question=question_text, response=raw_llm_response)
        assembler = PromptAssembler(
            task=PromptTask.ABSTENTION_DETECTION,
            interface=parsing_model.interface,
            capabilities=PortCapabilities(),
        )
        user_instructions = prompt_config.get_for_task(PromptTask.ABSTENTION_DETECTION.value) if prompt_config else None
        messages = assembler.assemble(
            system_text=ABSTENTION_DETECTION_SYS,
            user_text=user_prompt,
            user_instructions=user_instructions,
        )

        # Invoke with structured output
        response: LLMResponse = structured_llm.invoke(messages)
        usage_metadata = response.usage.to_dict()

        # Extract result from structured output or fall back to manual parsing
        result = extract_judge_result(response, AbstentionResult, "abstention_detected")
        if result is not None:
            logger.debug("Abstention check: %s - Reasoning: %s", result.abstention_detected, result.reasoning)
            return result.abstention_detected, True, result.reasoning, usage_metadata

        # Fallback: manual JSON parsing from content
        return fallback_json_parse(
            response.content, usage_metadata, "abstention_detected", False, "Abstention check (fallback)"
        )

    except json.JSONDecodeError as e:
        # JSON parsing failed: log and treat as check failure
        logger.warning("Failed to parse abstention detection response as JSON: %s", e)
        return False, False, None, usage_metadata

    except Exception as e:
        # Non-recoverable error: log and treat as check failure
        logger.warning("Abstention detection failed: %s", e)
        return False, False, None, usage_metadata
    finally:
        if structured_llm is not None:
            close_adapter(structured_llm)
        if llm is not None and llm is not structured_llm:
            close_adapter(llm)
