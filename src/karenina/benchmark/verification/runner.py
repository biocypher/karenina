"""Single model verification runner."""

import os
import re
import time
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from ...answers.generator import inject_question_id_into_answer_class
from ...llm.interface import init_chat_model_unified
from ...schemas.rubric_class import Rubric
from ..models import ModelConfig, VerificationResult
from .rubric_evaluator import RubricEvaluator
from .validation import validate_answer_template


def _should_expose_ground_truth() -> bool:
    """
    Check if ground truth should be exposed to the parser model.

    Reads from the KARENINA_EXPOSE_GROUND_TRUTH environment variable.
    Defaults to False for backward compatibility.

    Returns:
        True if ground truth should be exposed, False otherwise
    """
    return os.getenv("KARENINA_EXPOSE_GROUND_TRUTH", "false").lower() in ("true", "1", "yes", "on")


def _split_parsed_response(parsed_answer: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Split parsed answer into ground truth and LLM response components.

    Args:
        parsed_answer: The parsed answer object from LLM

    Returns:
        Tuple of (parsed_gt_response, parsed_llm_response)
        - parsed_gt_response: The 'correct' field content (ground truth)
        - parsed_llm_response: All other fields except 'id' and 'correct'
    """
    if not parsed_answer or not hasattr(parsed_answer, "model_dump"):
        return None, None

    parsed_dict = parsed_answer.model_dump()

    # Extract ground truth from 'correct' field
    parsed_gt_response = parsed_dict.get("correct")

    # Create LLM response by excluding 'id' and 'correct'
    parsed_llm_response = {k: v for k, v in parsed_dict.items() if k not in ("id", "correct")}

    return parsed_gt_response, parsed_llm_response


def _is_valid_md5_hash(hash_string: str) -> bool:
    """
    Validate that a string is a proper MD5 hash format.

    Args:
        hash_string: String to validate

    Returns:
        True if valid MD5 hash format, False otherwise
    """
    if not isinstance(hash_string, str):
        return False

    # MD5 hash is exactly 32 hexadecimal characters
    md5_pattern = re.compile(r"^[a-fA-F0-9]{32}$")
    return bool(md5_pattern.match(hash_string))


def _strip_markdown_fences(text: str) -> str:
    """
    Remove markdown JSON code fences from LLM response text.

    Args:
        text: Raw text response from LLM that may contain markdown fences

    Returns:
        Cleaned text with markdown fences removed
    """
    if not isinstance(text, str):
        return text

    # Strip leading and trailing markdown JSON fences
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]  # Remove ```json
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]  # Remove ```

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]  # Remove trailing ```

    return cleaned.strip()


def _construct_few_shot_prompt(
    question_text: str, few_shot_examples: list[dict[str, str]] | None, few_shot_enabled: bool
) -> str:
    """
    Construct a prompt with few-shot examples if enabled.

    Args:
        question_text: The main question to ask
        few_shot_examples: List of question-answer pairs for few-shot prompting
        few_shot_enabled: Whether few-shot prompting is enabled

    Returns:
        The constructed prompt with optional few-shot examples
    """
    if not few_shot_enabled or not few_shot_examples:
        return question_text

    # Build the prompt with examples
    prompt_parts = []

    for example in few_shot_examples:
        if "question" in example and "answer" in example:
            prompt_parts.append(f"Question: {example['question']}")
            prompt_parts.append(f"Answer: {example['answer']}")
            prompt_parts.append("")  # Empty line for separation

    # Add the actual question
    prompt_parts.append(f"Question: {question_text}")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)


def _system_prompt_compose(
    system_prompt: str | None, format_instructions: str, ground_truth: dict[str, Any] | None = None
) -> str:
    """
    Compose a system prompt with format instructions and optional ground truth information.

    Args:
        system_prompt: The system prompt to compose
        format_instructions: The format instructions to compose
        ground_truth: Optional ground truth information to include for parsing assistance

    Returns:
        The composed system prompt
    """
    prompt_parts = [
        f"<general_instructions>\n{system_prompt if system_prompt else ''}\n</general_instructions>",
        f"<format_instructions>\n{format_instructions}\n</format_instructions>",
    ]

    # Add ground truth instructions if provided
    if ground_truth is not None:
        import json

        ground_truth_str = json.dumps(ground_truth, indent=2, default=str)

        ground_truth_section = f"""<ground_truth_reference>
The following ground truth information is provided as reference to help with semantic matching and disambiguation.
Use this information carefully - do not blindly copy it, but it may help resolve ambiguities when the trace
and template are semantically close but differ in exact wording.

Ground Truth:
{ground_truth_str}
</ground_truth_reference>"""

        prompt_parts.append(ground_truth_section)

    return "\n\n".join(prompt_parts)


def run_single_model_verification(
    question_id: str,
    question_text: str,
    template_code: str,
    answering_model: ModelConfig,
    parsing_model: ModelConfig,
    run_name: str | None = None,
    job_id: str | None = None,
    answering_replicate: int | None = None,
    parsing_replicate: int | None = None,
    rubric: Rubric | None = None,
    keywords: list[str] | None = None,
    few_shot_examples: list[dict[str, str]] | None = None,
    few_shot_enabled: bool = False,
) -> VerificationResult:
    """
    Run verification for a single question with specific answering and parsing models.

    Args:
        question_id: Unique identifier for the question. For manual interface, this MUST be
                    a 32-character hexadecimal MD5 hash (generated during question extraction).
        question_text: The question to ask the LLM
        template_code: Python code defining the Answer class
        answering_model: Configuration for the answering model
        parsing_model: Configuration for the parsing model
        run_name: Optional run name for tracking
        job_id: Optional job ID for tracking
        answering_replicate: Optional replicate number for answering model
        parsing_replicate: Optional replicate number for parsing model
        rubric: Optional rubric for qualitative evaluation
        few_shot_examples: Optional list of question-answer pairs for few-shot prompting
        few_shot_enabled: Whether to use few-shot prompting (disabled by default)

    Returns:
        VerificationResult with all details and optional rubric scores

    Raises:
        ValueError: If question_id is not a valid MD5 hash when using manual interface
    """
    from datetime import datetime

    start_time = time.time()
    timestamp = datetime.now().isoformat()

    # For OpenRouter interface, don't include provider in the model string
    if answering_model.interface == "openrouter":
        answering_model_str = answering_model.model_name
    else:
        answering_model_str = f"{answering_model.model_provider}/{answering_model.model_name}"

    if parsing_model.interface == "openrouter":
        parsing_model_str = parsing_model.model_name
    else:
        parsing_model_str = f"{parsing_model.model_provider}/{parsing_model.model_name}"

    try:
        # Step 1: Validate the template
        is_valid, error_msg, RawAnswer = validate_answer_template(template_code)
        if not is_valid or RawAnswer is None:
            return VerificationResult(
                question_id=question_id,
                success=False,
                error=f"Template validation failed: {error_msg}",
                keywords=keywords,
                question_text=question_text,
                raw_llm_response="",
                parsed_gt_response=None,
                parsed_llm_response=None,
                answering_model=answering_model_str,
                parsing_model=parsing_model_str,
                evaluation_rubric=rubric.model_dump() if rubric else None,
                execution_time=time.time() - start_time,
                timestamp=timestamp,
                answering_system_prompt=answering_model.system_prompt,
                parsing_system_prompt=parsing_model.system_prompt,
                run_name=run_name,
                job_id=job_id,
                answering_replicate=answering_replicate,
                parsing_replicate=parsing_replicate,
            )

        # Step 1.5: Inject question ID into the Answer class
        Answer = inject_question_id_into_answer_class(RawAnswer, question_id)

        # Step 2: Initialize answering model
        # For manual interface, pass the question_id as question_hash
        # Note: question_id should be an MD5 hash from the question extraction process
        if answering_model.interface == "manual":
            # Validate that question_id is indeed an MD5 hash for manual interface
            if not _is_valid_md5_hash(question_id):
                raise ValueError(
                    f"Invalid question_id format for manual interface: '{question_id}'. "
                    "question_id must be a 32-character hexadecimal MD5 hash when using manual interface. "
                    "This hash is typically generated during question extraction from the question text."
                )

            answering_llm = init_chat_model_unified(
                model=answering_model.model_name,
                provider=answering_model.model_provider,
                temperature=answering_model.temperature,
                interface=answering_model.interface,
                question_hash=question_id,
            )
        else:
            answering_llm = init_chat_model_unified(
                model=answering_model.model_name,
                provider=answering_model.model_provider,
                temperature=answering_model.temperature,
                interface=answering_model.interface,
            )

        # Step 3: Get LLM response
        messages: list[BaseMessage] = []
        if answering_model.system_prompt:
            messages.append(SystemMessage(content=answering_model.system_prompt))

        # Construct prompt with optional few-shot examples
        constructed_prompt = _construct_few_shot_prompt(question_text, few_shot_examples, few_shot_enabled)
        messages.append(HumanMessage(content=constructed_prompt))

        try:
            response = answering_llm.invoke(messages)
            raw_llm_response = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            return VerificationResult(
                question_id=question_id,
                success=False,
                error=f"LLM call failed: {e}",
                keywords=keywords,
                question_text=question_text,
                raw_llm_response="",
                parsed_gt_response=None,
                parsed_llm_response=None,
                answering_model=answering_model_str,
                parsing_model=parsing_model_str,
                evaluation_rubric=rubric.model_dump() if rubric else None,
                execution_time=time.time() - start_time,
                timestamp=timestamp,
                answering_system_prompt=answering_model.system_prompt,
                parsing_system_prompt=parsing_model.system_prompt,
                run_name=run_name,
                job_id=job_id,
                answering_replicate=answering_replicate,
                parsing_replicate=parsing_replicate,
            )

        # Step 4: Initialize parsing model and parse response
        parsing_llm = init_chat_model_unified(
            model=parsing_model.model_name,
            provider=parsing_model.model_provider,
            temperature=parsing_model.temperature,
            interface=parsing_model.interface,
        )

        # Create PydanticOutputParser
        try:
            parser: Any = PydanticOutputParser(pydantic_object=Answer)
        except Exception as e:
            return VerificationResult(
                question_id=question_id,
                success=False,
                error=f"Failed to create PydanticOutputParser: {e}",
                keywords=keywords,
                question_text=question_text,
                raw_llm_response=raw_llm_response,
                parsed_gt_response=None,
                parsed_llm_response=None,
                answering_model=answering_model_str,
                parsing_model=parsing_model_str,
                evaluation_rubric=rubric.model_dump() if rubric else None,
                execution_time=time.time() - start_time,
                timestamp=timestamp,
                answering_system_prompt=answering_model.system_prompt,
                parsing_system_prompt=parsing_model.system_prompt,
                run_name=run_name,
                job_id=job_id,
                answering_replicate=answering_replicate,
                parsing_replicate=parsing_replicate,
            )

        # Extract ground truth if enabled
        ground_truth = None
        if _should_expose_ground_truth():
            try:
                from .template_utils import create_test_instance_from_answer_class

                # Create test instance and extract ground truth
                _, ground_truth = create_test_instance_from_answer_class(RawAnswer)
            except Exception as e:
                # If we can't extract ground truth, continue without it
                # This ensures the feature is robust and doesn't break existing functionality
                print(f"Warning: Could not extract ground truth for question {question_id}: {e}")

        # Create parsing prompt with format instructions and optional ground truth
        format_instructions = parser.get_format_instructions()
        combined_system_prompt = _system_prompt_compose(parsing_model.system_prompt, format_instructions, ground_truth)

        # Construct the parsing prompt (user message)
        parsing_prompt = f"""<response_to_parse>
{raw_llm_response}
</response_to_parse>"""

        parsing_messages: list[BaseMessage] = []
        if combined_system_prompt:
            parsing_messages.append(SystemMessage(content=combined_system_prompt))
        parsing_messages.append(HumanMessage(content=parsing_prompt))

        try:
            # Get raw text response from parsing LLM
            parsing_response = parsing_llm.invoke(parsing_messages)
            raw_parsing_response = (
                parsing_response.content if hasattr(parsing_response, "content") else str(parsing_response)
            )

            # Strip markdown fences and parse with PydanticOutputParser
            cleaned_response = _strip_markdown_fences(raw_parsing_response)
            parsed_answer = parser.parse(cleaned_response)
        except Exception as e:
            return VerificationResult(
                question_id=question_id,
                success=False,
                error=f"Parsing failed: {e}",
                keywords=keywords,
                question_text=question_text,
                raw_llm_response=raw_llm_response,
                parsed_gt_response=None,
                parsed_llm_response=None,
                answering_model=answering_model_str,
                parsing_model=parsing_model_str,
                evaluation_rubric=rubric.model_dump() if rubric else None,
                execution_time=time.time() - start_time,
                timestamp=timestamp,
                answering_system_prompt=answering_model.system_prompt,
                parsing_system_prompt=parsing_model.system_prompt,
                run_name=run_name,
                job_id=job_id,
                answering_replicate=answering_replicate,
                parsing_replicate=parsing_replicate,
            )

        # Step 5: Run verification
        try:
            verification_result = parsed_answer.verify()
        except Exception as e:
            return VerificationResult(
                question_id=question_id,
                success=False,
                error=f"Verification failed: {e}",
                keywords=keywords,
                question_text=question_text,
                raw_llm_response=raw_llm_response,
                answering_model=answering_model_str,
                parsing_model=parsing_model_str,
                parsed_gt_response=_split_parsed_response(parsed_answer)[0],
                parsed_llm_response=_split_parsed_response(parsed_answer)[1],
                evaluation_rubric=rubric.model_dump() if rubric else None,
                execution_time=time.time() - start_time,
                timestamp=timestamp,
                answering_system_prompt=answering_model.system_prompt,
                parsing_system_prompt=parsing_model.system_prompt,
                run_name=run_name,
                job_id=job_id,
                answering_replicate=answering_replicate,
                parsing_replicate=parsing_replicate,
            )

        # Step 6: Run rubric evaluation (optional)
        rubric_result = None
        if rubric and rubric.traits:
            try:
                # Use parsing model for rubric evaluation
                evaluator = RubricEvaluator(parsing_model)
                rubric_result = evaluator.evaluate_rubric(
                    question=question_text, answer=raw_llm_response, rubric=rubric
                )
            except (ValueError, RuntimeError) as e:
                # Handle specific rubric evaluator errors
                print(f"Warning: Rubric evaluator initialization/configuration failed for question {question_id}: {e}")
                rubric_result = None
            except Exception as e:
                # Don't fail the entire verification if rubric evaluation fails
                print(f"Warning: Rubric evaluation failed for question {question_id}: {e}")
                rubric_result = None

        return VerificationResult(
            question_id=question_id,
            success=True,
            verify_result=verification_result,
            verify_rubric=rubric_result,
            evaluation_rubric=rubric.model_dump() if rubric else None,
            keywords=keywords,
            question_text=question_text,
            raw_llm_response=raw_llm_response,
            answering_model=answering_model_str,
            parsing_model=parsing_model_str,
            parsed_gt_response=_split_parsed_response(parsed_answer)[0],
            parsed_llm_response=_split_parsed_response(parsed_answer)[1],
            execution_time=time.time() - start_time,
            timestamp=timestamp,
            answering_system_prompt=answering_model.system_prompt,
            parsing_system_prompt=parsing_model.system_prompt,
            run_name=run_name,
            job_id=job_id,
            answering_replicate=answering_replicate,
            parsing_replicate=parsing_replicate,
        )

    except Exception as e:
        return VerificationResult(
            question_id=question_id,
            success=False,
            error=f"Unexpected error: {e}",
            keywords=keywords,
            question_text=question_text,
            raw_llm_response="",
            parsed_gt_response=None,
            parsed_llm_response=None,
            answering_model=answering_model_str,
            parsing_model=parsing_model_str,
            evaluation_rubric=rubric.model_dump() if rubric else None,
            execution_time=time.time() - start_time,
            timestamp=timestamp,
            answering_system_prompt=answering_model.system_prompt,
            parsing_system_prompt=parsing_model.system_prompt,
            run_name=run_name,
            job_id=job_id,
            answering_replicate=answering_replicate,
            parsing_replicate=parsing_replicate,
        )
