"""Single model verification runner."""

import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from ...answers.generator import inject_question_id_into_answer_class
from ...llm.interface import init_chat_model_unified
from ..models import ModelConfiguration, VerificationResult
from .validation import validate_answer_template


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
    md5_pattern = re.compile(r'^[a-fA-F0-9]{32}$')
    return bool(md5_pattern.match(hash_string))


def run_single_model_verification(
    question_id: str,
    question_text: str,
    template_code: str,
    answering_model: ModelConfiguration,
    parsing_model: ModelConfiguration,
    run_name: str | None = None,
    job_id: str | None = None,
    answering_replicate: int | None = None,
    parsing_replicate: int | None = None,
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

    Returns:
        VerificationResult with all details
    
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
        if not is_valid:
            return VerificationResult(
                question_id=question_id,
                success=False,
                error=f"Template validation failed: {error_msg}",
                question_text=question_text,
                raw_llm_response="",
                answering_model=answering_model_str,
                parsing_model=parsing_model_str,
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
        messages = []
        if answering_model.system_prompt:
            messages.append(SystemMessage(content=answering_model.system_prompt))
        messages.append(HumanMessage(content=question_text))

        try:
            response = answering_llm.invoke(messages)
            raw_llm_response = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            return VerificationResult(
                question_id=question_id,
                success=False,
                error=f"LLM call failed: {e}",
                question_text=question_text,
                raw_llm_response="",
                answering_model=answering_model_str,
                parsing_model=parsing_model_str,
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

        # Create structured output parser
        try:
            structured_llm = parsing_llm.with_structured_output(Answer)
        except Exception as e:
            return VerificationResult(
                question_id=question_id,
                success=False,
                error=f"Failed to create structured output parser: {e}",
                question_text=question_text,
                raw_llm_response=raw_llm_response,
                answering_model=answering_model_str,
                parsing_model=parsing_model_str,
                execution_time=time.time() - start_time,
                timestamp=timestamp,
                answering_system_prompt=answering_model.system_prompt,
                parsing_system_prompt=parsing_model.system_prompt,
                run_name=run_name,
                job_id=job_id,
                answering_replicate=answering_replicate,
                parsing_replicate=parsing_replicate,
            )

        # Create parsing prompt
        parsing_prompt = f"""Parse the following response into the specified format.

Response to parse: {raw_llm_response}

Follow the schema exactly as defined."""

        parsing_messages = []
        if parsing_model.system_prompt:
            parsing_messages.append(SystemMessage(content=parsing_model.system_prompt))
        parsing_messages.append(HumanMessage(content=parsing_prompt))

        try:
            parsed_answer = structured_llm.invoke(parsing_messages)
        except Exception as e:
            return VerificationResult(
                question_id=question_id,
                success=False,
                error=f"Parsing failed: {e}",
                question_text=question_text,
                raw_llm_response=raw_llm_response,
                answering_model=answering_model_str,
                parsing_model=parsing_model_str,
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
                question_text=question_text,
                raw_llm_response=raw_llm_response,
                answering_model=answering_model_str,
                parsing_model=parsing_model_str,
                parsed_response=parsed_answer.model_dump()
                if hasattr(parsed_answer, "model_dump")
                else str(parsed_answer),
                execution_time=time.time() - start_time,
                timestamp=timestamp,
                answering_system_prompt=answering_model.system_prompt,
                parsing_system_prompt=parsing_model.system_prompt,
                run_name=run_name,
                job_id=job_id,
                answering_replicate=answering_replicate,
                parsing_replicate=parsing_replicate,
            )

        return VerificationResult(
            question_id=question_id,
            success=True,
            verify_result=verification_result,
            question_text=question_text,
            raw_llm_response=raw_llm_response,
            answering_model=answering_model_str,
            parsing_model=parsing_model_str,
            parsed_response=parsed_answer.model_dump() if hasattr(parsed_answer, "model_dump") else str(parsed_answer),
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
            question_text=question_text,
            raw_llm_response="",
            answering_model=answering_model_str,
            parsing_model=parsing_model_str,
            execution_time=time.time() - start_time,
            timestamp=timestamp,
            answering_system_prompt=answering_model.system_prompt,
            parsing_system_prompt=parsing_model.system_prompt,
            run_name=run_name,
            job_id=job_id,
            answering_replicate=answering_replicate,
            parsing_replicate=parsing_replicate,
        )
