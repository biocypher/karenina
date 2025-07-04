"""Orchestration logic for multi-model verification."""


from ...schemas.rubric_class import Rubric
from ..models import ModelConfiguration, VerificationConfig, VerificationResult
from .runner import run_single_model_verification


def run_question_verification(
    question_id: str, question_text: str, template_code: str, config: VerificationConfig, rubric: Rubric | None = None
) -> dict[str, VerificationResult]:
    """
    Run verification for a single question with all model combinations.

    Args:
        question_id: Unique identifier for the question
        question_text: The question to ask the LLM
        template_code: Python code defining the Answer class
        config: Verification configuration with multiple models
        rubric: Optional rubric for qualitative evaluation

    Returns:
        Dictionary of VerificationResult keyed by combination ID
    """
    results = {}

    # Handle legacy single model config
    if hasattr(config, "answering_model_provider") and config.answering_model_provider:
        # Legacy single model mode - create single model configs and handle replicates
        answering_model = ModelConfiguration(
            id="answering-legacy",
            model_provider=config.answering_model_provider,
            model_name=config.answering_model_name,
            temperature=config.answering_temperature or 0.1,
            interface=config.answering_interface or "langchain",
            system_prompt=config.answering_system_prompt
            or "You are an expert assistant. Answer the question accurately and concisely.",
        )

        parsing_model = ModelConfiguration(
            id="parsing-legacy",
            model_provider=config.parsing_model_provider,
            model_name=config.parsing_model_name,
            temperature=config.parsing_temperature or 0.1,
            interface=config.parsing_interface or "langchain",
            system_prompt=config.parsing_system_prompt
            or "You are a validation assistant. Parse and validate responses against the given Pydantic template.",
        )

        # Run with replicates for legacy mode too
        for replicate in range(1, getattr(config, "replicate_count", 1) + 1):
            # For single replicate, don't include replicate numbers
            if getattr(config, "replicate_count", 1) == 1:
                result_key = f"{question_id}_{answering_model.id}_{parsing_model.id}"
                results[result_key] = run_single_model_verification(
                    question_id=question_id,
                    question_text=question_text,
                    template_code=template_code,
                    answering_model=answering_model,
                    parsing_model=parsing_model,
                    run_name=getattr(config, "run_name", None),
                    job_id=getattr(config, "job_id", None),
                    rubric=rubric,
                )
            else:
                # Include replicate numbers for multiple replicates
                result_key = f"{question_id}_{answering_model.id}_{parsing_model.id}_rep{replicate}"
                results[result_key] = run_single_model_verification(
                    question_id=question_id,
                    question_text=question_text,
                    template_code=template_code,
                    answering_model=answering_model,
                    parsing_model=parsing_model,
                    run_name=getattr(config, "run_name", None),
                    job_id=getattr(config, "job_id", None),
                    answering_replicate=replicate,
                    parsing_replicate=replicate,
                    rubric=rubric,
                )

    else:
        # New multi-model mode
        answering_models = getattr(config, "answering_models", [])
        parsing_models = getattr(config, "parsing_models", [])
        replicate_count = getattr(config, "replicate_count", 1)

        for answering_model in answering_models:
            for parsing_model in parsing_models:
                for replicate in range(1, replicate_count + 1):
                    # For single replicate, don't include replicate numbers in the key
                    if replicate_count == 1:
                        result_key = f"{question_id}_{answering_model.id}_{parsing_model.id}"
                        results[result_key] = run_single_model_verification(
                            question_id=question_id,
                            question_text=question_text,
                            template_code=template_code,
                            answering_model=answering_model,
                            parsing_model=parsing_model,
                            run_name=getattr(config, "run_name", None),
                            job_id=getattr(config, "job_id", None),
                            rubric=rubric,
                        )
                    else:
                        # Include replicate numbers for multiple replicates
                        result_key = f"{question_id}_{answering_model.id}_{parsing_model.id}_rep{replicate}"
                        results[result_key] = run_single_model_verification(
                            question_id=question_id,
                            question_text=question_text,
                            template_code=template_code,
                            answering_model=answering_model,
                            parsing_model=parsing_model,
                            run_name=getattr(config, "run_name", None),
                            job_id=getattr(config, "job_id", None),
                            answering_replicate=replicate,
                            parsing_replicate=replicate,
                            rubric=rubric,
                        )

    return results
