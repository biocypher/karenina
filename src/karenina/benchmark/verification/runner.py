"""Single model verification runner.

Main entry point for running verification using the stage-based pipeline architecture.
"""

from typing import Any

from ...schemas.rubric_class import Rubric
from ...utils.checkpoint_converter import generate_template_id
from ..models import ModelConfig, VerificationResult
from .stage import VerificationContext
from .stage_orchestrator import StageOrchestrator


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
    abstention_enabled: bool = False,
    deep_judgment_enabled: bool = False,
    deep_judgment_max_excerpts_per_attribute: int = 3,
    deep_judgment_fuzzy_match_threshold: float = 0.80,
    deep_judgment_excerpt_retry_attempts: int = 2,
    deep_judgment_search_enabled: bool = False,
    deep_judgment_search_tool: str | Any = "tavily",
) -> VerificationResult:
    """
    Run verification for a single question with specific answering and parsing models.

    This function uses a stage-based pipeline architecture for modularity and testability.
    Each verification step (validation, generation, parsing, verification, etc.) is
    implemented as a discrete stage that can be independently tested and configured.

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
        keywords: Optional keywords associated with the question
        few_shot_examples: Optional list of question-answer pairs for few-shot prompting
        few_shot_enabled: Whether to use few-shot prompting (disabled by default)
        abstention_enabled: Whether to enable abstention detection
        deep_judgment_enabled: Whether to enable deep-judgment parsing
        deep_judgment_max_excerpts_per_attribute: Max excerpts per attribute (deep-judgment)
        deep_judgment_fuzzy_match_threshold: Similarity threshold for excerpts (deep-judgment)
        deep_judgment_excerpt_retry_attempts: Retry attempts for excerpt validation (deep-judgment)
        deep_judgment_search_enabled: Whether to enable search enhancement (deep-judgment)
        deep_judgment_search_tool: Search tool name or callable (deep-judgment)

    Returns:
        VerificationResult with all details and optional rubric scores

    Raises:
        ValueError: If question_id is not a valid MD5 hash when using manual interface
        RuntimeError: If stage orchestration fails critically
    """
    # Compute template_id from template_code (composite key component)
    template_id = generate_template_id(template_code)

    # Initialize verification context with all parameters
    context = VerificationContext(
        # Identity & Metadata
        question_id=question_id,
        template_id=template_id,
        question_text=question_text,
        template_code=template_code,
        # Configuration
        answering_model=answering_model,
        parsing_model=parsing_model,
        rubric=rubric,
        keywords=keywords,
        # Run Metadata
        run_name=run_name,
        job_id=job_id,
        answering_replicate=answering_replicate,
        parsing_replicate=parsing_replicate,
        # Feature Flags
        few_shot_enabled=few_shot_enabled,
        abstention_enabled=abstention_enabled,
        deep_judgment_enabled=deep_judgment_enabled,
        # Deep-Judgment Configuration
        deep_judgment_max_excerpts_per_attribute=deep_judgment_max_excerpts_per_attribute,
        deep_judgment_fuzzy_match_threshold=deep_judgment_fuzzy_match_threshold,
        deep_judgment_excerpt_retry_attempts=deep_judgment_excerpt_retry_attempts,
        deep_judgment_search_enabled=deep_judgment_search_enabled,
        deep_judgment_search_tool=deep_judgment_search_tool,
        # Few-Shot Configuration
        few_shot_examples=few_shot_examples,
    )

    # Compute model strings for result (needed even if validation fails)
    # For OpenRouter interface, don't include provider in the model string
    if answering_model.interface == "openrouter":
        answering_model_str = answering_model.model_name
    else:
        answering_model_str = f"{answering_model.model_provider}/{answering_model.model_name}"

    if parsing_model.interface == "openrouter":
        parsing_model_str = parsing_model.model_name
    else:
        parsing_model_str = f"{parsing_model.model_provider}/{parsing_model.model_name}"

    # Store model strings in context for early access (e.g., in error cases)
    context.set_artifact("answering_model_str", answering_model_str)
    context.set_artifact("parsing_model_str", parsing_model_str)

    # Build stage orchestrator from configuration
    orchestrator = StageOrchestrator.from_config(
        answering_model=answering_model,
        parsing_model=parsing_model,
        rubric=rubric,
        abstention_enabled=abstention_enabled,
        deep_judgment_enabled=deep_judgment_enabled,
    )

    # Execute verification pipeline
    result = orchestrator.execute(context)

    return result
