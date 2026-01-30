"""Single model verification runner.

Main entry point for running verification using the stage-based pipeline architecture.
"""

from typing import Any

from ...schemas.domain import Rubric
from ...schemas.workflow import ModelConfig, VerificationResult
from ...utils.checkpoint import generate_template_id
from .stages import StageOrchestrator, VerificationContext


def run_single_model_verification(
    question_id: str,
    question_text: str,
    template_code: str,
    answering_model: ModelConfig,
    parsing_model: ModelConfig,
    run_name: str | None = None,
    replicate: int | None = None,
    rubric: Rubric | None = None,
    keywords: list[str] | None = None,
    raw_answer: str | None = None,
    few_shot_examples: list[dict[str, str]] | None = None,
    few_shot_enabled: bool = False,
    abstention_enabled: bool = False,
    sufficiency_enabled: bool = False,
    deep_judgment_enabled: bool = False,
    rubric_evaluation_strategy: str = "batch",
    deep_judgment_max_excerpts_per_attribute: int = 3,
    deep_judgment_fuzzy_match_threshold: float = 0.80,
    deep_judgment_excerpt_retry_attempts: int = 2,
    deep_judgment_search_enabled: bool = False,
    deep_judgment_search_tool: str | Any = "tavily",
    # Deep-judgment rubric configuration (NEW)
    deep_judgment_rubric_mode: str = "disabled",
    deep_judgment_rubric_global_excerpts: bool = True,
    deep_judgment_rubric_config: dict[str, Any] | None = None,
    deep_judgment_rubric_max_excerpts_default: int = 7,
    deep_judgment_rubric_fuzzy_match_threshold_default: float = 0.80,
    deep_judgment_rubric_excerpt_retry_attempts_default: int = 2,
    deep_judgment_rubric_search_enabled: bool = False,
    deep_judgment_rubric_search_tool: str | Any = "tavily",
    evaluation_mode: str = "template_only",
    cached_answer_data: dict[str, Any] | None = None,
    # Trace filtering configuration (MCP Agent Evaluation)
    use_full_trace_for_template: bool = False,
    use_full_trace_for_rubric: bool = True,
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
        replicate: Optional replicate number for repeated runs of the same question
        rubric: Optional rubric for qualitative evaluation
        keywords: Optional keywords associated with the question
        few_shot_examples: Optional list of question-answer pairs for few-shot prompting
        few_shot_enabled: Whether to use few-shot prompting (disabled by default)
        abstention_enabled: Whether to enable abstention detection
        sufficiency_enabled: Whether to enable trace sufficiency detection
        deep_judgment_enabled: Whether to enable deep-judgment parsing
        rubric_evaluation_strategy: Strategy for evaluating LLM rubric traits:
            - "batch": All traits evaluated in single LLM call (default, efficient)
            - "sequential": Traits evaluated one-by-one (reliable, more expensive)
        deep_judgment_max_excerpts_per_attribute: Max excerpts per attribute (deep-judgment)
        deep_judgment_fuzzy_match_threshold: Similarity threshold for excerpts (deep-judgment)
        deep_judgment_excerpt_retry_attempts: Retry attempts for excerpt validation (deep-judgment)
        deep_judgment_search_enabled: Whether to enable search enhancement (deep-judgment)
        deep_judgment_search_tool: Search tool name or callable (deep-judgment)
        evaluation_mode: Evaluation mode determining which stages run:
            - "template_only": Template verification only (default)
            - "template_and_rubric": Template verification + rubric evaluation
            - "rubric_only": Skip template, only evaluate rubrics on raw response
        cached_answer_data: Optional cached answer data from previous generation.
            If provided, the GenerateAnswerStage will skip LLM invocation and use
            this cached data. Used to share answers across multiple judges.

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
        raw_answer=raw_answer,
        # Run Metadata
        run_name=run_name,
        replicate=replicate,
        # Feature Flags
        few_shot_enabled=few_shot_enabled,
        abstention_enabled=abstention_enabled,
        sufficiency_enabled=sufficiency_enabled,
        deep_judgment_enabled=deep_judgment_enabled,
        # Rubric Configuration
        rubric_evaluation_strategy=rubric_evaluation_strategy,
        # Deep-Judgment Configuration
        deep_judgment_max_excerpts_per_attribute=deep_judgment_max_excerpts_per_attribute,
        deep_judgment_fuzzy_match_threshold=deep_judgment_fuzzy_match_threshold,
        deep_judgment_excerpt_retry_attempts=deep_judgment_excerpt_retry_attempts,
        deep_judgment_search_enabled=deep_judgment_search_enabled,
        deep_judgment_search_tool=deep_judgment_search_tool,
        # Deep-Judgment Rubric Configuration (NEW)
        deep_judgment_rubric_mode=deep_judgment_rubric_mode,
        deep_judgment_rubric_global_excerpts=deep_judgment_rubric_global_excerpts,
        deep_judgment_rubric_config=deep_judgment_rubric_config,
        deep_judgment_rubric_max_excerpts_default=deep_judgment_rubric_max_excerpts_default,
        deep_judgment_rubric_fuzzy_match_threshold_default=deep_judgment_rubric_fuzzy_match_threshold_default,
        deep_judgment_rubric_excerpt_retry_attempts_default=deep_judgment_rubric_excerpt_retry_attempts_default,
        deep_judgment_rubric_search_enabled=deep_judgment_rubric_search_enabled,
        deep_judgment_rubric_search_tool=deep_judgment_rubric_search_tool,
        # Few-Shot Configuration
        few_shot_examples=few_shot_examples,
        # Trace Filtering Configuration (MCP Agent Evaluation)
        use_full_trace_for_template=use_full_trace_for_template,
        use_full_trace_for_rubric=use_full_trace_for_rubric,
        # Answer Caching
        cached_answer_data=cached_answer_data,
    )

    # Compute model strings for result (needed even if validation fails)
    # Centralized formatting via adapter registry
    from karenina.adapters import format_model_string

    answering_model_str = format_model_string(answering_model)
    parsing_model_str = format_model_string(parsing_model)

    # Store model strings in context for early access (e.g., in error cases)
    context.set_artifact("answering_model_str", answering_model_str)
    context.set_artifact("parsing_model_str", parsing_model_str)

    # Extract and store MCP server names for early access (e.g., in error cases)
    answering_mcp_servers = list(answering_model.mcp_urls_dict.keys()) if answering_model.mcp_urls_dict else None
    context.set_artifact("answering_mcp_servers", answering_mcp_servers)
    context.set_result_field("answering_mcp_servers", answering_mcp_servers)

    # Determine evaluation mode automatically if not explicitly set
    # If rubric is provided and mode is template_only, upgrade to template_and_rubric
    if (
        rubric
        and (rubric.llm_traits or rubric.regex_traits or rubric.callable_traits or rubric.metric_traits)
        and evaluation_mode == "template_only"
    ):
        evaluation_mode = "template_and_rubric"

    # Build stage orchestrator from configuration
    orchestrator = StageOrchestrator.from_config(
        rubric=rubric,
        abstention_enabled=abstention_enabled,
        sufficiency_enabled=sufficiency_enabled,
        deep_judgment_enabled=deep_judgment_enabled,
        evaluation_mode=evaluation_mode,
    )

    # Execute verification pipeline
    result = orchestrator.execute(context)

    return result
