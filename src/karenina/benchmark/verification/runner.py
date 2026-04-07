"""Single model verification runner.

Main entry point for running verification using the stage-based pipeline architecture.
"""

import logging
from pathlib import Path
from typing import Any

from karenina.schemas.config import ModelConfig
from karenina.schemas.entities import Rubric
from karenina.schemas.entities.rubric import DynamicRubric
from karenina.schemas.verification import PromptConfig, VerificationResult
from karenina.schemas.verification.config import (
    DEFAULT_DEEP_JUDGMENT_FUZZY_THRESHOLD,
    DEFAULT_DEEP_JUDGMENT_MAX_EXCERPTS,
    DEFAULT_DEEP_JUDGMENT_RETRY_ATTEMPTS,
    DEFAULT_RUBRIC_MAX_EXCERPTS,
    DeepJudgmentRubricCustomConfig,
)
from karenina.utils.checkpoint import generate_template_id
from karenina.utils.errors import ErrorCategory, ErrorRegistry
from karenina.utils.retry_policy import ErrorPatternConfig

from .stages import StageOrchestrator, VerificationContext

logger = logging.getLogger(__name__)


def _build_error_registry(patterns: list[ErrorPatternConfig]) -> ErrorRegistry:
    """Build an ErrorRegistry from declarative error pattern configs.

    Creates a fresh ErrorRegistry with built-in defaults, then registers
    each user-defined pattern from the config.

    Args:
        patterns: List of ErrorPatternConfig from VerificationConfig.custom_error_patterns.

    Returns:
        Configured ErrorRegistry with both built-in and custom patterns.
    """
    registry = ErrorRegistry()
    for pattern_config in patterns:
        registry.register_pattern(
            pattern_config.pattern,
            ErrorCategory(pattern_config.category),
            match_type=pattern_config.match_type,
        )
    return registry


def run_single_model_verification(
    question_id: str,
    question_text: str,
    template_code: str,
    answering_model: ModelConfig,
    parsing_model: ModelConfig,
    run_name: str | None = None,
    replicate: int | None = None,
    rubric: Rubric | None = None,
    dynamic_rubric: DynamicRubric | None = None,
    keywords: list[str] | None = None,
    raw_answer: str | None = None,
    few_shot_examples: list[dict[str, str]] | None = None,
    few_shot_enabled: bool = False,
    abstention_enabled: bool = False,
    sufficiency_enabled: bool = False,
    deep_judgment_mode: str = "disabled",
    rubric_evaluation_strategy: str = "batch",
    deep_judgment_max_excerpts_per_attribute: int = DEFAULT_DEEP_JUDGMENT_MAX_EXCERPTS,
    deep_judgment_fuzzy_match_threshold: float = DEFAULT_DEEP_JUDGMENT_FUZZY_THRESHOLD,
    deep_judgment_excerpt_retry_attempts: int = DEFAULT_DEEP_JUDGMENT_RETRY_ATTEMPTS,
    deep_judgment_search_enabled: bool = False,
    deep_judgment_search_tool: str | Any = "tavily",
    # Deep-judgment rubric configuration (NEW)
    deep_judgment_rubric_mode: str = "disabled",
    deep_judgment_rubric_global_excerpts: bool = True,
    deep_judgment_rubric_config: DeepJudgmentRubricCustomConfig | None = None,
    deep_judgment_rubric_max_excerpts_default: int = DEFAULT_RUBRIC_MAX_EXCERPTS,
    deep_judgment_rubric_fuzzy_match_threshold_default: float = DEFAULT_DEEP_JUDGMENT_FUZZY_THRESHOLD,
    deep_judgment_rubric_excerpt_retry_attempts_default: int = DEFAULT_DEEP_JUDGMENT_RETRY_ATTEMPTS,
    deep_judgment_rubric_search_enabled: bool = False,
    deep_judgment_rubric_search_tool: str | Any = "tavily",
    evaluation_mode: str = "template_only",
    cached_answer_data: dict[str, Any] | None = None,
    # Prompt configuration
    prompt_config: PromptConfig | None = None,
    # Trace filtering configuration (MCP Agent Evaluation)
    use_full_trace_for_template: bool = False,
    use_full_trace_for_rubric: bool = True,
    # Agentic parsing configuration
    agentic_parsing: bool = False,
    agentic_judge_context: str = "workspace_only",
    agentic_parsing_max_turns: int = 15,
    agentic_parsing_timeout: float = 120.0,
    workspace_root: Path | None = None,
    workspace_copy: bool = True,
    workspace_cleanup: bool = True,
    question_workspace_path: str | None = None,
    # Agentic rubric evaluation configuration
    agentic_rubric_strategy: str = "individual",
    agentic_rubric_parallel: bool = False,
    # Embedding check configuration
    embedding_check_enabled: bool = False,
    embedding_check_model: str | None = None,
    embedding_check_threshold: float | None = None,
    # Trait provenance
    trait_provenance: dict[str, str] | None = None,
    # Error classification
    custom_error_patterns: list[ErrorPatternConfig] | None = None,
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
        deep_judgment_mode: Template deep-judgment mode ("disabled", "reasoning_only", "full")
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
        dynamic_rubric=dynamic_rubric,
        keywords=keywords,
        raw_answer=raw_answer,
        # Run Metadata
        run_name=run_name,
        replicate=replicate,
        # Feature Flags
        few_shot_enabled=few_shot_enabled,
        abstention_enabled=abstention_enabled,
        sufficiency_enabled=sufficiency_enabled,
        deep_judgment_mode=deep_judgment_mode,
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
        # Prompt Configuration
        prompt_config=prompt_config,
        # Trace Filtering Configuration (MCP Agent Evaluation)
        use_full_trace_for_template=use_full_trace_for_template,
        use_full_trace_for_rubric=use_full_trace_for_rubric,
        # Answer Caching
        cached_answer_data=cached_answer_data,
        # Agentic Parsing
        agentic_parsing=agentic_parsing,
        agentic_judge_context=agentic_judge_context,
        agentic_parsing_max_turns=agentic_parsing_max_turns,
        agentic_parsing_timeout=agentic_parsing_timeout,
        question_workspace_path=question_workspace_path,
        workspace_root=workspace_root,
        workspace_copy=workspace_copy,
        workspace_cleanup=workspace_cleanup,
        # Agentic Rubric
        agentic_rubric_strategy=agentic_rubric_strategy,
        agentic_rubric_parallel=agentic_rubric_parallel,
        # Embedding Check
        embedding_check_enabled=embedding_check_enabled,
        embedding_check_model=embedding_check_model,
        embedding_check_threshold=embedding_check_threshold,
        # Trait Provenance
        trait_provenance=trait_provenance,
        # Error Classification
        error_registry=_build_error_registry(custom_error_patterns or []),
    )

    # Build ModelIdentity objects for pipeline use (needed even if validation fails)
    from karenina.schemas.verification.model_identity import ModelIdentity

    answering_identity = ModelIdentity.from_model_config(answering_model, role="answering")
    parsing_identity = ModelIdentity.from_model_config(parsing_model, role="parsing")

    # Store ModelIdentity objects in context for downstream stages (e.g., finalize_result)
    context.set_artifact("answering_model_identity", answering_identity)
    context.set_artifact("parsing_model_identity", parsing_identity)

    # Store MCP server names as result field for VerificationResultTemplate
    answering_mcp_servers = list(answering_model.mcp_urls_dict.keys()) if answering_model.mcp_urls_dict else None
    context.set_result_field("answering_mcp_servers", answering_mcp_servers)

    # Warn if rubric traits are provided but evaluation_mode won't use them.
    _has_rubric_traits = rubric and (
        rubric.llm_traits
        or rubric.regex_traits
        or rubric.callable_traits
        or rubric.metric_traits
        or rubric.agentic_traits
    )
    _has_dynamic_rubric_traits = dynamic_rubric is not None and not dynamic_rubric.is_empty()
    if (_has_rubric_traits or _has_dynamic_rubric_traits) and evaluation_mode == "template_only":
        logger.warning(
            "Rubric traits were provided but evaluation_mode='template_only'. "
            "Rubric evaluation will be skipped. Set evaluation_mode='template_and_rubric' "
            "to evaluate rubric traits."
        )

    if evaluation_mode == "rubric_only" and not _has_rubric_traits and not _has_dynamic_rubric_traits:
        logger.warning(
            "evaluation_mode='rubric_only' but no rubric traits provided. Rubric evaluation will produce no scores."
        )

    # Build stage orchestrator from configuration
    orchestrator = StageOrchestrator.from_config(
        rubric=rubric,
        dynamic_rubric=dynamic_rubric,
        abstention_enabled=abstention_enabled,
        sufficiency_enabled=sufficiency_enabled,
        deep_judgment_enabled=deep_judgment_mode != "disabled",
        evaluation_mode=evaluation_mode,
        agentic_parsing=agentic_parsing,
    )

    # Execute verification pipeline
    result = orchestrator.execute(context)

    return result
