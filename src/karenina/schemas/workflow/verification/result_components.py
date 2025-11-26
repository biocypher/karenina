"""Component classes for verification results."""

from typing import Any

from pydantic import BaseModel


class VerificationResultMetadata(BaseModel):
    """Core metadata and identification fields for a verification result."""

    question_id: str
    template_id: str  # MD5 of template or "no_template" (composite key component)
    completed_without_errors: bool
    error: str | None = None
    question_text: str
    raw_answer: str | None = None  # Ground truth answer from checkpoint
    keywords: list[str] | None = None  # Keywords associated with the question
    answering_model: str
    parsing_model: str
    answering_system_prompt: str | None = None  # System prompt used for answering model
    parsing_system_prompt: str | None = None  # System prompt used for parsing model
    execution_time: float
    timestamp: str
    run_name: str | None = None
    answering_replicate: int | None = None  # Replicate number for answering model (1, 2, 3, ...)
    parsing_replicate: int | None = None  # Replicate number for parsing model (1, 2, 3, ...)


class VerificationResultTemplate(BaseModel):
    """Template verification and answer generation fields."""

    raw_llm_response: str
    parsed_gt_response: dict[str, Any] | None = None  # Ground truth from 'correct' field
    parsed_llm_response: dict[str, Any] | None = None  # LLM extracted fields (excluding 'id' and 'correct')

    # Verification outcomes
    template_verification_performed: bool = False  # Whether template verification was executed
    verify_result: Any | None = None  # Template verification result (None if template verification skipped)
    verify_granular_result: Any | None = None

    # Embeddings
    embedding_check_performed: bool = False  # Whether embedding check was attempted
    embedding_similarity_score: float | None = None  # Similarity score (0.0 to 1.0)
    embedding_override_applied: bool = False  # Whether embedding check overrode the result
    embedding_model_used: str | None = None  # Name of the embedding model used

    # Regex checks
    regex_validations_performed: bool = False  # Whether regex validation was attempted
    regex_validation_results: dict[str, bool] | None = None  # Individual regex pattern results
    regex_validation_details: dict[str, dict[str, Any]] | None = None  # Detailed regex match information
    regex_overall_success: bool | None = None  # Overall regex validation result
    regex_extraction_results: dict[str, Any] | None = None  # What the regex patterns actually extracted

    # Recursion limit metadata
    recursion_limit_reached: bool = False  # Whether agent hit recursion limit

    # Abstention
    abstention_check_performed: bool = False  # Whether abstention check was attempted
    abstention_detected: bool | None = None  # Whether model refused/abstained from answering
    abstention_override_applied: bool = False  # Whether abstention check overrode the result
    abstention_reasoning: str | None = None  # LLM's reasoning for abstention determination

    # MCP
    answering_mcp_servers: list[str] | None = None  # Names of MCP servers attached to answering model

    # Usage
    usage_metadata: dict[str, dict[str, Any]] | None = None  # Token usage breakdown by verification stage
    # Structure: {
    #   "answer_generation": {
    #     "input_tokens": 150, "output_tokens": 200, "total_tokens": 350,
    #     "model": "gpt-4.1-mini-2025-04-14",
    #     "input_token_details": {"audio": 0, "cache_read": 0},
    #     "output_token_details": {"audio": 0, "reasoning": 0}
    #   },
    #   "parsing": {...}, "rubric_evaluation": {...}, "abstention_check": {...},
    #   "total": {"input_tokens": 600, "output_tokens": 360, "total_tokens": 960}
    # }
    agent_metrics: dict[str, Any] | None = None  # MCP agent execution metrics (only if agent used)
    # Structure: {
    #   "iterations": 3,  # Number of agent think-act cycles
    #   "tool_calls": 5,  # Total tool invocations
    #   "tools_used": ["mcp__brave_search", "mcp__read_resource"],  # Unique tool names used
    #   "suspect_failed_tool_calls": 2,  # Count of tool calls with error-like output patterns
    #   "suspect_failed_tools": ["mcp__brave_search"]  # List of tools with suspected failures
    # }


class VerificationResultRubric(BaseModel):
    """Rubric evaluation fields with split trait types."""

    rubric_evaluation_performed: bool = False  # Whether rubric evaluation was executed
    rubric_evaluation_strategy: str | None = None  # Strategy used: "batch" or "sequential"

    # Split trait scores by type (replaces old verify_rubric dict)
    llm_trait_scores: dict[str, int | bool] | None = None  # LLM-evaluated traits (1-5 scale or binary)
    regex_trait_scores: dict[str, bool] | None = None  # Regex-based traits (boolean)
    callable_trait_scores: dict[str, bool | int] | None = None  # Callable-based traits (boolean or score)
    metric_trait_scores: dict[str, dict[str, float]] | None = None  # Metric traits with nested metrics dict

    # Metric trait evaluation metadata (confusion-matrix analysis)
    metric_trait_confusion_lists: dict[str, dict[str, list[str]]] | None = None  # Confusion lists per metric trait
    # Structure: {"trait_name": {"tp": ["excerpt1", ...], "tn": [...], "fp": [...], "fn": [...]}}
    # Each trait has four lists (TP/TN/FP/FN) containing extracted excerpts from the answer

    def get_all_trait_scores(self) -> dict[str, int | bool | dict[str, float]]:
        """
        Get all trait scores across all trait types in a flat dictionary.

        Returns:
            dict: All trait scores, e.g.:
                {
                    "clarity": 4,
                    "mentions_regulatory_elements": True,
                    "feature_identification": {"precision": 1.0, "recall": 1.0, "f1": 1.0}
                }
        """
        scores: dict[str, int | bool | dict[str, float]] = {}

        if self.llm_trait_scores:
            scores.update(self.llm_trait_scores)
        if self.regex_trait_scores:
            scores.update(self.regex_trait_scores)
        if self.callable_trait_scores:
            scores.update(self.callable_trait_scores)
        if self.metric_trait_scores:
            scores.update(self.metric_trait_scores)

        return scores

    def get_trait_by_name(self, name: str) -> tuple[Any, str] | None:
        """
        Look up a trait by name across all trait types.

        Args:
            name: The trait name to search for

        Returns:
            tuple: (value, trait_type) where trait_type is "llm", "regex", "callable", or "metric"
            None: If the trait is not found
        """
        if self.llm_trait_scores and name in self.llm_trait_scores:
            return (self.llm_trait_scores[name], "llm")
        if self.regex_trait_scores and name in self.regex_trait_scores:
            return (self.regex_trait_scores[name], "regex")
        if self.callable_trait_scores and name in self.callable_trait_scores:
            return (self.callable_trait_scores[name], "callable")
        if self.metric_trait_scores and name in self.metric_trait_scores:
            return (self.metric_trait_scores[name], "metric")
        return None


class VerificationResultDeepJudgment(BaseModel):
    """Deep-judgment metadata (multi-stage parsing with excerpts and reasoning)."""

    deep_judgment_enabled: bool = False  # Whether deep-judgment was configured
    deep_judgment_performed: bool = False  # Whether deep-judgment was successfully executed
    extracted_excerpts: dict[str, list[dict[str, Any]]] | None = None  # Extracted excerpts per attribute
    # Structure: {"attribute_name": [{"text": str, "confidence": "low|medium|high", "similarity_score": float,
    #   "search_results"?: str, "hallucination_risk"?: "none|low|medium|high", "hallucination_justification"?: str}]}
    # Empty list [] indicates no excerpts found for that attribute (e.g., refusals, no corroborating evidence)
    # When search is enabled, each excerpt includes:
    #   - "search_results": External validation text from search tool
    #   - "hallucination_risk": Per-excerpt risk assessment (none/low/medium/high)
    #   - "hallucination_justification": Explanation for the risk level
    attribute_reasoning: dict[str, str] | None = None  # Reasoning traces per attribute
    # Structure: {"attribute_name": "reasoning text"}
    # Reasoning can exist even when excerpts are empty (explains why no excerpts found)
    deep_judgment_stages_completed: list[str] | None = None  # Stages completed: ["excerpts", "reasoning", "parameters"]
    deep_judgment_model_calls: int = 0  # Number of LLM invocations for deep-judgment
    deep_judgment_excerpt_retry_count: int = 0  # Number of retries for excerpt validation
    attributes_without_excerpts: list[str] | None = None  # Attributes with no corroborating excerpts

    # Search-enhanced deep-judgment metadata
    deep_judgment_search_enabled: bool = False  # Whether search enhancement was enabled for this verification
    hallucination_risk_assessment: dict[str, str] | None = None  # Hallucination risk per attribute
    # Structure: {"attribute_name": "none" | "low" | "medium" | "high"}
    # Scale: "none" = lowest risk (strong external evidence), "high" = highest risk (weak/no evidence)
    # Only populated when deep_judgment_search_enabled=True


class VerificationResultDeepJudgmentRubric(BaseModel):
    """Deep-judgment metadata for rubric trait evaluation (multi-stage with excerpts and reasoning)."""

    deep_judgment_rubric_performed: bool = False  # Whether deep-judgment rubric evaluation was executed

    # Per-trait excerpts (only for traits with deep_judgment_excerpt_enabled=True)
    extracted_rubric_excerpts: dict[str, list[dict[str, Any]]] | None = None
    # Structure: {"trait_name": [{"text": str, "confidence": "low|medium|high", "similarity_score": float,
    #   "search_results"?: list, "hallucination_risk"?: "none|low|medium|high",
    #   "hallucination_justification"?: str}]}
    # Empty list [] indicates no excerpts found for that trait after all retry attempts

    # Per-trait reasoning (ALL deep-judgment-enabled traits, with or without excerpts)
    rubric_trait_reasoning: dict[str, str] | None = None
    # Structure: {"trait_name": "reasoning text explaining how the score was determined"}
    # Reasoning exists for all deep-judgment traits regardless of excerpt extraction

    # Per-trait scores from deep-judgment evaluation
    deep_judgment_rubric_scores: dict[str, int | bool] | None = None
    # Structure: {"trait_name": score/boolean}
    # Scores for traits evaluated with deep-judgment

    # Standard evaluation scores (for non-deep-judgment traits in the same rubric)
    standard_rubric_scores: dict[str, int | bool] | None = None
    # Structure: {"trait_name": score/boolean}
    # Scores for traits evaluated without deep-judgment

    # Per-trait metadata for detailed tracking
    trait_metadata: dict[str, dict[str, Any]] | None = None
    # Structure: {"trait_name": {
    #   "stages_completed": ["excerpt_extraction", "reasoning_generation", "score_extraction"],
    #   "model_calls": 3,
    #   "had_excerpts": True,
    #   "excerpt_retry_count": 1,
    #   "excerpt_validation_failed": False
    # }}

    # Auto-fail tracking (only traits with excerpts enabled that exhausted retries)
    traits_without_valid_excerpts: list[str] | None = None
    # List of trait names that failed to extract valid excerpts after all retry attempts
    # These traits trigger auto-fail unless abstention was detected

    # Search-enhanced metadata (only traits with deep_judgment_search_enabled=True)
    rubric_hallucination_risk_assessment: dict[str, dict[str, Any]] | None = None
    # Structure: {"trait_name": {
    #   "overall_risk": "none|low|medium|high",
    #   "per_excerpt_risks": ["low", "medium", ...]
    # }}
    # Hallucination risk assessment per trait (overall risk = MAX of per-excerpt risks)

    # Aggregated statistics
    total_deep_judgment_model_calls: int = 0  # Total LLM calls across all deep-judgment traits
    total_traits_evaluated: int = 0  # Number of traits evaluated with deep-judgment
    total_excerpt_retries: int = 0  # Total retry attempts across all traits
