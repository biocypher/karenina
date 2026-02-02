"""Verification result model with backward compatibility."""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .result_components import (
    VerificationResultDeepJudgment,
    VerificationResultDeepJudgmentRubric,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)

if TYPE_CHECKING:
    pass


class VerificationResult(BaseModel):
    """Result of verifying a single question."""

    metadata: VerificationResultMetadata
    template: VerificationResultTemplate | None = None
    rubric: VerificationResultRubric | None = None
    deep_judgment: VerificationResultDeepJudgment | None = None
    deep_judgment_rubric: VerificationResultDeepJudgmentRubric | None = None

    # Shared trace filtering fields (for MCP agent responses)
    # These are at the root level because both template and rubric evaluation use the same input
    evaluation_input: str | None = None  # Input passed to evaluation (full trace or final AI message)
    used_full_trace: bool = True  # Whether full trace was used (True) or only final AI message (False)
    trace_extraction_error: str | None = None  # Error if final AI message extraction failed

    # Backward compatibility properties for common field access
    @property
    def question_id(self) -> str:
        """Backward compatibility accessor for question_id."""
        return self.metadata.question_id

    @property
    def completed_without_errors(self) -> bool:
        """Backward compatibility accessor for completed_without_errors."""
        return self.metadata.completed_without_errors

    @property
    def error(self) -> str | None:
        """Backward compatibility accessor for error."""
        return self.metadata.error

    @property
    def question_text(self) -> str:
        """Backward compatibility accessor for question_text."""
        return self.metadata.question_text

    @property
    def keywords(self) -> list[str] | None:
        """Backward compatibility accessor for keywords."""
        return self.metadata.keywords

    @property
    def answering_model(self) -> str:
        """Backward compatibility accessor for answering_model display string."""
        return self.metadata.answering_model

    @property
    def parsing_model(self) -> str:
        """Backward compatibility accessor for parsing_model display string."""
        return self.metadata.parsing_model

    @property
    def run_name(self) -> str | None:
        """Backward compatibility accessor for run_name."""
        return self.metadata.run_name

    @property
    def timestamp(self) -> str:
        """Backward compatibility accessor for timestamp."""
        return self.metadata.timestamp

    # Template field backward compatibility
    @property
    def raw_llm_response(self) -> str | None:
        """Backward compatibility accessor for raw_llm_response."""
        return self.template.raw_llm_response if self.template else None

    @property
    def usage_metadata(self) -> dict[str, Any] | None:
        """Backward compatibility accessor for usage_metadata."""
        return self.template.usage_metadata if self.template else None

    @property
    def agent_metrics(self) -> dict[str, Any] | None:
        """Backward compatibility accessor for agent_metrics."""
        return self.template.agent_metrics if self.template else None

    @property
    def recursion_limit_reached(self) -> bool | None:
        """Backward compatibility accessor for recursion_limit_reached."""
        return self.template.recursion_limit_reached if self.template else None

    @property
    def answering_mcp_servers(self) -> list[str] | None:
        """Backward compatibility accessor for answering_mcp_servers."""
        return self.template.answering_mcp_servers if self.template else None

    @property
    def abstention_detected(self) -> bool | None:
        """Backward compatibility accessor for abstention_detected."""
        return self.template.abstention_detected if self.template else None

    @property
    def abstention_override_applied(self) -> bool:
        """Backward compatibility accessor for abstention_override_applied."""
        return self.template.abstention_override_applied if self.template else False

    @property
    def abstention_check_performed(self) -> bool:
        """Backward compatibility accessor for abstention_check_performed."""
        return self.template.abstention_check_performed if self.template else False

    @property
    def parsed_gt_response(self) -> dict[str, Any] | None:
        """Backward compatibility accessor for parsed_gt_response."""
        return self.template.parsed_gt_response if self.template else None

    @property
    def parsed_llm_response(self) -> dict[str, Any] | None:
        """Backward compatibility accessor for parsed_llm_response."""
        return self.template.parsed_llm_response if self.template else None

    @property
    def verify_result(self) -> bool | None:
        """Backward compatibility accessor for verify_result."""
        return self.template.verify_result if self.template else None

    @property
    def verify_granular_result(self) -> Any | None:
        """Backward compatibility accessor for verify_granular_result."""
        return self.template.verify_granular_result if self.template else None

    @property
    def embedding_check_performed(self) -> bool:
        """Backward compatibility accessor for embedding_check_performed."""
        return self.template.embedding_check_performed if self.template else False

    @property
    def embedding_similarity_score(self) -> float | None:
        """Backward compatibility accessor for embedding_similarity_score."""
        return self.template.embedding_similarity_score if self.template else None

    @property
    def embedding_override_applied(self) -> bool:
        """Backward compatibility accessor for embedding_override_applied."""
        return self.template.embedding_override_applied if self.template else False

    @property
    def embedding_model_used(self) -> str | None:
        """Backward compatibility accessor for embedding_model_used."""
        return self.template.embedding_model_used if self.template else None

    @property
    def template_verification_performed(self) -> bool:
        """Backward compatibility accessor for template_verification_performed."""
        return self.template.template_verification_performed if self.template else False

    @property
    def abstention_reasoning(self) -> str | None:
        """Backward compatibility accessor for abstention_reasoning."""
        return self.template.abstention_reasoning if self.template else None

    @property
    def sufficiency_detected(self) -> bool | None:
        """Backward compatibility accessor for sufficiency_detected."""
        return self.template.sufficiency_detected if self.template else None

    @property
    def sufficiency_override_applied(self) -> bool:
        """Backward compatibility accessor for sufficiency_override_applied."""
        return self.template.sufficiency_override_applied if self.template else False

    @property
    def sufficiency_check_performed(self) -> bool:
        """Backward compatibility accessor for sufficiency_check_performed."""
        return self.template.sufficiency_check_performed if self.template else False

    @property
    def sufficiency_reasoning(self) -> str | None:
        """Backward compatibility accessor for sufficiency_reasoning."""
        return self.template.sufficiency_reasoning if self.template else None

    @property
    def regex_validations_performed(self) -> bool:
        """Backward compatibility accessor for regex_validations_performed."""
        return self.template.regex_validations_performed if self.template else False

    @property
    def regex_validation_results(self) -> dict[str, bool] | None:
        """Backward compatibility accessor for regex_validation_results."""
        return self.template.regex_validation_results if self.template else None

    @property
    def regex_validation_details(self) -> dict[str, dict[str, Any]] | None:
        """Backward compatibility accessor for regex_validation_details."""
        return self.template.regex_validation_details if self.template else None

    @property
    def regex_overall_success(self) -> bool | None:
        """Backward compatibility accessor for regex_overall_success."""
        return self.template.regex_overall_success if self.template else None

    @property
    def regex_extraction_results(self) -> dict[str, Any] | None:
        """Backward compatibility accessor for regex_extraction_results."""
        return self.template.regex_extraction_results if self.template else None

    # Deep judgment backward compatibility
    @property
    def deep_judgment_performed(self) -> bool:
        """Backward compatibility accessor for deep_judgment_performed."""
        return self.deep_judgment.deep_judgment_performed if self.deep_judgment else False

    @property
    def deep_judgment_enabled(self) -> bool:
        """Backward compatibility accessor for deep_judgment_enabled."""
        return self.deep_judgment.deep_judgment_enabled if self.deep_judgment else False

    @property
    def extracted_excerpts(self) -> dict[str, list[dict[str, Any]]] | None:
        """Backward compatibility accessor for extracted_excerpts."""
        return self.deep_judgment.extracted_excerpts if self.deep_judgment else None

    @property
    def attribute_reasoning(self) -> dict[str, str] | None:
        """Backward compatibility accessor for attribute_reasoning."""
        return self.deep_judgment.attribute_reasoning if self.deep_judgment else None

    @property
    def deep_judgment_stages_completed(self) -> list[str] | None:
        """Backward compatibility accessor for deep_judgment_stages_completed."""
        return self.deep_judgment.deep_judgment_stages_completed if self.deep_judgment else None

    @property
    def deep_judgment_model_calls(self) -> int:
        """Backward compatibility accessor for deep_judgment_model_calls."""
        return self.deep_judgment.deep_judgment_model_calls if self.deep_judgment else 0

    @property
    def deep_judgment_excerpt_retry_count(self) -> int:
        """Backward compatibility accessor for deep_judgment_excerpt_retry_count."""
        return self.deep_judgment.deep_judgment_excerpt_retry_count if self.deep_judgment else 0

    @property
    def attributes_without_excerpts(self) -> list[str] | None:
        """Backward compatibility accessor for attributes_without_excerpts."""
        return self.deep_judgment.attributes_without_excerpts if self.deep_judgment else None

    @property
    def deep_judgment_search_enabled(self) -> bool:
        """Backward compatibility accessor for deep_judgment_search_enabled."""
        return self.deep_judgment.deep_judgment_search_enabled if self.deep_judgment else False

    @property
    def hallucination_risk_assessment(self) -> dict[str, str] | None:
        """Backward compatibility accessor for hallucination_risk_assessment."""
        return self.deep_judgment.hallucination_risk_assessment if self.deep_judgment else None

    # Rubric field backward compatibility
    @property
    def rubric_evaluation_performed(self) -> bool:
        """Backward compatibility accessor for rubric_evaluation_performed."""
        return self.rubric.rubric_evaluation_performed if self.rubric else False

    @property
    def verify_rubric(self) -> dict[str, Any] | None:
        """Backward compatibility accessor for verify_rubric (combines all trait types)."""
        if not self.rubric:
            return None

        scores = self.rubric.get_all_trait_scores()
        return scores if scores else None

    @property
    def metric_trait_metrics(self) -> dict[str, dict[str, float]] | None:
        """Backward compatibility accessor for metric_trait_metrics."""
        return self.rubric.metric_trait_scores if self.rubric else None

    @property
    def metric_trait_confusion_lists(self) -> dict[str, dict[str, list[str]]] | None:
        """Backward compatibility accessor for metric_trait_confusion_lists."""
        return self.rubric.metric_trait_confusion_lists if self.rubric else None
