"""Verification result model with nested component structure.

All backward-compatibility property accessors have been removed.
Callers must now access fields through their nested objects:
- result.metadata.question_id (not result.question_id)
- result.template.raw_llm_response (not result.raw_llm_response)
- result.rubric.verify_rubric (not result.verify_rubric)
- result.deep_judgment.extracted_excerpts (not result.extracted_excerpts)
"""

from pydantic import BaseModel

from .result_components import (
    VerificationResultDeepJudgment,
    VerificationResultDeepJudgmentRubric,
    VerificationResultMetadata,
    VerificationResultRubric,
    VerificationResultTemplate,
)


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
