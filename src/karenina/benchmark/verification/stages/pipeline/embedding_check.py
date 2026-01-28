"""Embedding check stage.

Semantic similarity fallback for field verification failures.
"""

import logging

from ...utils.embedding_check import perform_embedding_check
from ...utils.llm_invocation import _split_parsed_response
from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext

# Set up logger
logger = logging.getLogger(__name__)


class EmbeddingCheckStage(BaseVerificationStage):
    """
    Applies semantic similarity check as fallback for field verification failures.

    This stage:
    1. Only runs if field verification failed
    2. Extracts ground truth and LLM response from parsed answer
    3. Computes embedding similarity
    4. Asks parsing model for semantic equivalence judgment
    5. If semantically equivalent, overrides field verification result
    6. Recalculates overall verification result if override applied

    Requires:
        - "parsed_answer": Parsed Pydantic object
        - "field_verification_result": Boolean result of field verification
        - "regex_verification_results": Regex results (for recalculation if override applied)

    Produces:
        - "embedding_check_performed": Whether check was attempted (bool)
        - "embedding_similarity_score": Similarity score 0-1 (float or None)
        - "embedding_model_used": Name of embedding model (str or None)
        - "embedding_override_applied": Whether override was applied (bool)

    Side Effects:
        - May update "verify_result" and "field_verification_result" if override applied

    Note:
        Embedding check does NOT override regex validation failures, only field failures.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "EmbeddingCheck"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage.

        Note: verify_result is NOT required as input - it's only written as output
        after being recalculated from field_verification_result and regex_verification_results.
        """
        return [
            ArtifactKeys.PARSED_ANSWER,
            ArtifactKeys.FIELD_VERIFICATION_RESULT,
            ArtifactKeys.REGEX_VERIFICATION_RESULTS,
        ]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            ArtifactKeys.EMBEDDING_CHECK_PERFORMED,
            ArtifactKeys.EMBEDDING_SIMILARITY_SCORE,
            ArtifactKeys.EMBEDDING_MODEL_USED,
            ArtifactKeys.EMBEDDING_OVERRIDE_APPLIED,
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run only if field verification failed.

        Embedding check is only useful when field verification failed,
        as it provides a semantic fallback.
        Inherits error-checking from BaseVerificationStage.
        """
        if not super().should_run(context):
            return False

        field_verification_result = context.get_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT)
        return field_verification_result is False

    def execute(self, context: VerificationContext) -> None:
        """
        Apply embedding check fallback.

        Args:
            context: Verification context

        Side Effects:
            - Sets embedding metadata artifacts
            - May override verify_result if semantically equivalent
            - Sets result fields for embedding metadata
        """
        parsed_answer = context.get_artifact(ArtifactKeys.PARSED_ANSWER)
        field_verification_result = context.get_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT)
        regex_verification_results = context.get_artifact(ArtifactKeys.REGEX_VERIFICATION_RESULTS)

        # Extract ground truth and LLM response for embedding check
        parsed_gt_response, parsed_llm_response = _split_parsed_response(parsed_answer)

        # Perform embedding check
        (should_override, similarity_score, model_name, check_performed) = perform_embedding_check(
            parsed_gt_response, parsed_llm_response, context.parsing_model, context.question_text
        )

        embedding_check_performed = check_performed
        embedding_similarity_score = similarity_score
        embedding_model_used = model_name
        embedding_override_applied = False

        # Apply override if semantic equivalence detected
        if should_override:
            # Override field verification
            field_verification_result = True
            # Recalculate overall result (field AND regex)
            verification_result = True and regex_verification_results["success"]
            embedding_override_applied = True

            # Update stored results
            context.set_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT, field_verification_result)
            context.set_artifact(ArtifactKeys.VERIFY_RESULT, verification_result)
            context.set_result_field(ArtifactKeys.VERIFY_RESULT, verification_result)

            logger.warning(
                f"Embedding check override applied for question {context.question_id} "
                f"(similarity: {similarity_score:.3f})"
            )

        # Store embedding metadata (both artifact and result field)
        self.set_artifact_and_result(context, ArtifactKeys.EMBEDDING_CHECK_PERFORMED, embedding_check_performed)
        self.set_artifact_and_result(context, ArtifactKeys.EMBEDDING_SIMILARITY_SCORE, embedding_similarity_score)
        self.set_artifact_and_result(context, ArtifactKeys.EMBEDDING_MODEL_USED, embedding_model_used)
        self.set_artifact_and_result(context, ArtifactKeys.EMBEDDING_OVERRIDE_APPLIED, embedding_override_applied)
