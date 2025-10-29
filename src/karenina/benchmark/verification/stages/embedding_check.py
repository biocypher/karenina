"""Embedding check stage.

Semantic similarity fallback for field verification failures.
"""

import logging

from ..embedding_utils import perform_embedding_check
from ..runner import _split_parsed_response
from ..stage import BaseVerificationStage, VerificationContext

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
        - "verification_result": Combined verification result
        - "regex_verification_results": Regex results (for recalculation)

    Produces:
        - "embedding_check_performed": Whether check was attempted (bool)
        - "embedding_similarity_score": Similarity score 0-1 (float or None)
        - "embedding_model_used": Name of embedding model (str or None)
        - "embedding_override_applied": Whether override was applied (bool)

    Side Effects:
        - May update "verification_result" and "field_verification_result" if override applied

    Note:
        Embedding check does NOT override regex validation failures, only field failures.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "EmbeddingCheck"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        return [
            "parsed_answer",
            "field_verification_result",
            "verification_result",
            "regex_verification_results",
        ]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            "embedding_check_performed",
            "embedding_similarity_score",
            "embedding_model_used",
            "embedding_override_applied",
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run only if field verification failed and no errors.

        Embedding check is only useful when field verification failed,
        as it provides a semantic fallback.
        """
        if context.error:
            return False

        field_verification_result = context.get_artifact("field_verification_result")
        return field_verification_result is False

    def execute(self, context: VerificationContext) -> None:
        """
        Apply embedding check fallback.

        Args:
            context: Verification context

        Side Effects:
            - Sets embedding metadata artifacts
            - May override verification_result if semantically equivalent
            - Sets result fields for embedding metadata
        """
        parsed_answer = context.get_artifact("parsed_answer")
        field_verification_result = context.get_artifact("field_verification_result")
        regex_verification_results = context.get_artifact("regex_verification_results")

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
            context.set_artifact("field_verification_result", field_verification_result)
            context.set_artifact("verification_result", verification_result)
            context.set_result_field("verify_result", verification_result)

            logger.info(
                f"Embedding check override applied for question {context.question_id} "
                f"(similarity: {similarity_score:.3f})"
            )

        # Store embedding metadata
        context.set_artifact("embedding_check_performed", embedding_check_performed)
        context.set_artifact("embedding_similarity_score", embedding_similarity_score)
        context.set_artifact("embedding_model_used", embedding_model_used)
        context.set_artifact("embedding_override_applied", embedding_override_applied)

        # Store in result builder
        context.set_result_field("embedding_check_performed", embedding_check_performed)
        context.set_result_field("embedding_similarity_score", embedding_similarity_score)
        context.set_result_field("embedding_override_applied", embedding_override_applied)
        context.set_result_field("embedding_model_used", embedding_model_used)
