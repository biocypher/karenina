"""Template verification stage.

Verifies parsed response matches template (field + regex validation).
"""

import logging

from ..stage import BaseVerificationStage, VerificationContext

# Set up logger
logger = logging.getLogger(__name__)


class VerifyTemplateStage(BaseVerificationStage):
    """
    Verifies parsed response against template.

    This stage:
    1. Runs field verification (calls parsed_answer.verify())
    2. Runs regex validation on raw LLM response
    3. Extracts regex matches for display
    4. Combines field and regex results

    Requires:
        - "parsed_answer": Parsed Pydantic object
        - "raw_llm_response": Raw LLM response text (for regex validation)

    Produces:
        - "field_verification_result": Boolean result of field verification
        - "regex_verification_results": Dict with results, details, success
        - "regex_extraction_results": Dict of actual regex matches
        - "verification_result": Combined field + regex result (bool)

    Note:
        This stage performs the core verification but does NOT apply
        overrides (embedding check, abstention, deep-judgment auto-fail).
        Those are handled by subsequent stages.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "VerifyTemplate"

    @property
    def requires(self) -> list[str]:
        """Artifacts required by this stage."""
        return ["parsed_answer", "raw_llm_response"]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            "field_verification_result",
            "regex_verification_results",
            "regex_extraction_results",
            "verification_result",
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """Run if we have parsed answer, no errors, and no recursion limit."""
        # Skip verification if recursion limit was reached (response is truncated/unreliable)
        if context.get_artifact("recursion_limit_reached", False):
            return False
        return context.has_artifact("parsed_answer") and context.has_artifact("raw_llm_response") and not context.error

    def execute(self, context: VerificationContext) -> None:
        """
        Verify parsed response against template.

        Args:
            context: Verification context

        Side Effects:
            - Sets context.artifacts["field_verification_result"]
            - Sets context.artifacts["regex_verification_results"]
            - Sets context.artifacts["regex_extraction_results"]
            - Sets context.artifacts["verification_result"]
            - Sets context.result_field for all verification metadata
            - Sets context.error if verification logic fails
        """
        parsed_answer = context.get_artifact("parsed_answer")
        raw_llm_response = context.get_artifact("raw_llm_response")

        try:
            # Step 1: Field verification
            field_verification_result = parsed_answer.verify()

            # Step 2: Regex verification on raw trace
            regex_verification_results = parsed_answer.verify_regex(raw_llm_response)

            # Step 3: Extract regex results for display (what the regex actually matched)
            regex_extraction_results = {}
            if regex_verification_results["details"]:
                for field_name, details in regex_verification_results["details"].items():
                    regex_extraction_results[field_name] = details.get("matches_found", [])

            # Step 4: Combine field and regex verification results
            verification_result = field_verification_result and regex_verification_results["success"]

            # Store results
            context.set_artifact("field_verification_result", field_verification_result)
            context.set_artifact("regex_verification_results", regex_verification_results)
            context.set_artifact("regex_extraction_results", regex_extraction_results)
            context.set_artifact("verification_result", verification_result)

            # Store in result builder
            context.set_result_field("verify_result", verification_result)
            context.set_result_field("regex_validations_performed", bool(regex_verification_results["results"]))
            context.set_result_field("regex_validation_results", regex_verification_results["results"])
            context.set_result_field("regex_validation_details", regex_verification_results["details"])
            context.set_result_field("regex_overall_success", regex_verification_results["success"])
            context.set_result_field("regex_extraction_results", regex_extraction_results)

        except Exception as e:
            error_msg = f"Verification failed: {e}"
            logger.error(error_msg)
            context.mark_error(error_msg)
