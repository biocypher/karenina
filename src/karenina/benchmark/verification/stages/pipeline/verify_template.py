"""Template verification stage.

Verifies parsed response matches template (field + regex validation).
"""

import logging

from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext

# Set up logger
logger = logging.getLogger(__name__)


class VerifyTemplateStage(BaseVerificationStage):
    """
    Verifies parsed response against template.

    This stage:
    1. Retrieves the TemplateEvaluator from context (or creates one)
    2. Runs field verification via evaluator
    3. Runs regex verification via evaluator
    4. Combines field and regex results

    Requires:
        - "parsed_answer": Parsed Pydantic object
        - "raw_llm_response": Raw LLM response text (for regex validation)
        - "template_evaluator": TemplateEvaluator instance (created by ParseTemplateStage)

    Produces:
        - "field_verification_result": Boolean result of field verification
        - "regex_verification_results": Dict with results, details, success
        - "regex_extraction_results": Dict of actual regex matches
        - "verify_result": Combined field + regex result (bool)
        - "verify_granular_result": Float (0.0-1.0) for multi-attribute templates only

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
        """Artifacts required by this stage.

        Note: TEMPLATE_EVALUATOR is not listed here because regex-only
        templates set it to None (no LLM parsing needed). The evaluator
        is checked at runtime instead.
        """
        return [
            ArtifactKeys.PARSED_ANSWER,
            ArtifactKeys.RAW_LLM_RESPONSE,
        ]

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            ArtifactKeys.FIELD_VERIFICATION_RESULT,
            ArtifactKeys.REGEX_VERIFICATION_RESULTS,
            ArtifactKeys.REGEX_EXTRACTION_RESULTS,
            ArtifactKeys.VERIFY_RESULT,
            ArtifactKeys.VERIFY_GRANULAR_RESULT,  # Only for multi-attribute templates
        ]

    def should_run(self, context: VerificationContext) -> bool:
        """
        Run if we have parsed answer, no errors, no recursion limit, and no abstention.

        Inherits error-checking from BaseVerificationStage.
        """
        if not super().should_run(context):
            return False
        # Skip verification if recursion limit was reached (response is truncated/unreliable)
        if context.get_artifact(ArtifactKeys.RECURSION_LIMIT_REACHED, False):
            return False
        # Skip verification if response was truncated by streaming timeout
        if context.get_artifact(ArtifactKeys.RESPONSE_TIMEOUT_PARTIAL, False):
            return False
        # Skip verification if abstention was detected (model refused to answer)
        if context.get_artifact(ArtifactKeys.ABSTENTION_DETECTED, False):
            return False
        return context.has_artifact(ArtifactKeys.PARSED_ANSWER) and context.has_artifact(ArtifactKeys.RAW_LLM_RESPONSE)

    def execute(self, context: VerificationContext) -> None:
        """
        Verify parsed response against template.

        Args:
            context: Verification context

        Side Effects:
            - Sets context.artifacts["field_verification_result"]
            - Sets context.artifacts["regex_verification_results"]
            - Sets context.artifacts["regex_extraction_results"]
            - Sets context.artifacts["verify_result"]
            - Sets context.result_field for all verification metadata
            - Sets context.error if verification logic fails
        """
        parsed_answer = context.get_artifact(ArtifactKeys.PARSED_ANSWER)
        raw_llm_response = context.get_artifact(ArtifactKeys.RAW_LLM_RESPONSE)

        # Inject raw trace for VerifiedField templates that use TracePrimitive fields.
        # _raw_trace is consumed by _compute_field_results() during verify().
        if not hasattr(parsed_answer, "_raw_trace") or parsed_answer._raw_trace is None:
            parsed_answer._raw_trace = raw_llm_response

        # Inject scenario context for ConditionalGroundTruth resolution.
        # _scenario_context is consumed by _compute_field_results() when
        # a field's ground_truth carries the __conditional__ marker.
        scenario_node_results = context.get_artifact(ArtifactKeys.SCENARIO_NODE_RESULTS)
        if scenario_node_results is not None:
            parsed_answer._scenario_context = {"node_results": scenario_node_results}

        # Get evaluator from context (created by ParseTemplateStage, or None for regex-only)
        evaluator = context.get_artifact(ArtifactKeys.TEMPLATE_EVALUATOR)

        try:
            if evaluator is not None:
                # Standard path: delegate to evaluator
                field_result = evaluator.verify_fields(parsed_answer)
                regex_result = evaluator.verify_regex(parsed_answer, raw_llm_response)

                field_verification_result = field_result.success
                regex_verification_results = {
                    "success": regex_result.success,
                    "results": regex_result.results,
                    "details": regex_result.details,
                }
                regex_extraction_results = regex_result.extraction_results

                if field_result.error:
                    logger.warning("Field verification error: %s", field_result.error)
                    context.set_result_field(ArtifactKeys.FIELD_VERIFICATION_ERROR, field_result.error)
                if regex_result.error:
                    logger.warning("Regex verification error: %s", regex_result.error)
            else:
                # Regex-only path: no evaluator, call answer methods directly
                logger.info("Regex-only verification — running verify_regex directly")

                # Field verification: call verify() if defined, else True (no fields)
                if hasattr(parsed_answer, "verify") and callable(parsed_answer.verify):
                    try:
                        field_verification_result = parsed_answer.verify()
                    except Exception as e:
                        logger.warning("Field verification error (regex-only): %s", e)
                        field_verification_result = False
                else:
                    field_verification_result = True

                # Regex verification: call verify_regex on the raw response
                regex_raw = parsed_answer.verify_regex(raw_llm_response)
                regex_verification_results = {
                    "success": regex_raw["success"],
                    "results": regex_raw["results"],
                    "details": regex_raw["details"],
                }
                regex_extraction_results = {}
                for field_name, details in regex_raw.get("details", {}).items():
                    regex_extraction_results[field_name] = details.get("matches_found", [])

            # Combine field and regex verification results
            verification_result = field_verification_result and regex_verification_results["success"]

            # Store results
            context.set_artifact(ArtifactKeys.FIELD_VERIFICATION_RESULT, field_verification_result)
            context.set_artifact(ArtifactKeys.REGEX_VERIFICATION_RESULTS, regex_verification_results)
            context.set_artifact(ArtifactKeys.REGEX_EXTRACTION_RESULTS, regex_extraction_results)
            context.set_artifact(ArtifactKeys.VERIFY_RESULT, verification_result)

            # Store in result builder
            context.set_result_field(ArtifactKeys.VERIFY_RESULT, verification_result)
            context.set_result_field(
                ArtifactKeys.REGEX_VALIDATIONS_PERFORMED, bool(regex_verification_results["results"])
            )
            context.set_result_field(ArtifactKeys.REGEX_VALIDATION_RESULTS, regex_verification_results["results"])
            context.set_result_field(ArtifactKeys.REGEX_VALIDATION_DETAILS, regex_verification_results["details"])
            context.set_result_field(ArtifactKeys.REGEX_OVERALL_SUCCESS, regex_verification_results["success"])
            context.set_result_field(ArtifactKeys.REGEX_EXTRACTION_RESULTS, regex_extraction_results)

            # Granular verification for multi-attribute templates
            # Skip when field verification raised an error (issue 148): avoids
            # contradictory signals (verify_result=False, verify_granular_result=1.0)
            if field_result.error if evaluator is not None else False:
                logger.debug("Skipping verify_granular() because field verification error is set")
            elif hasattr(parsed_answer, "verify_granular") and callable(parsed_answer.verify_granular):
                try:
                    granular_result = parsed_answer.verify_granular()
                    context.set_result_field(ArtifactKeys.VERIFY_GRANULAR_RESULT, granular_result)
                    logger.debug("Granular verification result: %s", granular_result)
                except Exception as e:
                    logger.warning("Granular verification failed: %s", e)

        except Exception as e:
            error_msg = f"Verification failed: {e}"
            logger.error(error_msg)
            context.mark_error(error_msg)
