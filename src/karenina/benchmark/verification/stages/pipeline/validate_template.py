"""Template validation stage.

Validates the Pydantic template syntax and injects question ID.
"""

import logging

from karenina.benchmark.authoring.answers.generator import inject_question_id_into_answer_class
from karenina.benchmark.verification.utils.template_validation import validate_answer_template
from karenina.schemas.entities.answer import BaseAnswer

from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext

logger = logging.getLogger(__name__)


class ValidateTemplateStage(BaseVerificationStage):
    """
    Validates template code and prepares Answer class.

    This stage:
    1. Validates the Pydantic template syntax
    2. Checks for required fields and correct structure
    3. Injects the question ID into the Answer class
    4. Detects the template mode (classic, verified, or mixed)
    5. Stores the validated Answer class for use by later stages

    If validation fails, marks the context as failed and skips remaining stages.

    Produces:
        - "RawAnswer": The validated Pydantic class (before question ID injection)
        - "Answer": The Answer class with question ID injected
        - "template_validation_error": Error message if validation failed
        - "template_mode": One of "classic", "verified", or "mixed"

    Error Handling:
        If template validation fails, marks context.error and sets
        completed_without_errors=False to skip remaining stages.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        return "ValidateTemplate"

    @property
    def produces(self) -> list[str]:
        """Artifacts produced by this stage."""
        return [
            ArtifactKeys.RAW_ANSWER,
            ArtifactKeys.ANSWER,
            ArtifactKeys.TEMPLATE_VALIDATION_ERROR,
            ArtifactKeys.TEMPLATE_MODE,
        ]

    def should_run(self, context: VerificationContext) -> bool:  # noqa: ARG002
        """Always run - this is the first stage."""
        return True

    def execute(self, context: VerificationContext) -> None:
        """
        Validate template and inject question ID.

        Args:
            context: Verification context

        Side Effects:
            - Sets context.artifacts["RawAnswer"] if validation succeeds
            - Sets context.artifacts["Answer"] if validation succeeds
            - Sets context.artifacts["template_validation_error"] if validation fails
            - Sets context.artifacts["template_mode"] if validation succeeds
            - Sets context.error if validation fails
            - Sets context.completed_without_errors=False if validation fails
        """
        # Validate the template
        is_valid, error_msg, RawAnswer = validate_answer_template(context.template_code)

        if not is_valid or RawAnswer is None:
            # Template validation failed: mark error and stop pipeline
            error_message = f"Template validation failed: {error_msg}"
            context.mark_error(error_message)
            context.set_artifact(ArtifactKeys.TEMPLATE_VALIDATION_ERROR, error_message)
            return

        # Store the validated RawAnswer class
        context.set_artifact(ArtifactKeys.RAW_ANSWER, RawAnswer)

        # Inject question ID into the Answer class
        Answer = inject_question_id_into_answer_class(RawAnswer, context.question_id)
        context.set_artifact(ArtifactKeys.ANSWER, Answer)

        # Mark validation as successful
        context.set_artifact(ArtifactKeys.TEMPLATE_VALIDATION_ERROR, None)

        # Detect template mode based on VerifiedField presence
        template_mode = _detect_template_mode(Answer)
        context.set_artifact(ArtifactKeys.TEMPLATE_MODE, template_mode)
        logger.debug("Template mode detected: %s", template_mode)


def _detect_template_mode(answer_class: type[BaseAnswer]) -> str:
    """Detect whether a template uses classic, verified, or mixed fields.

    Args:
        answer_class: The validated Answer class.

    Returns:
        "verified" if all user-defined fields use VerifiedField,
        "mixed" if some do and some do not,
        "classic" if none do.
    """
    verified_fields = answer_class._get_verified_fields()
    if not verified_fields:
        return "classic"

    # Determine which fields are user-defined (exclude inherited BaseAnswer fields)
    inherited = set(BaseAnswer.model_fields.keys())
    all_field_names = set(answer_class.model_fields.keys())
    user_fields = all_field_names - inherited

    verified_names = set(verified_fields.keys())
    if user_fields == verified_names:
        return "verified"
    return "mixed"
