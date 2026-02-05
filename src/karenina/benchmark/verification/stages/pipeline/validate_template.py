"""Template validation stage.

Validates the Pydantic template syntax and injects question ID.
"""

from karenina.benchmark.authoring.answers.generator import inject_question_id_into_answer_class
from karenina.benchmark.verification.utils.template_validation import validate_answer_template

from ..core.base import ArtifactKeys, BaseVerificationStage, VerificationContext


class ValidateTemplateStage(BaseVerificationStage):
    """
    Validates template code and prepares Answer class.

    This stage:
    1. Validates the Pydantic template syntax
    2. Checks for required fields and correct structure
    3. Injects the question ID into the Answer class
    4. Stores the validated Answer class for use by later stages

    If validation fails, marks the context as failed and skips remaining stages.

    Produces:
        - "RawAnswer": The validated Pydantic class (before question ID injection)
        - "Answer": The Answer class with question ID injected
        - "template_validation_error": Error message if validation failed

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
        return [ArtifactKeys.RAW_ANSWER, ArtifactKeys.ANSWER, ArtifactKeys.TEMPLATE_VALIDATION_ERROR]

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
            - Sets context.error if validation fails
            - Sets context.completed_without_errors=False if validation fails
        """
        # Validate the template
        is_valid, error_msg, RawAnswer = validate_answer_template(context.template_code)

        if not is_valid or RawAnswer is None:
            # Template validation failed - mark error and stop pipeline
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
