"""Core verification logic for benchmark system."""

from .verification.multi_model_orchestrator import run_question_verification
from .verification.runner import run_single_model_verification
from .verification.utils.validation import validate_answer_template

# Re-export the main functions to maintain backward compatibility
__all__ = ["validate_answer_template", "run_single_model_verification", "run_question_verification"]
