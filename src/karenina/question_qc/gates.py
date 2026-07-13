"""Evidence acceptance and terminal status derivation for question QC."""

from __future__ import annotations

from .models import QcClassification, Review, Validation


def passes_evidence_gate(validation: Validation | None) -> bool:
    """Accept only when the claim is supported and quality issues are empty."""
    if validation is None:
        return False
    return validation.passes_evidence_gate


def derive_terminal_status(
    review: Review | None,
    *,
    error_stage: str = "",
    error_message: str = "",
    timed_out: bool = False,
) -> str:
    """Derive a terminal status string for a QC result."""
    if timed_out:
        return "timed_out"
    if error_stage or error_message:
        return "error"
    if review is None:
        return "inconclusive"
    if review.classification is None:
        return "inconclusive"
    return str(review.classification.value)


def normalize_classification(value: str | None) -> QcClassification | None:
    """Map free-form classification strings to QcClassification."""
    if value is None:
        return None
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "supported": QcClassification.SUPPORTED,
        "pass": QcClassification.SUPPORTED,
        "unsupported": QcClassification.UNSUPPORTED,
        "question_unanswerable": QcClassification.UNSUPPORTED,
        "question_unanswerable_by_kg": QcClassification.UNSUPPORTED,
        "ill_formed": QcClassification.ILL_FORMED,
        "malformed": QcClassification.ILL_FORMED,
        "question_malformed": QcClassification.ILL_FORMED,
        "inconclusive": QcClassification.INCONCLUSIVE,
        "uncertain": QcClassification.INCONCLUSIVE,
        "rerun_needed": QcClassification.INCONCLUSIVE,
    }
    if key in aliases:
        return aliases[key]
    try:
        return QcClassification(key)
    except ValueError:
        return None
