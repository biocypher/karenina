from karenina.question_qc.gates import derive_terminal_status, passes_evidence_gate
from karenina.question_qc.models import QcClassification, Review, Validation


def test_passes_evidence_gate() -> None:
    assert passes_evidence_gate(Validation(supports_claim=True, quality_issues=""))
    assert not passes_evidence_gate(Validation(supports_claim=True, quality_issues="x"))
    assert not passes_evidence_gate(Validation(supports_claim=False, quality_issues=""))
    assert not passes_evidence_gate(None)


def test_derive_terminal() -> None:
    assert derive_terminal_status(None, timed_out=True) == "timed_out"
    assert derive_terminal_status(None, error_stage="proposer", error_message="x") == "error"
    assert derive_terminal_status(None) == "inconclusive"
    assert (
        derive_terminal_status(Review(classification=QcClassification.SUPPORTED, reasoning="r"))
        == "supported"
    )
