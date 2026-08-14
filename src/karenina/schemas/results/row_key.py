"""Canonical identity key for one verification result row.

A results file holds one row per question, answering model, parsing model,
and replicate. Post-hoc analyses often collapse parser siblings and work per
generated trace, which is what ``trace_identity`` expresses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from karenina.schemas.verification.result import VerificationResult


@dataclass(frozen=True, slots=True)
class ResultRowKey:
    """Identity of one result row.

    Attributes:
        question_id: The benchmark question id.
        answering_key: Canonical key of the answering model.
        parsing_key: Canonical key of the parsing model.
        replicate: Replicate index, or None for single-shot runs.
        run_name: Run label, or None when unnamed.
    """

    question_id: str
    answering_key: str
    parsing_key: str
    replicate: int | None
    run_name: str | None

    @classmethod
    def from_result(cls, result: VerificationResult) -> ResultRowKey:
        """Build a key from a verification result's metadata."""
        metadata = result.metadata
        return cls(
            question_id=metadata.question_id,
            answering_key=metadata.answering.canonical_key,
            parsing_key=metadata.parsing.canonical_key,
            replicate=metadata.replicate,
            run_name=metadata.run_name,
        )

    @property
    def trace_identity(self) -> tuple[str, str, int | None]:
        """Return the parser-agnostic identity of the generated answer."""
        return (self.question_id, self.answering_key, self.replicate)
