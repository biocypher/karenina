"""Template using correct_dict pattern with model_post_init.

This template demonstrates the recommended pattern where ground truth
is set via model_post_init and accessed via self.correct dictionary.
This pattern allows for easy modification of ground truth without
changing the verify() logic.
"""

from typing import Any

from pydantic import Field

from karenina.schemas.entities import BaseAnswer


class Answer(BaseAnswer):
    """Template using correct_dict pattern for ground truth management."""

    # Extracted fields
    gene_name: str = Field(description="The gene symbol/name")
    chromosome: str = Field(description="Chromosome location")
    function: str = Field(description="Gene function description")

    # Optional fields
    synonyms: list[str] | None = Field(default=None, description="Alternative gene symbols")
    omim_id: int | None = Field(default=None, description="OMIM database ID")

    def model_post_init(self, __context):
        """Set ground truth after model initialization.

        This pattern stores all ground truth values in a dictionary
        for easy access and modification.
        """
        self.correct: dict[str, Any] = {
            "gene_name": "BCL2",
            "chromosome": "18q21.33",
            "function": "Apoptosis regulator that inhibits programmed cell death",
            "synonyms": ["Bcl-2", "BCL2"],
            "omim_id": 151430,
        }

    def verify(self) -> bool:
        """Verify extracted fields against ground truth.

        The verify() logic simply compares each field to the corresponding
        entry in self.correct, making it easy to add new fields.
        """
        # Check required string fields with case-insensitive comparison
        if self.gene_name.upper() != self.correct["gene_name"].upper():
            return False

        if self.chromosome != self.correct["chromosome"]:
            return False

        # Fuzzy match for function (allow minor wording differences)
        if self.correct["function"].lower() not in self.function.lower():
            return False

        # Check optional fields
        if self.synonyms is not None:
            # Convert to sets for comparison, case-insensitive
            synonym_set = {s.upper() for s in self.synonyms}
            correct_set = {s.upper() for s in self.correct["synonyms"]}
            if not synonym_set.issubset(correct_set):
                return False

        if self.omim_id is not None and self.omim_id != self.correct["omim_id"]:  # noqa: SIM103
            return False

        return True
