"""Integration tests: VerifiedField templates through pipeline stages."""

import pytest
from pydantic import Field

from karenina.benchmark.verification.utils.class_discovery import find_answer_class
from karenina.benchmark.verification.utils.schema_builder import build_parsing_schema
from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.verified_field import VerifiedField
from karenina.schemas.primitives import BooleanMatch, ExactMatch, TraceRegex


@pytest.mark.integration
class TestVerifiedFieldPipeline:
    """Test VerifiedField templates through the pipeline."""

    def test_verified_template_validate_parse_verify(self):
        """Full path: validate -> schema build -> verify for verified template."""

        class DrugAnswer(BaseAnswer):
            target: str = VerifiedField(
                description="Protein target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )
            is_approved: bool = VerifiedField(
                description="FDA approved",
                ground_truth=True,
                verify_with=BooleanMatch(),
            )

        # Schema builder strips __verification__
        schema = build_parsing_schema(DrugAnswer)
        assert "__verification__" not in str(schema["properties"]["target"])
        assert "target" in schema["properties"]

        # Verify works without explicit verify()
        answer = DrugAnswer(target="bcl2", is_approved=True)
        assert answer.verify() is True
        assert answer.verify_granular() == 1.0

    def test_classic_template_backward_compat(self):
        """Classic template (no VerifiedField) still works through pipeline."""

        class ClassicAnswer(BaseAnswer):
            target: str = Field(description="target", default="")

            def ground_truth(self):
                self.correct = {"target": "BCL2"}

            def verify(self) -> bool:
                return self.target.lower() == self.correct["target"].lower()

        answer = ClassicAnswer(target="bcl2")
        assert answer.verify() is True
        schema = build_parsing_schema(ClassicAnswer)
        assert "target" in schema["properties"]

    def test_mixed_template(self):
        """Template with both VerifiedField and plain Field."""

        class MixedAnswer(BaseAnswer):
            target: str = VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )
            notes: str = Field(description="extra notes", default="")

        answer = MixedAnswer(target="BCL2", notes="some notes")
        assert answer.verify() is True
        schema = build_parsing_schema(MixedAnswer)
        assert "target" in schema["properties"]
        assert "notes" in schema["properties"]

    def test_trace_only_template(self):
        """All-trace template: no parsed fields for judge."""

        class TraceAnswer(BaseAnswer):
            has_cites: bool = VerifiedField(
                description="cites",
                ground_truth=True,
                verify_with=TraceRegex(pattern=r"\[\d+\]"),
            )

        schema = build_parsing_schema(TraceAnswer)
        assert "has_cites" not in schema.get("properties", {})

        answer = TraceAnswer(has_cites=False)
        answer._raw_trace = "See [1] and [2]"
        assert answer.verify() is True

    def test_find_answer_class_in_exec_namespace(self):
        """Simulate exec() of template code and class discovery."""
        code = """
from karenina.schemas.entities import BaseAnswer, VerifiedField, ExactMatch

class VenetoclaxAnswer(BaseAnswer):
    target: str = VerifiedField(
        description="target", ground_truth="BCL2",
        verify_with=ExactMatch(),
    )
"""
        ns = {}
        exec(code, ns)
        cls = find_answer_class(ns)
        assert cls.__name__ == "VenetoclaxAnswer"
        answer = cls(target="bcl2")
        assert answer.verify() is True
