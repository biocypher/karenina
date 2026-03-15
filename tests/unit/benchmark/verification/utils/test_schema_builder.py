"""Tests for build_parsing_schema utility."""

import pytest
from pydantic import Field

from karenina.benchmark.verification.utils.schema_builder import build_parsing_schema
from karenina.schemas.entities.answer import BaseAnswer
from karenina.schemas.entities.verified_field import VerifiedField
from karenina.schemas.primitives import ExactMatch, TraceRegex


@pytest.mark.unit
class TestBuildParsingSchema:
    """Test judge schema filtering."""

    def test_strips_verification_metadata(self):
        class MyAnswer(BaseAnswer):
            target: str = VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )

        schema = build_parsing_schema(MyAnswer)
        props = schema["properties"]["target"]
        assert "__verification__" not in props

    def test_excludes_trace_fields(self):
        class MyAnswer(BaseAnswer):
            target: str = VerifiedField(
                description="target",
                ground_truth="BCL2",
                verify_with=ExactMatch(),
            )
            has_citations: bool = VerifiedField(
                description="citations",
                ground_truth=True,
                verify_with=TraceRegex(pattern=r"\[\d+\]"),
            )

        schema = build_parsing_schema(MyAnswer)
        assert "target" in schema["properties"]
        assert "has_citations" not in schema["properties"]
        if "required" in schema:
            assert "has_citations" not in schema["required"]

    def test_classic_template_unchanged(self):
        class MyAnswer(BaseAnswer):
            target: str = Field(description="target", default="")

        schema = build_parsing_schema(MyAnswer)
        assert "target" in schema["properties"]

    def test_trace_only_returns_minimal_schema(self):
        class MyAnswer(BaseAnswer):
            has_citations: bool = VerifiedField(
                description="citations",
                ground_truth=True,
                verify_with=TraceRegex(pattern=r"\[\d+\]"),
            )

        schema = build_parsing_schema(MyAnswer)
        assert "has_citations" not in schema.get("properties", {})
