"""Unit tests for TemplateSpec models."""

import pytest

from karenina.schemas.entities.template_spec import (
    TemplateFieldSpec,
    TemplateSpec,
    VerifyStrategySpec,
)


@pytest.mark.unit
class TestTemplateFieldSpec:
    def test_basic_bool_field(self):
        field = TemplateFieldSpec(
            name="is_approved",
            type="bool",
            description="FDA approved",
            ground_truth=True,
            verify_with={"type": "BooleanMatch"},
        )
        assert field.name == "is_approved"
        assert field.weight == 1.0
        assert field.is_trace is False

    def test_str_field_with_extraction_hint(self):
        field = TemplateFieldSpec(
            name="target",
            type="str",
            description="Protein target",
            extraction_hint="Normalize to uppercase gene symbol",
            ground_truth="BCL2",
            verify_with={"type": "ExactMatch", "normalize": ["lowercase", "strip"]},
        )
        assert field.extraction_hint == "Normalize to uppercase gene symbol"

    def test_literal_field(self):
        field = TemplateFieldSpec(
            name="category",
            type="literal",
            description="Drug category",
            ground_truth="kinase_inhibitor",
            literal_values=[
                "kinase_inhibitor",
                "monoclonal_antibody",
                "small_molecule",
            ],
            verify_with={"type": "LiteralMatch"},
        )
        assert field.literal_values == [
            "kinase_inhibitor",
            "monoclonal_antibody",
            "small_molecule",
        ]

    def test_trace_field(self):
        field = TemplateFieldSpec(
            name="has_citations",
            type="bool",
            description="Has citations",
            ground_truth=True,
            verify_with={"type": "TraceRegex", "pattern": r"\[\d+\]"},
            is_trace=True,
        )
        assert field.is_trace is True


@pytest.mark.unit
class TestVerifyStrategySpec:
    def test_all_of_default(self):
        strategy = VerifyStrategySpec(type="all_of", conditions=[])
        assert strategy.type == "all_of"
        assert strategy.n is None

    def test_at_least_n(self):
        strategy = VerifyStrategySpec(
            type="at_least_n",
            n=2,
            conditions=[
                VerifyStrategySpec(type="field_check", field_name="a"),
                VerifyStrategySpec(type="field_check", field_name="b"),
                VerifyStrategySpec(type="field_check", field_name="c"),
            ],
        )
        assert strategy.n == 2
        assert len(strategy.conditions) == 3


@pytest.mark.unit
class TestTemplateSpec:
    def test_basic_spec(self):
        spec = TemplateSpec(
            fields=[
                TemplateFieldSpec(
                    name="target",
                    type="str",
                    description="Target",
                    ground_truth="BCL2",
                    verify_with={"type": "ExactMatch"},
                ),
            ],
        )
        assert len(spec.fields) == 1
        assert spec.verify_strategy is None  # Default: None means AllOf all fields

    def test_round_trip_serialization(self):
        spec = TemplateSpec(
            fields=[
                TemplateFieldSpec(
                    name="target",
                    type="str",
                    description="Target",
                    ground_truth="BCL2",
                    verify_with={"type": "ExactMatch"},
                ),
            ],
            verify_strategy=VerifyStrategySpec(type="all_of", conditions=[]),
        )
        data = spec.model_dump(mode="json")
        restored = TemplateSpec.model_validate(data)
        assert restored.fields[0].name == "target"
