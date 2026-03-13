"""Unit tests for template converter."""

import pytest

from karenina.benchmark.verification.utils.template_converter import (
    detect_template_mode,
    python_to_spec,
    spec_to_python,
    validate_spec,
)
from karenina.schemas.entities.template_spec import TemplateFieldSpec, TemplateSpec


@pytest.mark.unit
class TestDetectTemplateMode:
    def test_verified_template(self):
        code = """
from karenina.schemas.entities import BaseAnswer, VerifiedField, ExactMatch

class Answer(BaseAnswer):
    target: str = VerifiedField(
        description="target",
        ground_truth="BCL2",
        verify_with=ExactMatch(),
    )
"""
        assert detect_template_mode(code) == "verified"

    def test_classic_template(self):
        code = """
from pydantic import Field
from karenina.schemas.entities import BaseAnswer

class Answer(BaseAnswer):
    target: str = Field(description="target")

    def ground_truth(self):
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target == self.correct["target"]
"""
        assert detect_template_mode(code) == "classic"

    def test_mixed_template(self):
        code = """
from pydantic import Field
from karenina.schemas.entities import BaseAnswer, VerifiedField, ExactMatch

class Answer(BaseAnswer):
    target: str = VerifiedField(
        description="target",
        ground_truth="BCL2",
        verify_with=ExactMatch(),
    )
    notes: str = Field(description="extra notes", default="")
"""
        assert detect_template_mode(code) == "mixed"

    def test_invalid_code_raises(self):
        with pytest.raises(ValueError, match="Failed to compile"):
            detect_template_mode("def broken(")

    def test_no_answer_class_raises(self):
        code = """
class NotAnAnswer:
    pass
"""
        with pytest.raises(ValueError, match="No Answer class found"):
            detect_template_mode(code)


@pytest.mark.unit
class TestPythonToSpec:
    def test_basic_verified_template(self):
        code = """
from karenina.schemas.entities import BaseAnswer, VerifiedField, ExactMatch, BooleanMatch

class DrugAnswer(BaseAnswer):
    target: str = VerifiedField(
        description="Protein target",
        ground_truth="BCL2",
        verify_with=ExactMatch(normalize=["lowercase", "strip"]),
    )
    is_approved: bool = VerifiedField(
        description="FDA approved",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
"""
        spec = python_to_spec(code)
        assert isinstance(spec, TemplateSpec)
        assert spec.class_name == "DrugAnswer"
        assert len(spec.fields) == 2
        assert spec.fields[0].name == "target"
        assert spec.fields[0].ground_truth == "BCL2"
        assert spec.fields[0].verify_with["type"] == "ExactMatch"
        assert spec.fields[1].name == "is_approved"
        assert spec.fields[1].verify_with["type"] == "BooleanMatch"

    def test_trace_field_detected(self):
        code = """
from karenina.schemas.entities import BaseAnswer, VerifiedField, TraceRegex

class Answer(BaseAnswer):
    has_cites: bool = VerifiedField(
        description="citations",
        ground_truth=True,
        verify_with=TraceRegex(pattern=r"\\[\\d+\\]"),
    )
"""
        spec = python_to_spec(code)
        assert spec.fields[0].is_trace is True

    def test_extraction_hint_preserved(self):
        code = """
from karenina.schemas.entities import BaseAnswer, VerifiedField, ExactMatch

class Answer(BaseAnswer):
    target: str = VerifiedField(
        description="target",
        extraction_hint="Normalize to uppercase",
        ground_truth="BCL2",
        verify_with=ExactMatch(),
    )
"""
        spec = python_to_spec(code)
        assert spec.fields[0].extraction_hint == "Normalize to uppercase"

    def test_classic_template_raises(self):
        code = """
from karenina.schemas.entities import BaseAnswer
from pydantic import Field

class Answer(BaseAnswer):
    target: str = Field(description="target")

    def verify(self) -> bool:
        return True
"""
        with pytest.raises(ValueError, match="classic"):
            python_to_spec(code)

    def test_mixed_template_only_converts_verified_fields(self):
        """Mixed templates: only VerifiedField fields appear in the spec."""
        code = """
from pydantic import Field
from karenina.schemas.entities import BaseAnswer, VerifiedField, ExactMatch

class Answer(BaseAnswer):
    target: str = VerifiedField(
        description="target",
        ground_truth="BCL2",
        verify_with=ExactMatch(),
    )
    notes: str = Field(description="extra notes", default="")
"""
        spec = python_to_spec(code)
        assert len(spec.fields) == 1
        assert spec.fields[0].name == "target"

    def test_field_types_detected(self):
        code = """
from karenina.schemas.entities import BaseAnswer, VerifiedField, ExactMatch, BooleanMatch, NumericExact

class Answer(BaseAnswer):
    name: str = VerifiedField(
        description="name",
        ground_truth="test",
        verify_with=ExactMatch(),
    )
    count: int = VerifiedField(
        description="count",
        ground_truth=42,
        verify_with=NumericExact(),
    )
    active: bool = VerifiedField(
        description="active",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
"""
        spec = python_to_spec(code)
        type_map = {f.name: f.type for f in spec.fields}
        assert type_map["name"] == "str"
        assert type_map["count"] == "int"
        assert type_map["active"] == "bool"

    def test_weight_preserved(self):
        code = """
from karenina.schemas.entities import BaseAnswer, VerifiedField, ExactMatch

class Answer(BaseAnswer):
    target: str = VerifiedField(
        description="target",
        ground_truth="BCL2",
        verify_with=ExactMatch(),
        weight=2.5,
    )
"""
        spec = python_to_spec(code)
        assert spec.fields[0].weight == 2.5


@pytest.mark.unit
class TestSpecToPython:
    def test_basic_two_field_template(self):
        spec = TemplateSpec(
            fields=[
                TemplateFieldSpec(
                    name="target",
                    type="str",
                    description="Protein target",
                    ground_truth="BCL2",
                    verify_with={"type": "ExactMatch", "normalize": ["lowercase", "strip"]},
                ),
                TemplateFieldSpec(
                    name="is_approved",
                    type="bool",
                    description="FDA approved",
                    ground_truth=True,
                    verify_with={"type": "BooleanMatch"},
                ),
            ],
        )
        code = spec_to_python(spec)
        assert "class Answer(BaseAnswer):" in code
        assert "VerifiedField(" in code
        assert "ground_truth='BCL2'" in code
        assert "ExactMatch(" in code
        assert "BooleanMatch()" in code

    def test_custom_class_name(self):
        spec = TemplateSpec(
            class_name="DrugAnswer",
            fields=[
                TemplateFieldSpec(
                    name="target",
                    type="str",
                    description="target",
                    ground_truth="BCL2",
                    verify_with={"type": "ExactMatch"},
                ),
            ],
        )
        code = spec_to_python(spec)
        assert "class DrugAnswer(BaseAnswer):" in code

    def test_generated_code_is_executable(self):
        """Round-trip: spec -> python -> exec -> verify."""
        spec = TemplateSpec(
            fields=[
                TemplateFieldSpec(
                    name="target",
                    type="str",
                    description="target",
                    ground_truth="BCL2",
                    verify_with={"type": "ExactMatch"},
                ),
            ],
        )
        code = spec_to_python(spec)
        ns = {}
        exec(code, ns)
        from karenina.schemas.entities import BaseAnswer

        cls = next(v for v in ns.values() if isinstance(v, type) and issubclass(v, BaseAnswer) and v is not BaseAnswer)
        instance = cls(target="BCL2")
        assert instance.verify() is True

    def test_round_trip_python_to_spec_to_python(self):
        """Full round-trip: python -> spec -> python -> verify same results."""
        original_code = """
from karenina.schemas.entities import BaseAnswer, VerifiedField, ExactMatch, BooleanMatch

class Answer(BaseAnswer):
    target: str = VerifiedField(
        description="Protein target",
        ground_truth="BCL2",
        verify_with=ExactMatch(normalize=["lowercase", "strip"]),
    )
    is_approved: bool = VerifiedField(
        description="FDA approved",
        ground_truth=True,
        verify_with=BooleanMatch(),
    )
"""
        spec = python_to_spec(original_code)
        regenerated_code = spec_to_python(spec)

        from karenina.benchmark.verification.utils.class_discovery import find_answer_class
        from karenina.benchmark.verification.utils.template_validation import _build_exec_namespace

        ns = _build_exec_namespace()
        exec(regenerated_code, ns)
        cls = find_answer_class(ns)
        instance = cls(target="bcl2", is_approved=True)
        assert instance.verify() is True

    def test_trace_field_generated(self):
        spec = TemplateSpec(
            fields=[
                TemplateFieldSpec(
                    name="has_cites",
                    type="bool",
                    description="Has citations",
                    ground_truth=True,
                    verify_with={"type": "TraceRegex", "pattern": r"\[\d+\]"},
                    is_trace=True,
                ),
            ],
        )
        code = spec_to_python(spec)
        assert "TraceRegex(" in code

    def test_literal_field_generated(self):
        spec = TemplateSpec(
            fields=[
                TemplateFieldSpec(
                    name="category",
                    type="literal",
                    description="Drug category",
                    ground_truth="inhibitor",
                    literal_values=["inhibitor", "antibody", "small_molecule"],
                    verify_with={"type": "LiteralMatch"},
                ),
            ],
        )
        code = spec_to_python(spec)
        assert "Literal[" in code
        assert "LiteralMatch()" in code

    def test_imports_only_used_primitives(self):
        spec = TemplateSpec(
            fields=[
                TemplateFieldSpec(
                    name="target",
                    type="str",
                    description="target",
                    ground_truth="BCL2",
                    verify_with={"type": "ExactMatch"},
                ),
            ],
        )
        code = spec_to_python(spec)
        assert "ExactMatch" in code
        assert "BooleanMatch" not in code
        assert "NumericTolerance" not in code


def _field(name="target", type="str", ground_truth="BCL2", verify_with=None, **kwargs):
    """Helper to build a TemplateFieldSpec with sensible defaults."""
    return TemplateFieldSpec(
        name=name,
        type=type,
        description=f"Test field {name}",
        ground_truth=ground_truth,
        verify_with=verify_with or {"type": "ExactMatch"},
        **kwargs,
    )


@pytest.mark.unit
class TestValidateSpec:
    def test_valid_spec_returns_no_errors(self):
        spec = TemplateSpec(
            fields=[
                _field(name="target", type="str", verify_with={"type": "ExactMatch"}),
                _field(name="approved", type="bool", ground_truth=True, verify_with={"type": "BooleanMatch"}),
            ],
        )
        errors = validate_spec(spec)
        assert errors == []

    def test_unknown_primitive_name(self):
        spec = TemplateSpec(
            fields=[_field(verify_with={"type": "FuzzyMatch", "threshold": 0.6})],
        )
        errors = validate_spec(spec)
        assert len(errors) == 1
        assert "FuzzyMatch" in errors[0]
        assert "unknown primitive" in errors[0].lower()

    def test_missing_type_key_in_verify_with(self):
        spec = TemplateSpec(
            fields=[_field(verify_with={"threshold": 0.6})],
        )
        errors = validate_spec(spec)
        assert len(errors) == 1
        assert "'type'" in errors[0]

    def test_invalid_primitive_parameters(self):
        spec = TemplateSpec(
            fields=[_field(verify_with={"type": "ContainsAll", "terms": ["a", "b"]})],
        )
        errors = validate_spec(spec)
        assert len(errors) == 1
        assert "ContainsAll" in errors[0]

    def test_primitive_field_type_mismatch(self):
        spec = TemplateSpec(
            fields=[
                _field(name="flag", type="bool", ground_truth=True, verify_with={"type": "ExactMatch"}),
            ],
        )
        errors = validate_spec(spec)
        assert len(errors) == 1
        assert "not compatible" in errors[0].lower()

    def test_invalid_field_type(self):
        spec = TemplateSpec(
            fields=[_field(type="object")],
        )
        errors = validate_spec(spec)
        assert len(errors) >= 1
        assert any("object" in e for e in errors)

    def test_literal_field_missing_literal_values(self):
        spec = TemplateSpec(
            fields=[
                _field(
                    type="literal",
                    ground_truth="a",
                    verify_with={"type": "LiteralMatch"},
                    literal_values=None,
                ),
            ],
        )
        errors = validate_spec(spec)
        assert len(errors) == 1
        assert "literal_values" in errors[0].lower()

    def test_trace_primitive_on_non_bool_field(self):
        spec = TemplateSpec(
            fields=[
                _field(
                    type="str",
                    verify_with={"type": "TraceContains", "substring": "hello"},
                    is_trace=True,
                ),
            ],
        )
        errors = validate_spec(spec)
        assert any("trace" in e.lower() and "bool" in e.lower() for e in errors)

    def test_multiple_errors_reported(self):
        spec = TemplateSpec(
            fields=[
                _field(name="bad_prim", verify_with={"type": "FuzzyMatch"}),
                _field(name="bad_type", type="object"),
            ],
        )
        errors = validate_spec(spec)
        assert len(errors) >= 2

    def test_spec_to_python_rejects_invalid_spec(self):
        """spec_to_python should raise ValueError for invalid specs."""
        spec = TemplateSpec(
            fields=[_field(verify_with={"type": "FuzzyMatch"})],
        )
        with pytest.raises(ValueError, match="FuzzyMatch"):
            spec_to_python(spec)
