"""Unit tests for generator_code.py (VerifiedField-based class generation)."""

from typing import Any

import pytest

from karenina.benchmark.authoring.answers.generator_code import (
    generate_pydantic_class,
)


def _make_spec(attributes: list[dict[str, Any]], descriptions: dict[str, str] | None = None) -> dict[str, Any]:
    """Build a minimal spec dict for generate_pydantic_class.

    Args:
        attributes: List of dicts with "name", "type", and "ground_truth" keys.
        descriptions: Optional mapping of field name to description. If omitted,
            a placeholder description is generated from the field name.

    Returns:
        Spec dict ready for generate_pydantic_class.
    """
    if descriptions is None:
        descriptions = {attr["name"]: f"The {attr['name']} value" for attr in attributes}
    return {"attributes": attributes, "field_descriptions": descriptions}


@pytest.mark.unit
class TestGeneratePydanticClass:
    """Tests for generate_pydantic_class producing VerifiedField templates."""

    def test_generate_single_str_field(self) -> None:
        """Single string field maps to ExactMatch primitive."""
        spec = _make_spec(
            [{"name": "target", "type": "str", "ground_truth": "BCL2"}],
            {"target": "The protein target of the drug"},
        )
        code = generate_pydantic_class(spec)

        assert "VerifiedField(" in code
        assert "ExactMatch" in code
        assert 'normalize=["lowercase", "strip"]' in code
        assert "BooleanMatch" not in code
        assert "ground_truth=" in code
        assert "'BCL2'" in code

    def test_generate_bool_field(self) -> None:
        """Bool field maps to BooleanMatch primitive."""
        spec = _make_spec(
            [{"name": "is_approved", "type": "bool", "ground_truth": True}],
            {"is_approved": "Whether the drug is FDA-approved"},
        )
        code = generate_pydantic_class(spec)

        assert "VerifiedField(" in code
        assert "BooleanMatch()" in code
        assert "ExactMatch" not in code
        assert "ground_truth=True" in code

    def test_generate_float_field(self) -> None:
        """Float field maps to NumericTolerance primitive with default tolerance."""
        spec = _make_spec(
            [{"name": "score", "type": "float", "ground_truth": 0.95}],
            {"score": "The confidence score"},
        )
        code = generate_pydantic_class(spec)

        assert "VerifiedField(" in code
        assert "NumericTolerance(" in code
        assert 'mode="relative"' in code
        assert "tolerance=" in code
        assert "0.95" in code

    def test_generate_float_field_custom_tolerance(self) -> None:
        """Float field uses the float_tolerance parameter passed to generate_pydantic_class."""
        spec = _make_spec(
            [{"name": "score", "type": "float", "ground_truth": 0.5}],
        )
        code = generate_pydantic_class(spec, float_tolerance=0.05)

        assert "tolerance=0.05" in code

    def test_generate_list_field(self) -> None:
        """List[str] field maps to SetContainment primitive."""
        spec = _make_spec(
            [{"name": "targets", "type": "List[str]", "ground_truth": ["BCL2", "MCL1"]}],
            {"targets": "The protein targets"},
        )
        code = generate_pydantic_class(spec)

        assert "VerifiedField(" in code
        assert "SetContainment(" in code
        assert 'mode="exact"' in code
        assert "['BCL2', 'MCL1']" in code

    def test_generate_list_field_lowercase_syntax(self) -> None:
        """list[str] (lowercase) also maps to SetContainment."""
        spec = _make_spec(
            [{"name": "tags", "type": "list[str]", "ground_truth": ["a", "b"]}],
        )
        code = generate_pydantic_class(spec)

        assert "SetContainment(" in code

    def test_generate_literal_field(self) -> None:
        """Literal type maps to LiteralMatch primitive."""
        spec = _make_spec(
            [{"name": "category", "type": "Literal['drug', 'target']", "ground_truth": "drug"}],
            {"category": "The category of the entity"},
        )
        code = generate_pydantic_class(spec)

        assert "VerifiedField(" in code
        assert "LiteralMatch()" in code
        assert "'drug'" in code

    def test_generate_int_field(self) -> None:
        """Int field maps to NumericExact primitive."""
        spec = _make_spec(
            [{"name": "count", "type": "int", "ground_truth": 42}],
        )
        code = generate_pydantic_class(spec)

        assert "NumericExact()" in code
        assert "ground_truth=42" in code

    def test_generate_multiple_fields(self) -> None:
        """Multiple fields each get their own VerifiedField with correct primitives."""
        spec = _make_spec(
            [
                {"name": "target", "type": "str", "ground_truth": "BCL2"},
                {"name": "is_approved", "type": "bool", "ground_truth": True},
                {"name": "score", "type": "float", "ground_truth": 0.9},
            ],
            {
                "target": "The protein target",
                "is_approved": "FDA approval status",
                "score": "Confidence score",
            },
        )
        code = generate_pydantic_class(spec)

        assert "ExactMatch" in code
        assert "BooleanMatch" in code
        assert "NumericTolerance" in code
        assert code.count("VerifiedField(") == 3
        # No classic verify methods should appear
        assert "def verify(" not in code
        assert "def ground_truth(" not in code
        assert "def verify_granular(" not in code

    def test_no_verify_methods_generated(self) -> None:
        """Generated code contains no manual verify(), ground_truth(), or verify_granular()."""
        spec = _make_spec(
            [{"name": "x", "type": "bool", "ground_truth": True}],
        )
        code = generate_pydantic_class(spec)

        assert "def verify(" not in code
        assert "def ground_truth(" not in code
        assert "def verify_granular(" not in code

    def test_generated_code_is_executable(self) -> None:
        """exec() on generated code produces a working Answer class."""
        spec = _make_spec(
            [
                {"name": "target", "type": "str", "ground_truth": "BCL2"},
                {"name": "is_approved", "type": "bool", "ground_truth": True},
            ],
            {
                "target": "The protein target of the drug",
                "is_approved": "Whether the drug is FDA-approved",
            },
        )
        code = generate_pydantic_class(spec)

        # Use a single dict for both globals and locals so that the import
        # statement at the top of generated code is visible inside class bodies.
        exec_ns: dict[str, Any] = {"__builtins__": __builtins__}
        exec(code, exec_ns)  # noqa: S102
        Answer = exec_ns["Answer"]

        # Instantiate with correct values; verify() should pass
        correct_answer = Answer(target="bcl2", is_approved=True)
        assert correct_answer.verify() is True

        # Instantiate with wrong string; verify() should fail
        wrong_answer = Answer(target="MCL1", is_approved=True)
        assert wrong_answer.verify() is False

    def test_generated_code_verify_granular(self) -> None:
        """exec()-created class supports verify_granular() returning partial scores."""
        spec = _make_spec(
            [
                {"name": "target", "type": "str", "ground_truth": "BCL2"},
                {"name": "is_approved", "type": "bool", "ground_truth": True},
            ],
            {
                "target": "The protein target",
                "is_approved": "FDA approval status",
            },
        )
        code = generate_pydantic_class(spec)

        exec_ns: dict[str, Any] = {"__builtins__": __builtins__}
        exec(code, exec_ns)  # noqa: S102
        Answer = exec_ns["Answer"]

        # One of two fields correct: score should be 0.5
        partial = Answer(target="bcl2", is_approved=False)
        score = partial.verify_granular()
        assert score == pytest.approx(0.5)

        # All correct: score should be 1.0
        full = Answer(target="bcl2", is_approved=True)
        assert full.verify_granular() == pytest.approx(1.0)

    def test_imports_only_used_primitives(self) -> None:
        """Import line contains only the primitives actually used in the class."""
        # Bool-only spec: only BooleanMatch should appear in imports
        spec_bool = _make_spec(
            [{"name": "flag", "type": "bool", "ground_truth": False}],
        )
        code_bool = generate_pydantic_class(spec_bool)
        import_line_bool = code_bool.split("\n")[0]

        assert "BooleanMatch" in import_line_bool
        assert "ExactMatch" not in import_line_bool
        assert "NumericTolerance" not in import_line_bool
        assert "SetContainment" not in import_line_bool
        assert "BaseAnswer" in import_line_bool
        assert "VerifiedField" in import_line_bool

        # Str-only spec: only ExactMatch should appear in imports
        spec_str = _make_spec(
            [{"name": "name", "type": "str", "ground_truth": "x"}],
        )
        code_str = generate_pydantic_class(spec_str)
        import_line_str = code_str.split("\n")[0]

        assert "ExactMatch" in import_line_str
        assert "BooleanMatch" not in import_line_str

    def test_import_line_is_first_line(self) -> None:
        """The import statement appears on the very first line of generated code."""
        spec = _make_spec(
            [{"name": "value", "type": "str", "ground_truth": "test"}],
        )
        code = generate_pydantic_class(spec)
        first_line = code.split("\n")[0]

        assert first_line.startswith("from karenina.schemas.entities import ")

    def test_custom_class_name(self) -> None:
        """class_name parameter controls the generated class name."""
        spec = _make_spec(
            [{"name": "result", "type": "bool", "ground_truth": True}],
        )
        code = generate_pydantic_class(spec, class_name="MyAnswer")

        assert "class MyAnswer(BaseAnswer):" in code
        assert "class Answer(" not in code

    def test_field_description_included(self) -> None:
        """Field descriptions from spec appear in the VerifiedField call."""
        spec = _make_spec(
            [{"name": "target", "type": "str", "ground_truth": "BCL2"}],
            {"target": "The specific protein target inhibited by the drug"},
        )
        code = generate_pydantic_class(spec)

        assert "The specific protein target inhibited by the drug" in code
