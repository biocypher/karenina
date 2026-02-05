"""Test that answer template fixtures are valid and work correctly.

This module verifies the template fixtures in tests/fixtures/templates/:
- simple_extraction.py - Single field extraction
- multi_field.py - Nested and complex types
- with_correct_dict.py - Using model_post_init for ground truth
"""

from pathlib import Path

import pytest

from karenina.schemas.entities import BaseAnswer

# Import each template fixture for testing
# Note: Templates must be imported as modules and the Answer class extracted


def _load_template(template_path: Path, template_name: str = "Answer") -> type[BaseAnswer]:
    """Load an Answer class from a template file.

    Args:
        template_path: Path to the template .py file
        template_name: Name of the Answer class (default "Answer")

    Returns:
        The Answer class
    """
    import importlib.util
    import sys

    # Load the module from file
    spec = importlib.util.spec_from_file_location("template_module", template_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {template_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["template_module"] = module
    spec.loader.exec_module(module)

    # Get the Answer class from the module
    answer_class = getattr(module, template_name)
    return answer_class


@pytest.mark.unit
def test_simple_extraction_template_loads(fixtures_dir: Path) -> None:
    """Test that simple_extraction.py can be imported and instantiated."""
    template_path = fixtures_dir / "templates" / "simple_extraction.py"
    Answer = _load_template(template_path)

    # Verify it inherits from BaseAnswer
    assert issubclass(Answer, BaseAnswer)

    # Verify class name is "Answer"
    assert Answer.__name__ == "Answer"

    # Can instantiate with test data
    answer = Answer(value="42")
    assert answer.value == "42"


@pytest.mark.unit
def test_simple_extraction_template_verify(fixtures_dir: Path) -> None:
    """Test that simple_extraction.py verify() works correctly."""
    template_path = fixtures_dir / "templates" / "simple_extraction.py"
    Answer = _load_template(template_path)

    # Correct answer should pass
    correct = Answer(value="42")
    assert correct.verify() is True

    # Wrong answer should fail
    wrong = Answer(value="43")
    assert wrong.verify() is False

    # Whitespace handling
    with_spaces = Answer(value="  42  ")
    # Verify strips whitespace for comparison
    assert with_spaces.verify() is True


@pytest.mark.unit
def test_simple_extraction_has_correct_dict(fixtures_dir: Path) -> None:
    """Test that simple_extraction has model_post_init setting ground truth."""
    template_path = fixtures_dir / "templates" / "simple_extraction.py"
    Answer = _load_template(template_path)

    answer = Answer(value="42")
    assert hasattr(answer, "correct")
    assert answer.correct == {"value": "42"}


@pytest.mark.unit
def test_multi_field_template_loads(fixtures_dir: Path) -> None:
    """Test that multi_field.py can be imported and instantiated."""
    template_path = fixtures_dir / "templates" / "multi_field.py"
    Answer = _load_template(template_path)

    assert issubclass(Answer, BaseAnswer)
    assert Answer.__name__ == "Answer"

    # Can instantiate with test data
    answer = Answer(
        main_answer="The mitochondria is the powerhouse of the cell",
        confidence=0.95,
        keywords=["mitochondria", "cell"],
        entities=["cell"],
    )
    assert answer.main_answer == "The mitochondria is the powerhouse of the cell"
    assert answer.confidence == 0.95
    assert answer.keywords == ["mitochondria", "cell"]


@pytest.mark.unit
def test_multi_field_template_with_nested_citation(fixtures_dir: Path) -> None:
    """Test that multi_field.py nested Citation structure works."""
    template_path = fixtures_dir / "templates" / "multi_field.py"
    Answer = _load_template(template_path)

    # Instantiate with nested citation
    citation_class = Answer.__annotations__["citation"].__args__[0]  # Get the Citation class
    answer = Answer(
        main_answer="The mitochondria is the powerhouse of the cell",
        confidence=0.95,
        keywords=["mitochondria", "cell", "powerhouse", "organelle"],
        entities=["cell", "mitochondria"],
        citation=citation_class(identifier="[1]", page=42),
    )

    assert answer.citation is not None
    assert answer.citation.identifier == "[1]"
    assert answer.citation.page == 42


@pytest.mark.unit
def test_multi_field_template_verify(fixtures_dir: Path) -> None:
    """Test that multi_field.py verify() works correctly."""
    template_path = fixtures_dir / "templates" / "multi_field.py"
    Answer = _load_template(template_path)

    citation_class = Answer.__annotations__["citation"].__args__[0]

    # Correct data should pass
    correct = Answer(
        main_answer="The mitochondria is the powerhouse of the cell",
        confidence=0.95,
        keywords=["mitochondria", "cell", "powerhouse", "organelle"],
        entities=["cell", "mitochondria"],
        citation=citation_class(identifier="[1]", page=42),
    )
    assert correct.verify() is True

    # Wrong main_answer should fail
    wrong = Answer(
        main_answer="Wrong answer",
        confidence=0.95,
        keywords=["mitochondria", "cell", "powerhouse", "organelle"],
        entities=["cell", "mitochondria"],
        citation=citation_class(identifier="[1]", page=42),
    )
    assert wrong.verify() is False


@pytest.mark.unit
def test_with_correct_dict_template_loads(fixtures_dir: Path) -> None:
    """Test that with_correct_dict.py can be imported and instantiated."""
    template_path = fixtures_dir / "templates" / "with_correct_dict.py"
    Answer = _load_template(template_path)

    assert issubclass(Answer, BaseAnswer)
    assert Answer.__name__ == "Answer"

    # Can instantiate with test data
    answer = Answer(
        gene_name="BCL2",
        chromosome="18q21.33",
        function="Apoptosis regulator that inhibits programmed cell death",
    )
    assert answer.gene_name == "BCL2"


@pytest.mark.unit
def test_with_correct_dict_template_case_insensitive(fixtures_dir: Path) -> None:
    """Test that with_correct_dict.py does case-insensitive comparison."""
    template_path = fixtures_dir / "templates" / "with_correct_dict.py"
    Answer = _load_template(template_path)

    # Lowercase gene name should still match, function contains ground truth
    answer = Answer(
        gene_name="bcl2",
        chromosome="18q21.33",
        function="The BCL2 protein is an apoptosis regulator that inhibits programmed cell death",
    )
    assert answer.verify() is True


@pytest.mark.unit
def test_with_correct_dict_template_with_optional_fields(fixtures_dir: Path) -> None:
    """Test that with_correct_dict.py handles optional fields correctly."""
    template_path = fixtures_dir / "templates" / "with_correct_dict.py"
    Answer = _load_template(template_path)

    # With optional fields populated
    with_optional = Answer(
        gene_name="BCL2",
        chromosome="18q21.33",
        function="BCL2 is an apoptosis regulator that inhibits programmed cell death",
        synonyms=["Bcl-2", "BCL2"],
        omim_id=151430,
    )
    assert with_optional.verify() is True

    # Wrong omim_id should fail
    wrong_omim = Answer(
        gene_name="BCL2",
        chromosome="18q21.33",
        function="BCL2 is an apoptosis regulator that inhibits programmed cell death",
        synonyms=["Bcl-2", "BCL2"],
        omim_id=999999,
    )
    assert wrong_omim.verify() is False


@pytest.mark.unit
def test_all_templates_have_source_code(fixtures_dir: Path) -> None:
    """Test that all templates have source code available."""
    template_files = [
        "simple_extraction.py",
        "multi_field.py",
        "with_correct_dict.py",
    ]

    for template_file in template_files:
        template_path = fixtures_dir / "templates" / template_file
        Answer = _load_template(template_path)

        # Source code should be available (even if via inspect.getsource fails)
        # For exec-created or file-loaded classes, this might be None
        # But for our file-based templates, it should work
        source = Answer.get_source_code()
        assert source is not None, f"{template_file} should have source code"
        assert "class Answer" in source, f"{template_file} source should contain Answer class definition"
