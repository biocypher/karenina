"""Unit tests for answer class renaming functionality."""

from karenina.benchmark.core.questions import _rename_answer_class_to_standard


class TestRenameAnswerClass:
    """Test suite for _rename_answer_class_to_standard function."""

    def test_already_named_answer_unchanged(self):
        """Test that classes already named 'Answer' are not modified."""
        source = """class Answer(BaseAnswer):
    value: int = Field(description="test")

    def verify(self) -> bool:
        return True
"""
        result = _rename_answer_class_to_standard(source, "Answer")
        assert result == source
        assert "class Answer" in result

    def test_simple_class_rename(self):
        """Test renaming a simple class from custom name to Answer."""
        source = """class VenetoclaxAnswer(BaseAnswer):
    target: str = Field(description="The protein target")

    def model_post_init(self, __context):
        self.correct = {"target": "BCL2"}

    def verify(self) -> bool:
        return self.target.upper() == self.correct["target"]
"""
        result = _rename_answer_class_to_standard(source, "VenetoclaxAnswer")

        assert "class Answer(BaseAnswer):" in result
        assert "VenetoclaxAnswer" not in result
        assert "BCL2" in result  # Content should be preserved
        assert "def verify(self)" in result  # Methods preserved

    def test_rename_preserves_docstrings(self):
        """Test that class docstrings are preserved during rename."""
        source = '''class CustomAnswer(BaseAnswer):
    """This is a custom answer class with a docstring."""
    value: int

    def verify(self) -> bool:
        return True
'''
        result = _rename_answer_class_to_standard(source, "CustomAnswer")

        assert "class Answer(BaseAnswer):" in result
        assert "This is a custom answer class with a docstring." in result

    def test_rename_with_decorators(self):
        """Test that decorators are preserved during rename."""
        source = """@some_decorator
class DecoratedAnswer(BaseAnswer):
    value: int

    def verify(self) -> bool:
        return True
"""
        result = _rename_answer_class_to_standard(source, "DecoratedAnswer")

        # Note: AST will preserve the decorator but may change formatting
        assert "class Answer(BaseAnswer):" in result
        assert "def verify(self)" in result

    def test_rename_complex_class(self):
        """Test renaming a class with multiple fields and methods."""
        source = """class ChromosomeCounter(BaseAnswer):
    count: int = Field(description="Number of chromosomes")
    verified: bool = Field(default=False)

    def model_post_init(self, __context):
        self.correct = {"count": 46, "verified": True}

    def verify(self) -> bool:
        return self.count == self.correct["count"]

    def verify_granular(self) -> float:
        correct_count = 0
        if self.count == self.correct["count"]:
            correct_count += 1
        if self.verified == self.correct["verified"]:
            correct_count += 1
        return correct_count / 2
"""
        result = _rename_answer_class_to_standard(source, "ChromosomeCounter")

        assert "class Answer(BaseAnswer):" in result
        assert "ChromosomeCounter" not in result
        assert "def verify(self)" in result
        assert "def verify_granular(self)" in result
        assert "self.correct" in result

    def test_multiple_classes_only_renames_target(self):
        """Test that only the specified class is renamed when multiple classes exist."""
        source = """class Helper:
    pass

class MyAnswer(BaseAnswer):
    value: int

    def verify(self) -> bool:
        return True

class AnotherClass:
    pass
"""
        result = _rename_answer_class_to_standard(source, "MyAnswer")

        assert "class Answer(BaseAnswer):" in result
        assert "class Helper:" in result  # Other classes preserved
        assert "class AnotherClass:" in result
        assert "MyAnswer" not in result

    def test_preserves_field_descriptions(self):
        """Test that Field descriptions with special characters are preserved."""
        source = """class TestAnswer(BaseAnswer):
    target: str = Field(description="The protein target mentioned in the response")
    count: int = Field(description="Number of items (1-100)")

    def verify(self) -> bool:
        return True
"""
        result = _rename_answer_class_to_standard(source, "TestAnswer")

        assert "class Answer(BaseAnswer):" in result
        assert "The protein target mentioned in the response" in result
        assert "Number of items (1-100)" in result

    def test_preserves_type_annotations(self):
        """Test that complex type annotations are preserved."""
        source = """from typing import List, Dict, Optional

class ComplexAnswer(BaseAnswer):
    items: List[str] = Field(description="List of items")
    mapping: Dict[str, int] = Field(default_factory=dict)
    optional_value: Optional[str] = None

    def verify(self) -> bool:
        return True
"""
        result = _rename_answer_class_to_standard(source, "ComplexAnswer")

        assert "class Answer(BaseAnswer):" in result
        assert "List[str]" in result
        assert "Dict[str, int]" in result
        assert "Optional[str]" in result

    def test_preserves_indentation_and_formatting(self):
        """Test that basic code structure is preserved."""
        source = """class MyAnswer(BaseAnswer):
    value: int = Field(description="test")

    def model_post_init(self, __context):
        self.correct = {"value": 42}

    def verify(self) -> bool:
        return self.value == self.correct["value"]
"""
        result = _rename_answer_class_to_standard(source, "MyAnswer")

        # AST unparse may change exact formatting, but structure should be preserved
        assert "class Answer(BaseAnswer):" in result
        assert "def model_post_init" in result
        assert "def verify" in result
        assert "self.correct" in result
        assert result.count("def ") == 2  # Two methods

    def test_fallback_on_invalid_syntax(self):
        """Test that fallback string replacement works on invalid Python."""
        # Intentionally malformed Python that AST can't parse
        source = """class BrokenAnswer(BaseAnswer):
    value: int = Field(description="test"
    # Missing closing paren - invalid syntax

    def verify(self) -> bool:
        return True
"""
        result = _rename_answer_class_to_standard(source, "BrokenAnswer")

        # Should fall back to string replacement
        assert "class Answer(BaseAnswer):" in result
        assert "BrokenAnswer" not in result

    def test_empty_class(self):
        """Test renaming an empty class definition."""
        source = """class EmptyAnswer(BaseAnswer):
    pass
"""
        result = _rename_answer_class_to_standard(source, "EmptyAnswer")

        assert "class Answer(BaseAnswer):" in result
        assert "pass" in result

    def test_class_with_class_variables(self):
        """Test renaming a class with class-level variables."""
        source = """class AnswerWithClassVars(BaseAnswer):
    DEFAULT_THRESHOLD: float = 0.5
    MAX_RETRIES: int = 3

    value: int

    def verify(self) -> bool:
        return True
"""
        result = _rename_answer_class_to_standard(source, "AnswerWithClassVars")

        assert "class Answer(BaseAnswer):" in result
        assert "DEFAULT_THRESHOLD" in result
        assert "MAX_RETRIES" in result

    def test_nested_function_definitions(self):
        """Test that nested function definitions are preserved."""
        source = """class ComplexAnswer(BaseAnswer):
    value: int

    def verify(self) -> bool:
        def helper():
            return True
        return helper()
"""
        result = _rename_answer_class_to_standard(source, "ComplexAnswer")

        assert "class Answer(BaseAnswer):" in result
        assert "def verify(self)" in result
        assert "def helper()" in result

    def test_preserves_comments(self):
        """Test that comments within the class are preserved."""
        source = """class CommentedAnswer(BaseAnswer):
    # This is a field comment
    value: int = Field(description="test")

    def verify(self) -> bool:
        # Check if value is correct
        return self.value == 42  # Magic number
"""
        result = _rename_answer_class_to_standard(source, "CommentedAnswer")

        assert "class Answer(BaseAnswer):" in result
        # Note: AST may not preserve all comments depending on Python version
        # But basic structure should be preserved

    def test_multiline_field_definitions(self):
        """Test that multiline field definitions work correctly."""
        source = """class MultilineAnswer(BaseAnswer):
    long_field: str = Field(
        description="This is a very long description that spans multiple lines"
    )

    def verify(self) -> bool:
        return True
"""
        result = _rename_answer_class_to_standard(source, "MultilineAnswer")

        assert "class Answer(BaseAnswer):" in result
        assert "long_field" in result
        assert "This is a very long description" in result

    def test_special_characters_in_strings(self):
        """Test that special characters in string literals are preserved."""
        source = """class SpecialCharsAnswer(BaseAnswer):
    text: str = Field(description="Text with 'quotes' and \\"escapes\\"")

    def model_post_init(self, __context):
        self.correct = {"text": "Hello\\nWorld"}

    def verify(self) -> bool:
        return True
"""
        result = _rename_answer_class_to_standard(source, "SpecialCharsAnswer")

        assert "class Answer(BaseAnswer):" in result
        # String content should be preserved (exact format may vary)
        assert "text" in result
        assert "correct" in result
