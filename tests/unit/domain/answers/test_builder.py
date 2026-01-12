"""Unit tests for AnswerBuilder class.

Tests the fluent interface for building Answer templates programmatically.
"""

import pytest

from karenina.domain.answers.builder import AnswerBuilder


class TestAnswerBuilderInit:
    """Tests for AnswerBuilder initialization."""

    def test_init_creates_empty_builder(self) -> None:
        """Test that new builder has empty attributes and patterns."""
        builder = AnswerBuilder()
        assert builder.attributes == []
        assert builder.field_descriptions == {}
        assert builder.regex_patterns == {}
        assert builder.regex_descriptions == {}


class TestAddAttribute:
    """Tests for add_attribute method."""

    def test_add_attribute_string(self) -> None:
        """Test adding a string attribute."""
        builder = AnswerBuilder()
        result = builder.add_attribute("name", "str", "The name", "Alice")
        assert result is builder  # Method chaining
        assert len(builder.attributes) == 1
        assert builder.attributes[0].name == "name"
        assert builder.attributes[0].type == "str"
        assert builder.attributes[0].ground_truth == "Alice"
        assert builder.field_descriptions["name"] == "The name"

    def test_add_attribute_int(self) -> None:
        """Test adding an integer attribute."""
        builder = AnswerBuilder()
        builder.add_attribute("count", "int", "The count", 42)
        assert builder.attributes[0].ground_truth == 42

    def test_add_attribute_bool(self) -> None:
        """Test adding a boolean attribute."""
        builder = AnswerBuilder()
        builder.add_attribute("is_valid", "bool", "Validity flag", True)
        assert builder.attributes[0].ground_truth is True

    def test_add_attribute_list(self) -> None:
        """Test adding a list attribute."""
        builder = AnswerBuilder()
        builder.add_attribute("items", "List[str]", "List of items", ["a", "b", "c"])
        assert builder.attributes[0].ground_truth == ["a", "b", "c"]

    def test_add_attribute_invalid_name_raises(self) -> None:
        """Test that invalid Python identifier raises ValueError."""
        builder = AnswerBuilder()
        with pytest.raises(ValueError, match="Invalid field name"):
            builder.add_attribute("123invalid", "str", "Description", "value")

    def test_add_attribute_duplicate_name_raises(self) -> None:
        """Test that duplicate attribute name raises ValueError."""
        builder = AnswerBuilder()
        builder.add_attribute("name", "str", "First", "value1")
        with pytest.raises(ValueError, match="already exists"):
            builder.add_attribute("name", "str", "Second", "value2")

    def test_add_attribute_name_conflict_with_regex_raises(self) -> None:
        """Test that name conflict with existing regex pattern raises ValueError."""
        builder = AnswerBuilder()
        builder.add_regex("pattern1", r"\d+", 3)
        with pytest.raises(ValueError, match="already used for regex pattern"):
            builder.add_attribute("pattern1", "str", "Description", "value")

    def test_add_multiple_attributes(self) -> None:
        """Test adding multiple attributes."""
        builder = (
            AnswerBuilder()
            .add_attribute("name", "str", "Name", "Bob")
            .add_attribute("age", "int", "Age", 30)
            .add_attribute("active", "bool", "Active", True)
        )
        assert len(builder.attributes) == 3
        assert {attr.name for attr in builder.attributes} == {"name", "age", "active"}


class TestRemoveAttribute:
    """Tests for remove_attribute method."""

    def test_remove_attribute(self) -> None:
        """Test removing an existing attribute."""
        builder = AnswerBuilder()
        builder.add_attribute("name", "str", "Name", "Alice")
        builder.add_attribute("age", "int", "Age", 30)
        result = builder.remove_attribute("name")
        assert result is builder  # Method chaining
        assert len(builder.attributes) == 1
        assert builder.attributes[0].name == "age"
        assert "name" not in builder.field_descriptions

    def test_remove_nonexistent_attribute_raises(self) -> None:
        """Test that removing non-existent attribute raises ValueError."""
        builder = AnswerBuilder()
        builder.add_attribute("name", "str", "Name", "Alice")
        with pytest.raises(ValueError, match="not found"):
            builder.remove_attribute("nonexistent")


class TestAddRegex:
    """Tests for add_regex method."""

    def test_add_regex_exact(self) -> None:
        """Test adding a regex pattern with exact match."""
        builder = AnswerBuilder()
        result = builder.add_regex("citation", r"\[\d+\]", 1, match_type="exact")
        assert result is builder  # Method chaining
        assert "citation" in builder.regex_patterns
        assert builder.regex_patterns["citation"]["pattern"] == r"\[\d+\]"
        assert builder.regex_patterns["citation"]["expected"] == 1
        assert builder.regex_patterns["citation"]["match_type"] == "exact"

    def test_add_regex_contains(self) -> None:
        """Test adding a regex pattern with contains match."""
        builder = AnswerBuilder()
        builder.add_regex("has_keyword", r"\bimportant\b", True, match_type="contains")
        assert builder.regex_patterns["has_keyword"]["match_type"] == "contains"

    def test_add_regex_count(self) -> None:
        """Test adding a regex pattern with count match."""
        builder = AnswerBuilder()
        builder.add_regex("citations", r"\[\d+\]", 3, match_type="count")
        assert builder.regex_patterns["citations"]["match_type"] == "count"

    def test_add_regex_all(self) -> None:
        """Test adding a regex pattern with all match."""
        builder = AnswerBuilder()
        builder.add_regex("tokens", r"\w+", ["a", "b"], match_type="all")
        assert builder.regex_patterns["tokens"]["match_type"] == "all"

    def test_add_regex_with_description(self) -> None:
        """Test adding regex with description."""
        builder = AnswerBuilder()
        builder.add_regex("pattern1", r"\d+", 5, description="Count of digits")
        assert builder.regex_descriptions["pattern1"] == "Count of digits"

    def test_add_regex_invalid_name_raises(self) -> None:
        """Test that invalid pattern name raises ValueError."""
        builder = AnswerBuilder()
        with pytest.raises(ValueError, match="Invalid pattern name"):
            builder.add_regex("123invalid", r"\d+", 1)

    def test_add_regex_duplicate_name_raises(self) -> None:
        """Test that duplicate regex name raises ValueError."""
        builder = AnswerBuilder()
        builder.add_regex("pattern1", r"\d+", 1)
        with pytest.raises(ValueError, match="already exists"):
            builder.add_regex("pattern1", r"\w+", 2)

    def test_add_regex_name_conflict_with_attribute_raises(self) -> None:
        """Test that name conflict with existing attribute raises ValueError."""
        builder = AnswerBuilder()
        builder.add_attribute("field1", "str", "Description", "value")
        with pytest.raises(ValueError, match="already used for attribute"):
            builder.add_regex("field1", r"\d+", 1)

    def test_add_regex_invalid_pattern_raises(self) -> None:
        """Test that invalid regex pattern raises ValueError."""
        builder = AnswerBuilder()
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            builder.add_regex("bad", r"[unclosed", 1)

    def test_add_regex_invalid_match_type_raises(self) -> None:
        """Test that invalid match_type raises ValueError."""
        builder = AnswerBuilder()
        with pytest.raises(ValueError, match="Invalid match_type"):
            builder.add_regex("pattern", r"\d+", 1, match_type="invalid")


class TestRemoveRegex:
    """Tests for remove_regex method."""

    def test_remove_regex(self) -> None:
        """Test removing an existing regex pattern."""
        builder = AnswerBuilder()
        builder.add_regex("pattern1", r"\d+", 1, description="First")
        builder.add_regex("pattern2", r"\w+", 2, description="Second")
        result = builder.remove_regex("pattern1")
        assert result is builder  # Method chaining
        assert "pattern1" not in builder.regex_patterns
        assert "pattern2" in builder.regex_patterns
        assert "pattern1" not in builder.regex_descriptions

    def test_remove_nonexistent_regex_raises(self) -> None:
        """Test that removing non-existent regex raises ValueError."""
        builder = AnswerBuilder()
        with pytest.raises(ValueError, match="not found"):
            builder.remove_regex("nonexistent")


class TestCompile:
    """Tests for compile method."""

    def test_compile_single_attribute(self) -> None:
        """Test compiling a builder with a single attribute."""
        builder = AnswerBuilder()
        builder.add_attribute("value", "str", "The value", "42")
        Answer = builder.compile()
        # Create an instance
        answer = Answer(value="42")
        assert answer.value == "42"

    def test_compile_multiple_attributes(self) -> None:
        """Test compiling a builder with multiple attributes."""
        builder = AnswerBuilder()
        builder.add_attribute("name", "str", "Name", "Alice")
        builder.add_attribute("age", "int", "Age", 30)
        builder.add_attribute("active", "bool", "Active", True)
        Answer = builder.compile()
        answer = Answer(name="Alice", age=30, active=True)
        assert answer.name == "Alice"
        assert answer.age == 30
        assert answer.active is True

    def test_compile_with_regex_only(self) -> None:
        """Test compiling a builder with only regex patterns."""
        builder = AnswerBuilder()
        builder.add_regex("citations", r"\[\d+\]", 2, match_type="count")
        Answer = builder.compile()
        answer = Answer()
        assert "citations" in answer.regex
        assert answer.regex["citations"]["pattern"] == r"\[\d+\]"

    def test_compile_with_attributes_and_regex(self) -> None:
        """Test compiling a builder with both attributes and regex."""
        builder = AnswerBuilder()
        builder.add_attribute("text", "str", "Answer text", "The mitochondria")
        builder.add_regex("citations", r"\[\d+\]", 1, match_type="count")
        Answer = builder.compile()
        answer = Answer(text="The mitochondria")
        assert answer.text == "The mitochondria"
        assert "citations" in answer.regex

    def test_compile_empty_builder_raises(self) -> None:
        """Test that compiling empty builder raises ValueError."""
        builder = AnswerBuilder()
        with pytest.raises(ValueError, match="no attributes or regex patterns"):
            builder.compile()

    def test_compile_custom_class_name(self) -> None:
        """Test compiling with a custom class name."""
        builder = AnswerBuilder()
        builder.add_attribute("value", "str", "Value", "test")
        CustomAnswer = builder.compile(class_name="CustomAnswer")
        # Check the class name
        assert CustomAnswer.__name__ == "CustomAnswer"

    def test_compiled_class_has_source_code(self) -> None:
        """Test that compiled class has _source_code attribute."""
        builder = AnswerBuilder()
        builder.add_attribute("value", "str", "Value", "test")
        Answer = builder.compile()
        assert hasattr(Answer, "_source_code")
        assert "class Answer(BaseAnswer):" in Answer._source_code

    def test_compile_regex_exact_match_type(self) -> None:
        """Test that exact match type is preserved in compiled class."""
        builder = AnswerBuilder()
        builder.add_regex("pattern", r"\d+", 42, match_type="exact")
        Answer = builder.compile()
        answer = Answer()
        assert answer.regex["pattern"]["match_type"] == "exact"

    def test_compile_regex_contains_match_type(self) -> None:
        """Test that contains match type is preserved in compiled class."""
        builder = AnswerBuilder()
        builder.add_regex("pattern", r"keyword", True, match_type="contains")
        Answer = builder.compile()
        answer = Answer()
        assert answer.regex["pattern"]["match_type"] == "contains"

    def test_compile_regex_count_match_type(self) -> None:
        """Test that count match type is preserved in compiled class."""
        builder = AnswerBuilder()
        builder.add_regex("pattern", r"\d+", 3, match_type="count")
        Answer = builder.compile()
        answer = Answer()
        assert answer.regex["pattern"]["match_type"] == "count"

    def test_compile_regex_all_match_type(self) -> None:
        """Test that all match type is preserved in compiled class."""
        builder = AnswerBuilder()
        builder.add_regex("pattern", r"\w+", ["a", "b"], match_type="all")
        Answer = builder.compile()
        answer = Answer()
        assert answer.regex["pattern"]["match_type"] == "all"

    def test_compile_preserves_field_descriptions(self) -> None:
        """Test that field descriptions are preserved in compiled class."""
        builder = AnswerBuilder()
        builder.add_attribute("value", "str", "The extracted value", "test")
        Answer = builder.compile()
        # The description should be in the Field
        field_info = Answer.model_fields["value"]
        assert field_info.description == "The extracted value"

    def test_compile_with_list_type(self) -> None:
        """Test compiling with a list type attribute."""
        builder = AnswerBuilder()
        builder.add_attribute("items", "List[str]", "List items", ["a", "b"])
        Answer = builder.compile()
        answer = Answer(items=["a", "b"])
        assert answer.items == ["a", "b"]

    def test_compile_verify_returns_true_for_correct_value(self) -> None:
        """Test that verify returns true for correct ground truth."""
        builder = AnswerBuilder()
        builder.add_attribute("value", "str", "Value", "correct")
        Answer = builder.compile()
        answer = Answer(value="correct")
        assert answer.verify() is True

    def test_compile_verify_returns_false_for_incorrect_value(self) -> None:
        """Test that verify returns false for incorrect value."""
        builder = AnswerBuilder()
        builder.add_attribute("value", "str", "Value", "correct")
        Answer = builder.compile()
        answer = Answer(value="incorrect")
        assert answer.verify() is False

    def test_compile_chaining_remove_then_add(self) -> None:
        """Test method chaining through remove then add."""
        builder = AnswerBuilder()
        builder.add_attribute("temp", "str", "Temp", "value")
        builder.remove_attribute("temp")
        builder.add_attribute("permanent", "str", "Permanent", "final")
        Answer = builder.compile()
        answer = Answer(permanent="final")
        assert answer.permanent == "final"


class TestRepr:
    """Tests for __repr__ method."""

    def test_repr_empty_builder(self) -> None:
        """Test repr of empty builder."""
        builder = AnswerBuilder()
        result = repr(builder)
        assert "AnswerBuilder:" in result
        assert "Classical Attributes (0):" in result
        assert "(none)" in result
        assert "Regex Patterns (0):" in result

    def test_repr_with_attributes(self) -> None:
        """Test repr with attributes."""
        builder = AnswerBuilder()
        builder.add_attribute("name", "str", "The name", "Alice")
        builder.add_attribute("age", "int", "The age", 30)
        result = repr(builder)
        assert "Classical Attributes (2):" in result
        assert "- name: str = 'Alice'" in result
        assert "- age: int = 30" in result
        assert "The name" in result
        assert "The age" in result

    def test_repr_with_regex_patterns(self) -> None:
        """Test repr with regex patterns."""
        builder = AnswerBuilder()
        builder.add_regex("citations", r"\[\d+\]", 3, match_type="count", description="Citation count")
        result = repr(builder)
        assert "Regex Patterns (1):" in result
        assert "- citations:" in result
        assert "count" in result
        assert "Citation count" in result

    def test_repr_with_both(self) -> None:
        """Test repr with both attributes and regex patterns."""
        builder = AnswerBuilder()
        builder.add_attribute("text", "str", "Text", "answer")
        builder.add_regex("pattern", r"\d+", 1, match_type="exact")
        result = repr(builder)
        assert "Classical Attributes (1):" in result
        assert "Regex Patterns (1):" in result
