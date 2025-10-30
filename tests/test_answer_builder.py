"""Tests for the AnswerBuilder class."""

import pytest

from karenina.answers.builder import AnswerBuilder
from karenina.benchmark import Benchmark
from karenina.schemas.domain import BaseAnswer


class TestAnswerBuilder:
    """Test suite for AnswerBuilder functionality."""

    def test_empty_builder_initialization(self) -> None:
        """Test that empty AnswerBuilder initializes correctly."""
        builder = AnswerBuilder()

        assert builder.attributes == []
        assert builder.field_descriptions == {}
        assert builder.regex_patterns == {}
        assert builder.regex_descriptions == {}

    def test_add_attribute_basic(self) -> None:
        """Test adding a basic attribute."""
        builder = AnswerBuilder()
        result = builder.add_attribute("test_field", "bool", "Test boolean field", True)

        # Should return self for chaining
        assert result is builder

        # Should add the attribute
        assert len(builder.attributes) == 1
        assert builder.attributes[0].name == "test_field"
        assert builder.attributes[0].type == "bool"
        assert builder.attributes[0].ground_truth is True
        assert builder.field_descriptions["test_field"] == "Test boolean field"

    def test_add_attribute_chaining(self) -> None:
        """Test method chaining when adding multiple attributes."""
        builder = AnswerBuilder()

        result = (
            builder.add_attribute("field1", "bool", "First field", True)
            .add_attribute("field2", "int", "Second field", 42)
            .add_attribute("field3", "str", "Third field", "test")
        )

        assert result is builder
        assert len(builder.attributes) == 3

        # Verify all attributes are added correctly
        names = [attr.name for attr in builder.attributes]
        assert names == ["field1", "field2", "field3"]

    def test_add_attribute_validation_errors(self) -> None:
        """Test validation errors when adding attributes."""
        builder = AnswerBuilder()

        # Invalid Python identifier
        with pytest.raises(ValueError, match="Invalid field name"):
            builder.add_attribute("123invalid", "bool", "Invalid name", True)

        with pytest.raises(ValueError, match="Invalid field name"):
            builder.add_attribute("invalid-name", "bool", "Invalid name", True)

        # Duplicate attribute name
        builder.add_attribute("duplicate", "bool", "First", True)
        with pytest.raises(ValueError, match="Attribute 'duplicate' already exists"):
            builder.add_attribute("duplicate", "int", "Second", 42)

    def test_remove_attribute_basic(self) -> None:
        """Test removing an attribute."""
        builder = AnswerBuilder()
        builder.add_attribute("field1", "bool", "First field", True)
        builder.add_attribute("field2", "int", "Second field", 42)

        result = builder.remove_attribute("field1")

        # Should return self for chaining
        assert result is builder

        # Should remove the attribute and description
        assert len(builder.attributes) == 1
        assert builder.attributes[0].name == "field2"
        assert "field1" not in builder.field_descriptions
        assert "field2" in builder.field_descriptions

    def test_remove_attribute_not_found(self) -> None:
        """Test error when removing non-existent attribute."""
        builder = AnswerBuilder()

        with pytest.raises(ValueError, match="Attribute 'nonexistent' not found"):
            builder.remove_attribute("nonexistent")

    def test_add_regex_basic(self) -> None:
        """Test adding a basic regex pattern."""
        builder = AnswerBuilder()
        result = builder.add_regex(
            "citations", r"\[\d+\]", expected=3, match_type="count", description="Citation count"
        )

        # Should return self for chaining
        assert result is builder

        # Should add the regex pattern
        assert len(builder.regex_patterns) == 1
        assert builder.regex_patterns["citations"]["pattern"] == r"\[\d+\]"
        assert builder.regex_patterns["citations"]["expected"] == 3
        assert builder.regex_patterns["citations"]["match_type"] == "count"
        assert builder.regex_descriptions["citations"] == "Citation count"

    def test_add_regex_default_parameters(self) -> None:
        """Test adding regex with default parameters."""
        builder = AnswerBuilder()
        builder.add_regex("test_pattern", r"test\d+", expected="test123")

        # Should use default match_type
        assert builder.regex_patterns["test_pattern"]["match_type"] == "exact"
        # Should have empty description
        assert "test_pattern" not in builder.regex_descriptions

    def test_add_regex_validation_errors(self) -> None:
        """Test validation errors when adding regex patterns."""
        builder = AnswerBuilder()

        # Invalid Python identifier
        with pytest.raises(ValueError, match="Invalid pattern name"):
            builder.add_regex("123invalid", r"test", "expected")

        # Invalid regex pattern
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            builder.add_regex("invalid_pattern", r"[unclosed", "expected")

        # Invalid match_type
        with pytest.raises(ValueError, match="Invalid match_type"):
            builder.add_regex("test", r"test", "expected", match_type="invalid")

        # Duplicate regex name
        builder.add_regex("duplicate", r"test", "expected")
        with pytest.raises(ValueError, match="Regex pattern 'duplicate' already exists"):
            builder.add_regex("duplicate", r"other", "expected")

    def test_name_conflicts_between_attributes_and_regex(self) -> None:
        """Test that attribute and regex names cannot conflict."""
        builder = AnswerBuilder()

        # Add attribute first, then try regex with same name
        builder.add_attribute("conflict", "bool", "Test field", True)
        with pytest.raises(ValueError, match="Name 'conflict' already used for attribute"):
            builder.add_regex("conflict", r"test", "expected")

        # Add regex first, then try attribute with same name
        builder2 = AnswerBuilder()
        builder2.add_regex("conflict2", r"test", "expected")
        with pytest.raises(ValueError, match="Name 'conflict2' already used for regex pattern"):
            builder2.add_attribute("conflict2", "bool", "Test field", True)

    def test_remove_regex_basic(self) -> None:
        """Test removing a regex pattern."""
        builder = AnswerBuilder()
        builder.add_regex("pattern1", r"test1", "expected1", description="First pattern")
        builder.add_regex("pattern2", r"test2", "expected2", description="Second pattern")

        result = builder.remove_regex("pattern1")

        # Should return self for chaining
        assert result is builder

        # Should remove the pattern and description
        assert len(builder.regex_patterns) == 1
        assert "pattern1" not in builder.regex_patterns
        assert "pattern2" in builder.regex_patterns
        assert "pattern1" not in builder.regex_descriptions
        assert "pattern2" in builder.regex_descriptions

    def test_remove_regex_not_found(self) -> None:
        """Test error when removing non-existent regex pattern."""
        builder = AnswerBuilder()

        with pytest.raises(ValueError, match="Regex pattern 'nonexistent' not found"):
            builder.remove_regex("nonexistent")

    def test_compile_with_attributes_only(self) -> None:
        """Test compiling builder with only classical attributes."""
        builder = AnswerBuilder()
        builder.add_attribute("test_bool", "bool", "Test boolean", True)
        builder.add_attribute("test_int", "int", "Test integer", 42)

        Answer = builder.compile()

        # Should be a subclass of BaseAnswer
        assert issubclass(Answer, BaseAnswer)
        assert Answer.__name__ == "Answer"

        # Should have source code stored
        assert hasattr(Answer, "_source_code")
        assert Answer._source_code is not None
        assert "class Answer(BaseAnswer):" in Answer._source_code

        # Should be able to instantiate and verify
        answer = Answer(test_bool=True, test_int=42)
        assert answer.verify() is True

        # Wrong values should fail verification
        answer_wrong = Answer(test_bool=False, test_int=42)
        assert answer_wrong.verify() is False

    def test_compile_with_regex_only(self) -> None:
        """Test compiling builder with only regex patterns."""
        builder = AnswerBuilder()
        builder.add_regex("test_pattern", r"test\d+", expected="test123", match_type="exact")

        Answer = builder.compile()

        # Should be a subclass of BaseAnswer
        assert issubclass(Answer, BaseAnswer)

        # Should have regex field
        answer = Answer()
        assert hasattr(answer, "regex")
        assert isinstance(answer.regex, dict)
        assert "test_pattern" in answer.regex

        # Should have source code stored
        assert Answer._source_code is not None
        assert "regex: dict = Field" in Answer._source_code

        # Should be able to verify regex
        test_text = "This has test123 in it"
        regex_result = answer.verify_regex(test_text)
        assert regex_result["success"] is True
        assert regex_result["results"]["test_pattern"] is True

    def test_compile_with_both_attributes_and_regex(self) -> None:
        """Test compiling builder with both attributes and regex patterns."""
        builder = AnswerBuilder()
        builder.add_attribute("mentions_drug", "bool", "Whether drug is mentioned", True)
        builder.add_attribute("dosage", "int", "Dosage amount", 500)
        builder.add_regex("citations", r"\[\d+\]", expected=2, match_type="count")

        Answer = builder.compile()

        # Should have both classical and regex functionality
        answer = Answer(mentions_drug=True, dosage=500)
        assert answer.verify() is True

        # Should have regex field
        assert hasattr(answer, "regex")
        assert "citations" in answer.regex

        # Test regex verification
        test_text = "Drug X at 500mg [1][2]"
        regex_result = answer.verify_regex(test_text)
        assert regex_result["success"] is True

        # Test with wrong citation count
        wrong_text = "Drug X at 500mg [1]"
        regex_result_wrong = answer.verify_regex(wrong_text)
        assert regex_result_wrong["success"] is False

    def test_compile_custom_class_name(self) -> None:
        """Test compiling with custom class name."""
        builder = AnswerBuilder()
        builder.add_attribute("value", "int", "Test value", 123)

        CustomAnswer = builder.compile(class_name="CustomAnswer")

        assert CustomAnswer.__name__ == "CustomAnswer"
        assert "class CustomAnswer(BaseAnswer):" in CustomAnswer._source_code

    def test_compile_empty_builder_error(self) -> None:
        """Test that empty builder cannot be compiled."""
        builder = AnswerBuilder()

        with pytest.raises(ValueError, match="Cannot compile empty AnswerBuilder"):
            builder.compile()

    def test_repr_empty_builder(self) -> None:
        """Test string representation of empty builder."""
        builder = AnswerBuilder()
        repr_str = repr(builder)

        assert "AnswerBuilder:" in repr_str
        assert "Classical Attributes (0):" in repr_str
        assert "Regex Patterns (0):" in repr_str
        assert "(none)" in repr_str

    def test_repr_with_attributes_and_regex(self) -> None:
        """Test string representation with attributes and regex."""
        builder = AnswerBuilder()
        builder.add_attribute("test_bool", "bool", "Test boolean field", True)
        builder.add_attribute("test_int", "int", "Test integer field", 42)
        builder.add_regex("pattern1", r"test\\d+", expected="test123", description="Test pattern")
        builder.add_regex("pattern2", r"\\[\\d+\\]", expected=2, match_type="count")

        repr_str = repr(builder)

        # Should show counts
        assert "Classical Attributes (2):" in repr_str
        assert "Regex Patterns (2):" in repr_str

        # Should show attribute details
        assert "test_bool: bool = True" in repr_str
        assert "Test boolean field" in repr_str
        assert "test_int: int = 42" in repr_str

        # Should show regex details
        assert "pattern1: test\\\\d+ (exact, expected='test123')" in repr_str
        assert "Test pattern" in repr_str
        assert "pattern2: \\\\[\\\\d+\\\\] (count, expected=2)" in repr_str

    def test_different_match_types(self) -> None:
        """Test all supported regex match types."""
        builder = AnswerBuilder()

        # Test all match types
        builder.add_regex("exact_pattern", r"exact", "exact", match_type="exact")
        builder.add_regex("contains_pattern", r"\w+", "text", match_type="contains")  # "text" is in the matches
        builder.add_regex("count_pattern", r"x", 2, match_type="count")  # "x" appears in "exact" and "text"
        builder.add_regex("all_pattern", r"\w+", ["word1", "word2"], match_type="all")

        Answer = builder.compile()
        answer = Answer()

        # Test text that should match all patterns
        test_text = "exact text with word1 and word2 and 123"
        result = answer.verify_regex(test_text)

        assert result["success"] is True
        assert result["results"]["exact_pattern"] is True
        assert result["results"]["contains_pattern"] is True
        assert result["results"]["count_pattern"] is True
        assert result["results"]["all_pattern"] is True

    def test_complex_types_and_values(self) -> None:
        """Test builder with complex types and values."""
        builder = AnswerBuilder()

        # Test various complex types
        builder.add_attribute("simple_list", "List[str]", "List of strings", ["a", "b", "c"])
        builder.add_attribute("literal_type", "Literal['high', 'medium', 'low']", "Severity level", "high")
        builder.add_attribute("float_value", "float", "Floating point number", 3.14159)
        builder.add_attribute("dict_value", "Dict[str, int]", "String to int mapping", {"key": 123})

        Answer = builder.compile()

        # Test instantiation with complex values
        answer = Answer(simple_list=["a", "b", "c"], literal_type="high", float_value=3.14159, dict_value={"key": 123})

        assert answer.verify() is True

        # Test with wrong values
        answer_wrong = Answer(
            simple_list=["x", "y", "z"],  # Wrong list
            literal_type="high",
            float_value=3.14159,
            dict_value={"key": 123},
        )

        assert answer_wrong.verify() is False


class TestAnswerBuilderIntegration:
    """Test AnswerBuilder integration with QuestionManager."""

    def test_integration_with_question_manager(self) -> None:
        """Test that compiled Answer classes work with QuestionManager."""
        # Create a benchmark
        benchmark = Benchmark("Test Benchmark")

        # Build Answer template
        builder = AnswerBuilder()
        builder.add_attribute("mentions_treatment", "bool", "Mentions treatment", True)
        builder.add_attribute("patient_count", "int", "Number of patients", 150)
        builder.add_regex("pmid_citations", r"PMID:\s*(\d+)", expected=2, match_type="count")

        Answer = builder.compile()

        # Add question to benchmark
        question_id = benchmark.add_question(
            question="How many patients were treated in the study?",
            raw_answer="150 patients were treated with the new therapy. PMID: 12345 and PMID: 67890",
            answer_template=Answer,
        )

        # Verify question was added successfully
        assert question_id in benchmark
        question_data = benchmark.get_question(question_id)

        # Verify the Answer class source code was stored
        template_code = question_data["answer_template"]
        assert "class Answer(BaseAnswer):" in template_code
        assert "mentions_treatment: bool" in template_code
        assert "regex: dict = Field" in template_code

    def test_builder_fluent_interface_realistic_example(self) -> None:
        """Test realistic usage with fluent interface."""
        # Realistic medical research question example
        Answer = (
            AnswerBuilder()
            .add_attribute("mentions_study_drug", "bool", "Whether response mentions the study drug", True)
            .add_attribute("mentions_control_group", "bool", "Whether response mentions control group", True)
            .add_attribute("sample_size", "int", "Total sample size reported", 245)
            .add_attribute(
                "efficacy_category", "Literal['high', 'medium', 'low', 'none']", "Efficacy level reported", "high"
            )
            .add_regex(
                "pmid_references",
                r"PMID:\s*(\d+)",
                expected=3,
                match_type="count",
                description="Should cite 3 PMID references",
            )
            .add_regex(
                "p_values",
                r"p\s*[<>=]\s*0\.\d+",
                expected=1,
                match_type="count",
                description="Should report at least one p-value",
            )
            .compile()
        )

        # Test the compiled class
        assert issubclass(Answer, BaseAnswer)

        # Create instance with correct values
        answer = Answer(
            mentions_study_drug=True, mentions_control_group=True, sample_size=245, efficacy_category="high"
        )

        # Test classical verification
        assert answer.verify() is True

        # Test regex verification
        test_response = """
        The study drug showed high efficacy compared to the control group.
        Sample size was 245 patients (p = 0.001).
        References: PMID: 12345, PMID: 67890, PMID: 11111
        """

        regex_result = answer.verify_regex(test_response)
        assert regex_result["success"] is True
        assert regex_result["results"]["pmid_references"] is True
        assert regex_result["results"]["p_values"] is True
