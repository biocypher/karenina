"""Tests for attribute name extraction from Answer templates.

This module tests the _extract_attribute_names_from_class function used
in deep-judgment to automatically extract attribute names from templates.
"""

from pydantic import Field

from karenina.benchmark.verification.parser_utils import _extract_attribute_names_from_class
from karenina.schemas.answer_class import BaseAnswer


class TestExtractAttributeNames:
    """Tests for _extract_attribute_names_from_class function."""

    def test_simple_answer_class(self):
        """Test extraction from simple answer class with a few attributes."""

        class SimpleAnswer(BaseAnswer):
            drug_target: str
            mechanism: str
            confidence: str

        attributes = _extract_attribute_names_from_class(SimpleAnswer)

        assert len(attributes) == 3
        assert "drug_target" in attributes
        assert "mechanism" in attributes
        assert "confidence" in attributes
        # Should exclude configuration fields
        assert "id" not in attributes
        assert "correct" not in attributes
        assert "regex" not in attributes

    def test_excludes_id_field(self):
        """Test that 'id' field is excluded from extraction."""

        class AnswerWithId(BaseAnswer):
            id: str | None = None
            drug_target: str

        attributes = _extract_attribute_names_from_class(AnswerWithId)

        # 'id' should be excluded (it's a metadata field)
        assert "id" not in attributes
        assert "drug_target" in attributes

    def test_excludes_correct_field(self):
        """Test that 'correct' field is excluded from extraction."""

        class AnswerWithCorrect(BaseAnswer):
            drug_target: str
            correct: dict | None = None

        attributes = _extract_attribute_names_from_class(AnswerWithCorrect)

        # 'correct' should be excluded (it's the ground truth field)
        assert "correct" not in attributes
        assert "drug_target" in attributes

    def test_excludes_regex_field(self):
        """Test that 'regex' field is excluded from extraction."""

        class AnswerWithRegex(BaseAnswer):
            drug_target: str
            regex: dict | None = None

        attributes = _extract_attribute_names_from_class(AnswerWithRegex)

        # 'regex' should be excluded (it's a validation configuration field)
        assert "regex" not in attributes
        assert "drug_target" in attributes

    def test_excludes_all_config_fields(self):
        """Test that all configuration fields are excluded together."""

        class AnswerWithAllConfig(BaseAnswer):
            id: str | None = None
            drug_target: str
            mechanism: str
            correct: dict | None = None
            regex: dict | None = None

        attributes = _extract_attribute_names_from_class(AnswerWithAllConfig)

        assert len(attributes) == 2
        assert "drug_target" in attributes
        assert "mechanism" in attributes
        assert "id" not in attributes
        assert "correct" not in attributes
        assert "regex" not in attributes

    def test_complex_answer_class(self):
        """Test extraction from complex answer class with many attributes."""

        class ComplexAnswer(BaseAnswer):
            drug_target: str
            mechanism: str
            confidence: str
            evidence: str
            limitations: str
            safety_profile: str
            efficacy_data: str
            clinical_trials: str
            approval_status: str
            dosing: str

        attributes = _extract_attribute_names_from_class(ComplexAnswer)

        assert len(attributes) == 10
        assert "drug_target" in attributes
        assert "dosing" in attributes
        # Configuration fields still excluded
        assert "id" not in attributes
        assert "correct" not in attributes

    def test_minimal_answer_class(self):
        """Test extraction from minimal answer class with single attribute."""

        class MinimalAnswer(BaseAnswer):
            answer: str

        attributes = _extract_attribute_names_from_class(MinimalAnswer)

        assert len(attributes) == 1
        assert "answer" in attributes

    def test_answer_with_field_descriptions(self):
        """Test extraction works with Field descriptions."""

        class DescribedAnswer(BaseAnswer):
            drug_target: str = Field(description="The drug target mentioned in the response")
            mechanism: str = Field(description="The mechanism of action")
            confidence: str = Field(description="Confidence level (high/medium/low)")

        attributes = _extract_attribute_names_from_class(DescribedAnswer)

        assert len(attributes) == 3
        assert "drug_target" in attributes
        assert "mechanism" in attributes
        assert "confidence" in attributes

    def test_answer_with_default_values(self):
        """Test extraction works with default values."""

        class AnswerWithDefaults(BaseAnswer):
            drug_target: str = "unknown"
            mechanism: str = "not specified"
            confidence: str = "low"

        attributes = _extract_attribute_names_from_class(AnswerWithDefaults)

        assert len(attributes) == 3
        assert "drug_target" in attributes
        assert "mechanism" in attributes
        assert "confidence" in attributes

    def test_answer_with_optional_fields(self):
        """Test extraction works with optional fields."""

        class AnswerWithOptional(BaseAnswer):
            drug_target: str
            mechanism: str | None = None
            additional_info: str | None = None

        attributes = _extract_attribute_names_from_class(AnswerWithOptional)

        assert len(attributes) == 3
        assert "drug_target" in attributes
        assert "mechanism" in attributes
        assert "additional_info" in attributes

    def test_answer_with_list_fields(self):
        """Test extraction works with list-type fields."""

        class AnswerWithLists(BaseAnswer):
            drug_targets: list[str]
            mechanisms: list[str]
            confidence: str

        attributes = _extract_attribute_names_from_class(AnswerWithLists)

        assert len(attributes) == 3
        assert "drug_targets" in attributes
        assert "mechanisms" in attributes
        assert "confidence" in attributes

    def test_answer_with_dict_fields(self):
        """Test extraction works with dict-type fields."""

        class AnswerWithDicts(BaseAnswer):
            metadata: dict[str, str]
            properties: dict[str, float]
            answer: str

        attributes = _extract_attribute_names_from_class(AnswerWithDicts)

        assert len(attributes) == 3
        assert "metadata" in attributes
        assert "properties" in attributes
        assert "answer" in attributes

    def test_attribute_order_preserved(self):
        """Test that attribute order is preserved from class definition."""

        class OrderedAnswer(BaseAnswer):
            first: str
            second: str
            third: str
            fourth: str

        attributes = _extract_attribute_names_from_class(OrderedAnswer)

        # Order should match definition order
        assert attributes == ["first", "second", "third", "fourth"]

    def test_snake_case_attributes(self):
        """Test extraction works with various naming conventions."""

        class SnakeCaseAnswer(BaseAnswer):
            drug_target: str
            mechanism_of_action: str
            confidence_level: str
            safety_profile_summary: str

        attributes = _extract_attribute_names_from_class(SnakeCaseAnswer)

        assert len(attributes) == 4
        assert "drug_target" in attributes
        assert "mechanism_of_action" in attributes
        assert "confidence_level" in attributes
        assert "safety_profile_summary" in attributes

    def test_real_world_answer_template(self):
        """Test with a realistic answer template structure."""

        class RealWorldAnswer(BaseAnswer):
            """Example answer template for drug information extraction."""

            drug_name: str = Field(description="Name of the drug")
            drug_target: str = Field(description="Primary molecular target")
            mechanism: str = Field(description="Mechanism of action")
            indication: str = Field(description="Primary indication")
            phase: str = Field(description="Clinical trial phase")
            confidence: str = Field(description="Confidence in extraction (high/medium/low)")

            # Configuration fields (should be excluded)
            id: str | None = None
            correct: dict | None = None
            regex: dict | None = None

        attributes = _extract_attribute_names_from_class(RealWorldAnswer)

        # Should extract only the 6 actual answer fields
        assert len(attributes) == 6
        assert "drug_name" in attributes
        assert "drug_target" in attributes
        assert "mechanism" in attributes
        assert "indication" in attributes
        assert "phase" in attributes
        assert "confidence" in attributes

        # Configuration fields excluded
        assert "id" not in attributes
        assert "correct" not in attributes
        assert "regex" not in attributes

    def test_empty_answer_class_edge_case(self):
        """Test handling of answer class with no extractable attributes."""

        class EmptyAnswer(BaseAnswer):
            id: str | None = None
            correct: dict | None = None

        attributes = _extract_attribute_names_from_class(EmptyAnswer)

        # Should return empty list when only config fields present
        assert len(attributes) == 0

    def test_pydantic_v2_compatibility(self):
        """Test that function works with Pydantic v2 model_fields."""

        class Pydantic2Answer(BaseAnswer):
            drug_target: str
            mechanism: str

        # Should work with model_fields (Pydantic v2)
        assert hasattr(Pydantic2Answer, "model_fields")
        attributes = _extract_attribute_names_from_class(Pydantic2Answer)

        assert len(attributes) == 2
        assert "drug_target" in attributes
        assert "mechanism" in attributes
