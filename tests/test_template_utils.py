"""Tests for template utility functions."""

from typing import Literal

import pytest
from pydantic import Field

from karenina.benchmark.verification.utils.parsing import (
    create_test_instance_from_answer_class,
    extract_ground_truth_from_template_code,
)
from karenina.schemas.domain import BaseAnswer


class TestTemplateUtils:
    """Test the template utility functions."""

    def test_create_test_instance_simple(self) -> None:
        """Test creating a test instance with a simple Answer class."""

        class Answer(BaseAnswer):
            answer: int = Field(description="The answer", default=0)

            def model_post_init(self, __context) -> None:
                self.correct = {"answer": 42}

            def verify(self) -> bool:
                return self.answer == self.correct["answer"]

        test_instance, ground_truth = create_test_instance_from_answer_class(Answer)

        assert test_instance is not None
        assert hasattr(test_instance, "answer")
        assert test_instance.answer == 0  # Default value
        assert ground_truth is not None
        assert ground_truth == {"answer": 42}

    def test_create_test_instance_complex_types(self) -> None:
        """Test creating a test instance with complex types like Literal."""

        class Answer(BaseAnswer):
            phase: Literal["Phase I", "Phase II", "Phase III"] = Field(description="Phase", default="Phase I")
            status: Literal["Active", "Completed"] = Field(description="Status", default="Active")

            def model_post_init(self, __context) -> None:
                self.correct = {"phase": "Phase II", "status": "Completed"}

            def verify(self) -> bool:
                return self.phase == self.correct["phase"] and self.status == self.correct["status"]

        test_instance, ground_truth = create_test_instance_from_answer_class(Answer)

        assert test_instance is not None
        assert test_instance.phase == "Phase I"  # First literal value
        assert test_instance.status == "Active"  # First literal value
        assert ground_truth is not None
        assert ground_truth == {"phase": "Phase II", "status": "Completed"}

    def test_create_test_instance_no_ground_truth(self) -> None:
        """Test creating a test instance with no model_post_init (no ground truth)."""

        class Answer(BaseAnswer):
            answer: str = Field(description="The answer", default="")

            def verify(self) -> bool:
                return True

        test_instance, ground_truth = create_test_instance_from_answer_class(Answer)

        assert test_instance is not None
        assert test_instance.answer == ""
        assert ground_truth is None

    def test_create_test_instance_various_types(self) -> None:
        """Test creating a test instance with various field types."""

        class Answer(BaseAnswer):
            integer_field: int = Field(description="An integer", default=0)
            string_field: str = Field(description="A string", default="")
            float_field: float = Field(description="A float", default=0.0)
            bool_field: bool = Field(description="A boolean", default=False)
            list_field: list = Field(description="A list", default=[])

            def model_post_init(self, __context) -> None:
                self.correct = {
                    "integer_field": 123,
                    "string_field": "test",
                    "float_field": 3.14,
                    "bool_field": True,
                    "list_field": ["a", "b", "c"],
                }

            def verify(self) -> bool:
                return True

        test_instance, ground_truth = create_test_instance_from_answer_class(Answer)

        assert test_instance is not None
        assert test_instance.integer_field == 0
        assert test_instance.string_field == ""
        assert test_instance.float_field == 0.0
        assert test_instance.bool_field is False
        assert test_instance.list_field == []
        assert ground_truth is not None
        assert ground_truth["integer_field"] == 123
        assert ground_truth["string_field"] == "test"
        assert ground_truth["float_field"] == 3.14
        assert ground_truth["bool_field"] is True
        assert ground_truth["list_field"] == ["a", "b", "c"]

    def test_extract_ground_truth_from_template_code(self) -> None:
        """Test extracting ground truth from template code string."""

        template_code = """class Answer(BaseAnswer):
    answer: int = Field(description="The answer", default=0)

    def model_post_init(self, __context):
        self.correct = {"answer": 42}

    def verify(self) -> bool:
        return self.answer == self.correct["answer"]
"""

        ground_truth = extract_ground_truth_from_template_code(template_code)

        assert ground_truth is not None
        assert ground_truth == {"answer": 42}

    def test_extract_ground_truth_from_template_code_no_ground_truth(self) -> None:
        """Test extracting ground truth from template code with no model_post_init."""

        template_code = """class Answer(BaseAnswer):
    answer: str = Field(description="The answer", default="")

    def verify(self) -> bool:
        return True
"""

        ground_truth = extract_ground_truth_from_template_code(template_code)

        assert ground_truth is None

    def test_extract_ground_truth_invalid_template_code(self) -> None:
        """Test error handling for invalid template code."""

        template_code = "invalid python code @#$%"

        with pytest.raises(SyntaxError):
            extract_ground_truth_from_template_code(template_code)

    def test_extract_ground_truth_no_answer_class(self) -> None:
        """Test error handling for template code with no Answer class."""

        template_code = """class SomeOtherClass:
    pass
"""

        with pytest.raises(ValueError, match="No 'Answer' class found"):
            extract_ground_truth_from_template_code(template_code)

    def test_create_test_instance_error_handling(self) -> None:
        """Test error handling when test instance creation fails."""

        class BadAnswer(BaseAnswer):
            required_field: int  # No default value

            def model_post_init(self, __context) -> None:
                self.correct = {"required_field": 42}

            def verify(self) -> bool:
                return True

        # This should work because our utility provides default values
        test_instance, ground_truth = create_test_instance_from_answer_class(BadAnswer)

        assert test_instance is not None
        assert test_instance.required_field == 0  # Our default value
        assert ground_truth == {"required_field": 42}
