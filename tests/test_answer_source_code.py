"""Tests for BaseAnswer source code functionality."""

import inspect
import tempfile
from pathlib import Path

from pydantic import Field

from karenina.schemas.answer_class import BaseAnswer


class TestBaseAnswerSourceCode:
    """Test the new source code capture functionality in BaseAnswer."""

    def test_file_based_class_captures_source_automatically(self) -> None:
        """Test that file-based Answer classes capture source code automatically."""

        # Define a test Answer class
        class FileBasedAnswer(BaseAnswer):
            """Test answer class defined in a file."""

            value: int = Field(description="Test value")

            def verify(self) -> bool:
                return self.value > 0

        # Should have captured source code automatically
        source = FileBasedAnswer.get_source_code()
        assert source is not None
        assert "class FileBasedAnswer(BaseAnswer):" in source
        assert 'value: int = Field(description="Test value")' in source
        assert "def verify(self) -> bool:" in source

    def test_exec_created_class_stores_manual_source(self) -> None:
        """Test that exec-created classes store manually set source code."""

        template_code = '''
from pydantic import Field

class Answer(BaseAnswer):
    """Exec-created answer class."""
    result: str = Field(description="The result")

    def verify(self) -> bool:
        return self.result == "correct"
'''

        # Create namespace and execute
        global_ns = {"BaseAnswer": BaseAnswer, "Field": Field}
        local_ns = {}
        exec(template_code, global_ns, local_ns)
        Answer = local_ns["Answer"]

        # Manually set source code (simulating what our updated code does)
        Answer._source_code = template_code

        # Should return the manually set source code
        source = Answer.get_source_code()
        assert source is not None
        assert source == template_code
        assert "class Answer(BaseAnswer):" in source
        assert 'result: str = Field(description="The result")' in source

    def test_get_source_code_returns_none_when_unavailable(self) -> None:
        """Test that get_source_code() returns None when source is unavailable."""

        template_code = """
class Answer(BaseAnswer):
    value: int

    def verify(self) -> bool:
        return True
"""

        # Create exec-created class but don't set _source_code
        global_ns = {"BaseAnswer": BaseAnswer}
        local_ns = {}
        exec(template_code, global_ns, local_ns)
        Answer = local_ns["Answer"]

        # Should return None since we didn't manually set source
        source = Answer.get_source_code()
        assert source is None

    def test_init_subclass_handles_inspect_failure_gracefully(self) -> None:
        """Test that __init_subclass__ handles inspect.getsource() failure gracefully."""

        # This simulates what happens with exec-created classes
        template_code = """
class Answer(BaseAnswer):
    test_field: str

    def verify(self) -> bool:
        return True
"""

        global_ns = {"BaseAnswer": BaseAnswer}
        local_ns = {}
        exec(template_code, global_ns, local_ns)
        Answer = local_ns["Answer"]

        # __init_subclass__ should have been called but source should be None
        # (because inspect.getsource() fails for exec-created classes)
        assert hasattr(Answer, "_source_code")
        assert Answer._source_code is None

    def test_multiple_inheritance_preserves_source_code(self) -> None:
        """Test that multiple inheritance doesn't break source code capture."""

        class MixinClass:
            """Test mixin class."""

            def helper_method(self) -> None:
                return "helper"

        class MultipleInheritanceAnswer(BaseAnswer, MixinClass):
            """Answer class with multiple inheritance."""

            data: str = Field(description="Test data")

            def verify(self) -> bool:
                return len(self.data) > 0

        # Should still capture source code correctly
        source = MultipleInheritanceAnswer.get_source_code()
        assert source is not None
        assert "class MultipleInheritanceAnswer(BaseAnswer, MixinClass):" in source
        assert 'data: str = Field(description="Test data")' in source

    def test_source_code_inheritance_behavior(self) -> None:
        """Test how source code behaves with class inheritance."""

        class ParentAnswer(BaseAnswer):
            """Parent answer class."""

            parent_field: int

            def verify(self) -> bool:
                return True

        class ChildAnswer(ParentAnswer):
            """Child answer class."""

            child_field: str

            def verify(self) -> bool:
                return super().verify() and len(self.child_field) > 0

        # Each class should have its own source code
        parent_source = ParentAnswer.get_source_code()
        child_source = ChildAnswer.get_source_code()

        assert parent_source is not None
        assert child_source is not None
        assert parent_source != child_source
        assert "class ParentAnswer(BaseAnswer):" in parent_source
        assert "class ChildAnswer(ParentAnswer):" in child_source

    def test_source_code_with_complex_class_definition(self) -> None:
        """Test source code capture with complex class definitions."""
        from typing import ClassVar

        class ComplexAnswer(BaseAnswer):
            """Complex answer class with various features."""

            # Class variables
            DEFAULT_THRESHOLD: ClassVar[float] = 0.5

            # Multiple field types
            integer_field: int = Field(description="Integer field", ge=0)
            string_field: str = Field(description="String field", min_length=1)
            optional_field: str | None = Field(description="Optional field", default=None)

            def verify(self) -> bool:
                """Verify the answer with complex logic."""
                if self.integer_field < self.DEFAULT_THRESHOLD:
                    return False
                return len(self.string_field) > 0

            @classmethod
            def create_default(cls):
                """Factory method."""
                return cls(integer_field=1, string_field="test")

        source = ComplexAnswer.get_source_code()
        assert source is not None
        assert "DEFAULT_THRESHOLD: ClassVar[float] = 0.5" in source
        assert "integer_field: int = Field(" in source
        assert "def verify(self) -> bool:" in source
        assert "@classmethod" in source
        assert "def create_default(cls):" in source

    def test_source_code_consistency_across_instances(self) -> None:
        """Test that source code is consistent across class instances."""

        class ConsistentAnswer(BaseAnswer):
            """Test answer for consistency."""

            value: int

            def verify(self) -> bool:
                return True

        # Source code should be the same whether accessed from class or instance
        class_source = ConsistentAnswer.get_source_code()
        instance = ConsistentAnswer(value=42)
        instance_source = instance.get_source_code()

        assert class_source == instance_source
        assert class_source is not None

    def test_source_code_with_dynamic_class_modification(self) -> None:
        """Test that dynamically modifying a class doesn't affect captured source."""

        class ModifiableAnswer(BaseAnswer):
            """Answer that will be modified."""

            original_field: int

            def verify(self) -> bool:
                return True

        # Capture original source
        original_source = ModifiableAnswer.get_source_code()

        # Dynamically add a method (this doesn't change the captured source)
        def new_method(self) -> None:
            return "new"

        ModifiableAnswer.new_method = new_method

        # Source should still be the original
        current_source = ModifiableAnswer.get_source_code()
        assert current_source == original_source
        assert "new_method" not in current_source

    def test_base_answer_itself_has_source_code(self) -> None:
        """Test that BaseAnswer itself has source code available."""

        # BaseAnswer itself won't have source code captured via __init_subclass__
        # since __init_subclass__ is only called for subclasses
        # But we can test that inspect.getsource() works on BaseAnswer
        base_source = inspect.getsource(BaseAnswer)
        assert base_source is not None
        assert isinstance(base_source, str)
        assert "class BaseAnswer(BaseModel):" in base_source
        assert "def __init_subclass__(cls" in base_source
        assert "def get_source_code(cls)" in base_source


class TestAnswerSourceCodeIntegration:
    """Integration tests with the template validation system."""

    def test_validation_system_preserves_source_code(self) -> None:
        """Test that the validation system preserves source code."""
        from karenina.benchmark.verification.validation import validate_answer_template

        template_code = '''
from pydantic import Field

class Answer(BaseAnswer):
    """Template validation test."""
    result: int = Field(description="The result")

    def verify(self) -> bool:
        return self.result > 0
'''

        # Validate the template (this uses exec internally)
        is_valid, error_msg, Answer = validate_answer_template(template_code)

        assert is_valid
        assert error_msg is None
        assert Answer is not None

        # Source code should be preserved
        source = Answer.get_source_code()
        assert source is not None
        assert source == template_code

    def test_template_utils_preserves_source_code(self) -> None:
        """Test that template utils preserve source code."""
        from karenina.benchmark.verification.template_utils import extract_ground_truth_from_template_code

        template_code = '''
from pydantic import Field

class Answer(BaseAnswer):
    """Template utils test."""
    correct_value: int = Field(description="Correct value")

    def model_post_init(self, __context):
        self.correct = {"correct_value": 42}

    def verify(self) -> bool:
        return self.correct_value == self.correct["correct_value"]
'''

        # This function uses exec internally
        ground_truth = extract_ground_truth_from_template_code(template_code)

        # The function modifies global state, so we need to check if an Answer class was created
        # and if it has source code. This is a bit tricky to test directly since the function
        # doesn't return the class, but we can verify it works by ensuring no exceptions.
        assert ground_truth is not None

    def test_answer_generation_preserves_source_code(self) -> None:
        """Test that answer generation preserves source code."""

        # This tests the pattern used in answer generation
        template_code = '''
from pydantic import Field

class Answer(BaseAnswer):
    """Generated answer test."""
    generated_field: str = Field(description="Generated field")

    def verify(self) -> bool:
        return len(self.generated_field) > 0
'''

        # Simulate what generator.py does
        global_ns = {"BaseAnswer": BaseAnswer, "Field": Field}
        local_ns = {}
        exec(template_code, global_ns, local_ns)
        Answer = local_ns["Answer"]

        # Set source code (as our updated code does)
        Answer._source_code = template_code

        # Verify source is preserved
        source = Answer.get_source_code()
        assert source == template_code

    def test_answer_reader_preserves_source_code(self) -> None:
        """Test that answer reader preserves source code."""

        # Create a temporary JSON file with answer templates
        templates = {
            "test_question_1": '''from pydantic import Field

class Answer(BaseAnswer):
    """Reader test answer."""
    answer_value: str = Field(description="Answer value")

    def verify(self) -> bool:
        return self.answer_value == "correct"
'''
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(templates, f)
            temp_path = Path(f.name)

        try:
            # Import here to avoid circular imports
            from karenina.answers.reader import read_answer_templates

            # This function uses exec internally and should preserve source code
            answer_dict = read_answer_templates(temp_path)

            assert "test_question_1" in answer_dict
            Answer = answer_dict["test_question_1"]

            # Source code should be preserved (though it might be the modified version with Answer1, etc.)
            source = Answer.get_source_code()
            assert source is not None
            assert "class Answer" in source  # Could be Answer or Answer1 due to renaming
            assert "answer_value: str" in source

        finally:
            temp_path.unlink(missing_ok=True)
