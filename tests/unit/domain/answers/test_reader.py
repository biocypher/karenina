"""Unit tests for answer template reader.

Tests for reading Answer class definitions from JSON files.
"""

import json

import pytest

from karenina.domain.answers.reader import read_answer_templates


class TestReadAnswerTemplates:
    """Tests for read_answer_templates function."""

    def test_read_single_template(self, tmp_path) -> None:
        """Test reading a JSON file with a single Answer template."""
        # Create test JSON file
        answers_file = tmp_path / "answers.json"
        template_code = '''
class Answer(BaseAnswer):
    """Simple answer template."""
    value: str = Field(description="The answer value")

    def model_post_init(self, __context):
        self.correct = {"value": "42"}

    def verify(self) -> bool:
        return self.value == self.correct["value"]
'''
        answers_data = {
            "q1_hash": template_code
        }
        answers_file.write_text(json.dumps(answers_data))

        # Read templates
        result = read_answer_templates(answers_file)

        # Verify result
        assert "q1_hash" in result
        Answer = result["q1_hash"]
        assert Answer.__name__ == "Answer1"
        answer = Answer(value="42")
        assert answer.verify() is True

    def test_read_multiple_templates(self, tmp_path) -> None:
        """Test reading a JSON file with multiple Answer templates."""
        answers_file = tmp_path / "answers.json"
        template1 = '''
class Answer(BaseAnswer):
    value1: str = Field(description="First value")
    def model_post_init(self, __context):
        self.correct = {"value1": "a"}
    def verify(self) -> bool:
        return self.value1 == self.correct["value1"]
'''
        template2 = '''
class Answer(BaseAnswer):
    value2: int = Field(description="Second value")
    def model_post_init(self, __context):
        self.correct = {"value2": 42}
    def verify(self) -> bool:
        return self.value2 == self.correct["value2"]
'''
        answers_data = {
            "hash1": template1,
            "hash2": template2,
        }
        answers_file.write_text(json.dumps(answers_data))

        result = read_answer_templates(answers_file)

        assert len(result) == 2
        assert "hash1" in result
        assert "hash2" in result

        # Verify first template
        Answer1 = result["hash1"]
        assert Answer1.__name__ == "Answer1"
        answer1 = Answer1(value1="a")
        assert answer1.verify() is True

        # Verify second template
        Answer2 = result["hash2"]
        assert Answer2.__name__ == "Answer2"
        answer2 = Answer2(value2=42)
        assert answer2.verify() is True

    def test_read_with_complex_template(self, tmp_path) -> None:
        """Test reading a template with multiple fields and types."""
        answers_file = tmp_path / "answers.json"
        template = '''
class Answer(BaseAnswer):
    name: str = Field(description="Entity name")
    count: int = Field(description="Item count")
    active: bool = Field(default=True, description="Active status")
    tags: list[str] = Field(default_factory=list, description="Tags")

    def model_post_init(self, __context):
        self.correct = {"name": "Test", "count": 5, "active": True, "tags": []}

    def verify(self) -> bool:
        return (self.name == self.correct["name"] and
                self.count == self.correct["count"] and
                self.active == self.correct["active"] and
                self.tags == self.correct["tags"])
'''
        answers_data = {"complex_hash": template}
        answers_file.write_text(json.dumps(answers_data))

        result = read_answer_templates(answers_file)

        Answer = result["complex_hash"]
        answer = Answer(name="Test", count=5, active=True, tags=[])
        assert answer.verify() is True

    def test_read_preserves_source_code(self, tmp_path) -> None:
        """Test that the original source code is preserved."""
        answers_file = tmp_path / "answers.json"
        template = '''
class Answer(BaseAnswer):
    value: str = Field(description="Value")
    def model_post_init(self, __context):
        self.correct = {"value": "test"}
    def verify(self) -> bool:
        return self.value == self.correct["value"]
'''
        answers_data = {"hash1": template}
        answers_file.write_text(json.dumps(answers_data))

        result = read_answer_templates(answers_file)
        Answer = result["hash1"]

        assert hasattr(Answer, "_source_code")
        # The source code should be the original template
        assert "class Answer(BaseAnswer):" in Answer._source_code

    def test_read_with_literal_type(self, tmp_path) -> None:
        """Test reading a template that uses Literal type."""
        answers_file = tmp_path / "answers.json"
        template = '''
from typing import Literal

class Answer(BaseAnswer):
    choice: Literal["A", "B", "C"] = Field(description="Multiple choice")

    def model_post_init(self, __context):
        self.correct = {"choice": "B"}

    def verify(self) -> bool:
        return self.choice == self.correct["choice"]
'''
        answers_data = {"hash1": template}
        answers_file.write_text(json.dumps(answers_data))

        result = read_answer_templates(answers_file)
        Answer = result["hash1"]

        answer = Answer(choice="B")
        assert answer.choice == "B"

    def test_read_with_list_type(self, tmp_path) -> None:
        """Test reading a template that uses List type."""
        answers_file = tmp_path / "answers.json"
        template = '''
from typing import List

class Answer(BaseAnswer):
    items: List[str] = Field(description="List of items")

    def model_post_init(self, __context):
        self.correct = {"items": ["a", "b", "c"]}

    def verify(self) -> bool:
        return self.items == self.correct["items"]
'''
        answers_data = {"hash1": template}
        answers_file.write_text(json.dumps(answers_data))

        result = read_answer_templates(answers_file)
        Answer = result["hash1"]

        answer = Answer(items=["a", "b", "c"])
        assert answer.items == ["a", "b", "c"]

    def test_read_nonexistent_file_raises(self, tmp_path) -> None:
        """Test that reading a non-existent file raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            read_answer_templates(nonexistent)

    def test_read_with_empty_json_raises(self, tmp_path) -> None:
        """Test reading an empty JSON file."""
        answers_file = tmp_path / "answers.json"
        answers_file.write_text("{}")

        result = read_answer_templates(answers_file)
        assert result == {}

    def test_read_with_invalid_json_raises(self, tmp_path) -> None:
        """Test reading a file with invalid JSON raises error."""
        answers_file = tmp_path / "answers.json"
        answers_file.write_text("{invalid json}")

        with pytest.raises(json.JSONDecodeError):
            read_answer_templates(answers_file)

    def test_read_with_invalid_syntax_raises(self, tmp_path) -> None:
        """Test reading a file with invalid Python syntax raises error."""
        answers_file = tmp_path / "answers.json"
        invalid_template = '''
class Answer(BaseAnswer)
    # Missing colon and body
'''
        answers_data = {"hash1": invalid_template}
        answers_file.write_text(json.dumps(answers_data))

        # The exec() should raise a SyntaxError
        with pytest.raises(Exception):  # Could be SyntaxError or other
            read_answer_templates(answers_file)

    def test_read_injects_question_id(self, tmp_path) -> None:
        """Test that question ID is injected into the Answer class."""
        answers_file = tmp_path / "answers.json"
        template = '''
class Answer(BaseAnswer):
    value: str = Field(description="Value")
    def model_post_init(self, __context):
        self.correct = {"value": "test"}
    def verify(self) -> bool:
        return self.value == self.correct["value"]
'''
        question_id = "test_question_123"
        answers_data = {question_id: template}
        answers_file.write_text(json.dumps(answers_data))

        result = read_answer_templates(answers_file)
        Answer = result[question_id]

        # Check that question_id is injected as instance attribute 'id'
        answer = Answer(value="test")
        assert hasattr(answer, "id")
        assert answer.id == question_id

    def test_read_multiple_classes_get_different_names(self, tmp_path) -> None:
        """Test that multiple Answer classes get unique names (Answer1, Answer2, etc.)."""
        answers_file = tmp_path / "answers.json"
        template = '''
class Answer(BaseAnswer):
    value: str = Field(description="Value")
    def model_post_init(self, __context):
        self.correct = {"value": "test"}
    def verify(self) -> bool:
        return self.value == self.correct["value"]
'''
        answers_data = {
            "hash1": template,
            "hash2": template,
            "hash3": template,
        }
        answers_file.write_text(json.dumps(answers_data))

        result = read_answer_templates(answers_file)

        # Each class should have a unique name
        assert result["hash1"].__name__ == "Answer1"
        assert result["hash2"].__name__ == "Answer2"
        assert result["hash3"].__name__ == "Answer3"

    def test_read_with_optional_field(self, tmp_path) -> None:
        """Test reading a template with optional fields."""
        answers_file = tmp_path / "answers.json"
        template = '''
from typing import Optional

class Answer(BaseAnswer):
    required: str = Field(description="Required field")
    optional: Optional[str] = Field(default=None, description="Optional field")

    def model_post_init(self, __context):
        self.correct = {"required": "value", "optional": None}

    def verify(self) -> bool:
        return self.required == self.correct["required"] and self.optional == self.correct["optional"]
'''
        answers_data = {"hash1": template}
        answers_file.write_text(json.dumps(answers_data))

        result = read_answer_templates(answers_file)
        Answer = result["hash1"]

        # Test with optional field
        answer = Answer(required="value", optional=None)
        assert answer.verify() is True

        # Test without optional field
        answer2 = Answer(required="value")
        assert answer2.verify() is True

    def test_read_with_union_types(self, tmp_path) -> None:
        """Test reading a template with Union types."""
        answers_file = tmp_path / "answers.json"
        template = '''
from typing import Union

class Answer(BaseAnswer):
    value: Union[str, int] = Field(description="String or int value")

    def model_post_init(self, __context):
        self.correct = {"value": 42}

    def verify(self) -> bool:
        # Handle both int and string representations
        if isinstance(self.value, str):
            return int(self.value) == self.correct["value"]
        return self.value == self.correct["value"]
'''
        answers_data = {"hash1": template}
        answers_file.write_text(json.dumps(answers_data))

        result = read_answer_templates(answers_file)
        Answer = result["hash1"]

        # Should accept int
        answer1 = Answer(value=42)
        assert answer1.verify() is True

        # Should also accept str (with proper comparison)
        answer2 = Answer(value="42")
        assert answer2.verify() is True
