import json
import os
import tempfile

import pytest

from karenina.domain.answers.reader import read_answer_templates


def test_read_answer_templates_success() -> None:
    """Test successfully reading answer templates from a JSON file."""
    # Create test answer templates data
    answer_templates = {
        "test1": """class Answer(BaseAnswer):
    answer: bool = Field(description="Test answer")

    def model_post_init(self, __context):
        self.id = "test1"
        self.correct = True""",
        "test2": """class Answer(BaseAnswer):
    answer: str = Field(description="String answer")

    def model_post_init(self, __context):
        self.id = "test2"
        self.correct = False""",
    }

    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(answer_templates, tmp)
        tmp_path = tmp.name

    try:
        # Test reading the answer templates
        result = read_answer_templates(tmp_path)

        assert len(result) == 2
        assert "test1" in result
        assert "test2" in result

        # Test that the classes were created correctly
        Answer1 = result["test1"]
        Answer2 = result["test2"]

        # Create instances and test them
        answer1 = Answer1(answer=True)
        assert answer1.id == "test1"
        assert answer1.correct is True

        answer2 = Answer2(answer="test_string")
        assert answer2.id == "test2"
        assert answer2.correct is False

    finally:
        os.unlink(tmp_path)


def test_read_answer_templates_with_pathlib() -> None:
    """Test reading answer templates using a pathlib.Path object."""
    from pathlib import Path

    answer_templates = {
        "path_test": """class Answer(BaseAnswer):
    answer: int = Field(description="Integer answer")

    def model_post_init(self, __context):
        self.id = "path_test"
        self.correct = True"""
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(answer_templates, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = read_answer_templates(tmp_path)
        assert len(result) == 1
        assert "path_test" in result

        Answer = result["path_test"]
        answer = Answer(answer=42)
        assert answer.id == "path_test"
        assert answer.correct is True

    finally:
        os.unlink(tmp_path)


def test_read_answer_templates_multiple_classes() -> None:
    """Test reading multiple answer templates with different class structures."""
    answer_templates = {
        "class1": '''class Answer(BaseAnswer):
    value: str = Field(description="String value")
    count: int = Field(default=0, description="Count value")

    def model_post_init(self, __context):
        self.id = "class1"''',
        "class2": '''class Answer(BaseAnswer):
    items: List[str] = Field(description="List of items")
    category: Literal["A", "B", "C"] = Field(description="Category")

    def model_post_init(self, __context):
        self.id = "class2"''',
        "class3": '''class Answer(BaseAnswer):
    is_valid: bool = Field(description="Validity flag")

    def model_post_init(self, __context):
        self.id = "class3"''',
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(answer_templates, tmp)
        tmp_path = tmp.name

    try:
        result = read_answer_templates(tmp_path)
        assert len(result) == 3

        # Test class1
        Answer1 = result["class1"]
        answer1 = Answer1(value="test", count=5)
        assert answer1.id == "class1"
        assert answer1.value == "test"
        assert answer1.count == 5

        # Test class2
        Answer2 = result["class2"]
        answer2 = Answer2(items=["item1", "item2"], category="A")
        assert answer2.id == "class2"
        assert answer2.items == ["item1", "item2"]
        assert answer2.category == "A"

        # Test class3
        Answer3 = result["class3"]
        answer3 = Answer3(is_valid=True)
        assert answer3.id == "class3"
        assert answer3.is_valid is True

    finally:
        os.unlink(tmp_path)


def test_read_answer_templates_file_not_found() -> None:
    """Test that FileNotFoundError is raised for non-existent file."""
    with pytest.raises(FileNotFoundError):
        read_answer_templates("/path/that/does/not/exist.json")


def test_read_answer_templates_invalid_json() -> None:
    """Test handling of invalid JSON file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        tmp.write("invalid json content")
        tmp_path = tmp.name

    try:
        with pytest.raises(json.JSONDecodeError):
            read_answer_templates(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_read_answer_templates_invalid_python_code() -> None:
    """Test handling of invalid Python code in answer templates."""
    answer_templates = {"invalid": "this is not valid python code !!!"}

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(answer_templates, tmp)
        tmp_path = tmp.name

    try:
        with pytest.raises(SyntaxError):
            read_answer_templates(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_read_answer_templates_empty_file() -> None:
    """Test reading from an empty JSON file."""
    answer_templates = {}

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json.dump(answer_templates, tmp)
        tmp_path = tmp.name

    try:
        result = read_answer_templates(tmp_path)
        assert result == {}
    finally:
        os.unlink(tmp_path)
