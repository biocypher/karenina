import os
import tempfile
from unittest.mock import patch

import pytest

from karenina.questions.reader import read_questions_from_file
from karenina.schemas.question_class import Question


def test_read_questions_from_file_success():
    """Test successfully reading questions from a valid Python file."""
    # Create a temporary questions.py file
    questions_content = """
from karenina.schemas.question_class import Question

question_1 = Question(
    id="test1",
    question="Test question 1?",
    raw_answer="Yes",
    tags=["tag1"],
)

question_2 = Question(
    id="test2",
    question="Test question 2?",
    raw_answer="No",
    tags=[],
)

all_questions = [question_1, question_2]
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write(questions_content)
        tmp_path = tmp.name

    try:
        # Test reading the questions
        questions = read_questions_from_file(tmp_path)

        assert len(questions) == 2
        assert isinstance(questions[0], Question)
        assert isinstance(questions[1], Question)

        assert questions[0].id == "test1"
        assert questions[0].question == "Test question 1?"
        assert questions[0].raw_answer == "Yes"
        assert questions[0].tags == ["tag1"]

        assert questions[1].id == "test2"
        assert questions[1].question == "Test question 2?"
        assert questions[1].raw_answer == "No"
        assert questions[1].tags == []

    finally:
        os.unlink(tmp_path)


def test_read_questions_from_file_nonexistent():
    """Test reading from a non-existent file raises FileNotFoundError."""
    non_existent_path = "/path/that/does/not/exist.py"

    with pytest.raises(FileNotFoundError) as exc_info:
        read_questions_from_file(non_existent_path)

    assert "Questions file not found" in str(exc_info.value)
    assert non_existent_path in str(exc_info.value)


def test_read_questions_from_file_missing_all_questions():
    """Test reading from a file without 'all_questions' variable raises AttributeError."""
    # Create a file without all_questions
    questions_content = """
from karenina.schemas.question_class import Question

some_other_variable = "test"
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write(questions_content)
        tmp_path = tmp.name

    try:
        with pytest.raises(AttributeError) as exc_info:
            read_questions_from_file(tmp_path)

        assert "does not contain 'all_questions' variable" in str(exc_info.value)

    finally:
        os.unlink(tmp_path)


def test_read_questions_from_file_invalid_python():
    """Test reading from a file with invalid Python syntax raises ImportError."""
    # Create a file with invalid Python syntax
    invalid_content = """
from karenina.schemas.question_class import Question

this is not valid python syntax !!!
all_questions = []
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write(invalid_content)
        tmp_path = tmp.name

    try:
        with pytest.raises((SyntaxError, ImportError)):
            read_questions_from_file(tmp_path)

    finally:
        os.unlink(tmp_path)


def test_read_questions_from_file_empty_all_questions():
    """Test reading from a file with empty all_questions list."""
    questions_content = """
all_questions = []
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write(questions_content)
        tmp_path = tmp.name

    try:
        questions = read_questions_from_file(tmp_path)
        assert questions == []

    finally:
        os.unlink(tmp_path)


def test_read_questions_from_file_spec_creation_fails():
    """Test handling when module spec creation fails."""
    # Create a temporary file that exists but will fail spec creation
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write("# dummy content")
        tmp_path = tmp.name

    try:
        # Mock importlib.util.spec_from_file_location to return None
        with patch("otarbench.questions.reader.importlib.util.spec_from_file_location", return_value=None):
            with pytest.raises(ImportError) as exc_info:
                read_questions_from_file(tmp_path)

            assert "Could not create module spec" in str(exc_info.value)
    finally:
        os.unlink(tmp_path)


def test_read_questions_from_file_spec_loader_none():
    """Test handling when module spec has no loader."""
    # Create a temporary file that exists but will have no loader
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write("# dummy content")
        tmp_path = tmp.name

    try:
        # Create a mock spec with no loader
        class MockSpec:
            loader = None

        with patch("otarbench.questions.reader.importlib.util.spec_from_file_location", return_value=MockSpec()):
            with pytest.raises(ImportError) as exc_info:
                read_questions_from_file(tmp_path)

            assert "Could not create module spec" in str(exc_info.value)
    finally:
        os.unlink(tmp_path)


def test_read_questions_from_file_different_filenames():
    """Test reading questions from files with different names."""
    questions_content = """
from karenina.schemas.question_class import Question

all_questions = [
    Question(id="test", question="Q?", raw_answer="A", tags=[])
]
"""

    # Test with different file extensions and names
    for suffix in [".py", "_questions.py", "_test.py"]:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w") as tmp:
            tmp.write(questions_content)
            tmp_path = tmp.name

        try:
            questions = read_questions_from_file(tmp_path)
            assert len(questions) == 1
            assert questions[0].id == "test"

        finally:
            os.unlink(tmp_path)


def test_read_questions_from_file_module_sys_registration():
    """Test that modules are properly registered in sys.modules."""
    questions_content = """
from karenina.schemas.question_class import Question

all_questions = [
    Question(id="test", question="Q?", raw_answer="A", tags=[])
]
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write(questions_content)
        tmp_path = tmp.name
        module_name = os.path.splitext(os.path.basename(tmp_path))[0]

    try:
        # Ensure module is not already in sys.modules
        import sys

        if module_name in sys.modules:
            del sys.modules[module_name]

        questions = read_questions_from_file(tmp_path)

        # Verify module was registered
        assert module_name in sys.modules
        assert len(questions) == 1

    finally:
        os.unlink(tmp_path)
        # Clean up sys.modules
        if module_name in sys.modules:
            del sys.modules[module_name]


def test_read_questions_from_file_return_dict():
    """Test reading questions and returning as dictionary with return_dict=True."""
    # Create a temporary questions.py file with multiple questions
    questions_content = """
from karenina.schemas.question_class import Question

question_1 = Question(
    id="hash1",
    question="First question?",
    raw_answer="First answer",
    tags=["tag1"],
)

question_2 = Question(
    id="hash2",
    question="Second question?",
    raw_answer="Second answer",
    tags=["tag2"],
)

all_questions = [question_1, question_2]
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write(questions_content)
        tmp_path = tmp.name

    try:
        # Test returning as dictionary
        questions_dict = read_questions_from_file(tmp_path, return_dict=True)

        # Verify it's a dictionary
        assert isinstance(questions_dict, dict)
        assert len(questions_dict) == 2

        # Verify keys are question IDs and values are Question objects
        assert "hash1" in questions_dict
        assert "hash2" in questions_dict

        assert isinstance(questions_dict["hash1"], Question)
        assert isinstance(questions_dict["hash2"], Question)

        # Verify question content
        assert questions_dict["hash1"].question == "First question?"
        assert questions_dict["hash1"].raw_answer == "First answer"
        assert questions_dict["hash1"].tags == ["tag1"]

        assert questions_dict["hash2"].question == "Second question?"
        assert questions_dict["hash2"].raw_answer == "Second answer"
        assert questions_dict["hash2"].tags == ["tag2"]

    finally:
        os.unlink(tmp_path)


def test_read_questions_from_file_return_dict_vs_list():
    """Test that return_dict=False returns list and return_dict=True returns dict with same content."""
    questions_content = """
from karenina.schemas.question_class import Question

question_1 = Question(
    id="id1",
    question="Test question?",
    raw_answer="Test answer",
    tags=[],
)

all_questions = [question_1]
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write(questions_content)
        tmp_path = tmp.name

    try:
        # Get both formats
        questions_list = read_questions_from_file(tmp_path, return_dict=False)
        questions_dict = read_questions_from_file(tmp_path, return_dict=True)

        # Verify types
        assert isinstance(questions_list, list)
        assert isinstance(questions_dict, dict)

        # Verify same content
        assert len(questions_list) == 1
        assert len(questions_dict) == 1

        question_from_list = questions_list[0]
        question_from_dict = questions_dict["id1"]

        # They should be equivalent
        assert question_from_list.id == question_from_dict.id
        assert question_from_list.question == question_from_dict.question
        assert question_from_list.raw_answer == question_from_dict.raw_answer
        assert question_from_list.tags == question_from_dict.tags

    finally:
        os.unlink(tmp_path)


def test_read_questions_from_file_return_dict_empty():
    """Test return_dict=True with empty all_questions."""
    questions_content = """
all_questions = []
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write(questions_content)
        tmp_path = tmp.name

    try:
        questions_dict = read_questions_from_file(tmp_path, return_dict=True)
        assert isinstance(questions_dict, dict)
        assert len(questions_dict) == 0
        assert questions_dict == {}

    finally:
        os.unlink(tmp_path)


def test_read_questions_from_file_default_behavior_unchanged():
    """Test that default behavior (return_dict=False) is unchanged."""
    questions_content = """
from karenina.schemas.question_class import Question

all_questions = [
    Question(id="test", question="Q?", raw_answer="A", tags=[])
]
"""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
        tmp.write(questions_content)
        tmp_path = tmp.name

    try:
        # Test default behavior (no return_dict parameter)
        questions_default = read_questions_from_file(tmp_path)

        # Test explicit return_dict=False
        questions_explicit = read_questions_from_file(tmp_path, return_dict=False)

        # Both should return lists and be identical
        assert isinstance(questions_default, list)
        assert isinstance(questions_explicit, list)
        assert len(questions_default) == len(questions_explicit) == 1
        assert questions_default[0].id == questions_explicit[0].id

    finally:
        os.unlink(tmp_path)
