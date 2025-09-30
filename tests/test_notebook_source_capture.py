"""Tests for notebook source code capture functionality."""

from unittest.mock import MagicMock, patch

from pydantic import Field

from karenina.schemas.answer_class import BaseAnswer, capture_answer_source


class TestNotebookSourceCapture:
    """Test the notebook-specific source code capture functionality."""

    def test_capture_answer_source_decorator(self) -> None:
        """Test using capture_answer_source as a decorator."""

        # Mock IPython environment
        mock_ip = MagicMock()
        mock_history = [
            (
                1,
                1,
                """class TestAnswer(BaseAnswer):
    value: int = Field(description="Test value")

    def verify(self) -> bool:
        return self.value > 0""",
            )
        ]
        mock_ip.history_manager.get_range.return_value = mock_history

        with patch("IPython.get_ipython", return_value=mock_ip):

            @capture_answer_source
            class TestAnswer(BaseAnswer):
                value: int = Field(description="Test value")

                def verify(self) -> bool:
                    return self.value > 0

            # Should have captured source code
            source = TestAnswer.get_source_code()
            assert source is not None
            assert "class TestAnswer(BaseAnswer):" in source
            assert "value: int = Field" in source

    def test_capture_answer_source_function(self) -> None:
        """Test using capture_answer_source as a function."""

        # Mock IPython environment
        mock_ip = MagicMock()
        mock_history = [
            (
                1,
                1,
                """class FunctionTestAnswer(BaseAnswer):
    result: str = Field(description="Test result")

    def verify(self) -> bool:
        return len(self.result) > 0""",
            )
        ]
        mock_ip.history_manager.get_range.return_value = mock_history

        with patch("IPython.get_ipython", return_value=mock_ip):

            class FunctionTestAnswer(BaseAnswer):
                result: str = Field(description="Test result")

                def verify(self) -> bool:
                    return len(self.result) > 0

            # Apply the function
            FunctionTestAnswer = capture_answer_source(FunctionTestAnswer)

            # Should have captured source code
            source = FunctionTestAnswer.get_source_code()
            assert source is not None
            assert "class FunctionTestAnswer(BaseAnswer):" in source
            assert "result: str = Field" in source

    def test_set_source_code_from_notebook_method(self) -> None:
        """Test the set_source_code_from_notebook method directly."""

        # Create class dynamically using exec so inspect.getsource fails
        exec_globals = {"BaseAnswer": BaseAnswer, "Field": Field}
        exec(
            """
class DirectTestAnswer(BaseAnswer):
    data: int = Field(description="Test data")

    def verify(self) -> bool:
        return self.data >= 0
""",
            exec_globals,
        )
        DirectTestAnswer = exec_globals["DirectTestAnswer"]

        # Mock IPython environment with class in history
        mock_ip = MagicMock()
        mock_history = [
            (
                1,
                1,
                """class DirectTestAnswer(BaseAnswer):
    data: int = Field(description="Test data")

    def verify(self) -> bool:
        return self.data >= 0""",
            )
        ]
        mock_ip.history_manager.get_range.return_value = mock_history

        with patch("IPython.get_ipython", return_value=mock_ip):
            # Should start with no source code (exec-created class)
            assert DirectTestAnswer.get_source_code() is None

            # Capture source code
            DirectTestAnswer.set_source_code_from_notebook()

            # Should now have source code
            source = DirectTestAnswer.get_source_code()
            assert source is not None
            assert "class DirectTestAnswer(BaseAnswer):" in source

    def test_notebook_capture_no_ipython(self) -> None:
        """Test behavior when IPython is not available."""

        # Create class dynamically using exec so inspect.getsource fails
        exec_globals = {"BaseAnswer": BaseAnswer, "Field": Field}
        exec(
            """
class NoIPythonAnswer(BaseAnswer):
    test: bool = Field(description="Test field")

    def verify(self) -> bool:
        return self.test
""",
            exec_globals,
        )
        NoIPythonAnswer = exec_globals["NoIPythonAnswer"]

        # Mock ImportError when trying to import IPython
        with patch("IPython.get_ipython", side_effect=ImportError):
            # Should handle gracefully
            NoIPythonAnswer.set_source_code_from_notebook()

            # Should still be None (no IPython available)
            assert NoIPythonAnswer.get_source_code() is None

    def test_notebook_capture_not_in_ipython(self) -> None:
        """Test behavior when not in IPython environment."""

        # Create class dynamically using exec so inspect.getsource fails
        exec_globals = {"BaseAnswer": BaseAnswer, "Field": Field}
        exec(
            """
class NotInIPythonAnswer(BaseAnswer):
    flag: bool = Field(description="Test flag")

    def verify(self) -> bool:
        return self.flag
""",
            exec_globals,
        )
        NotInIPythonAnswer = exec_globals["NotInIPythonAnswer"]

        # Mock get_ipython returning None (not in IPython)
        with patch("IPython.get_ipython", return_value=None):
            # Should handle gracefully
            NotInIPythonAnswer.set_source_code_from_notebook()

            # Should still be None (not in IPython environment)
            assert NotInIPythonAnswer.get_source_code() is None

    def test_notebook_capture_class_not_found(self) -> None:
        """Test behavior when class definition is not found in history."""

        # Create class dynamically using exec so inspect.getsource fails
        exec_globals = {"BaseAnswer": BaseAnswer, "Field": Field}
        exec(
            """
class NotFoundAnswer(BaseAnswer):
    missing: str = Field(description="Missing field")

    def verify(self) -> bool:
        return len(self.missing) > 0
""",
            exec_globals,
        )
        NotFoundAnswer = exec_globals["NotFoundAnswer"]

        # Mock IPython with empty history
        mock_ip = MagicMock()
        mock_ip.history_manager.get_range.return_value = []

        with patch("IPython.get_ipython", return_value=mock_ip):
            # Should handle gracefully
            NotFoundAnswer.set_source_code_from_notebook()

            # Should still be None (class not found in history)
            assert NotFoundAnswer.get_source_code() is None

    def test_complex_class_extraction(self) -> None:
        """Test extracting class with complex indentation and structure."""

        class ComplexAnswer(BaseAnswer):
            pass

        # Mock complex cell content
        complex_cell = '''# Some comment
from pydantic import Field

class ComplexAnswer(BaseAnswer):
    """Complex answer class."""

    # Class variable
    DEFAULT_VALUE = 42

    # Fields
    primary: int = Field(description="Primary value")
    secondary: str = Field(description="Secondary value", default="test")

    def model_post_init(self, __context):
        self.correct = {"primary": 42, "secondary": "test"}

    def verify(self) -> bool:
        return (
            self.primary == self.correct["primary"] and
            self.secondary == self.correct["secondary"]
        )

# Some other code after the class
print("Hello")'''

        mock_ip = MagicMock()
        mock_ip.history_manager.get_range.return_value = [(1, 1, complex_cell)]

        with patch("IPython.get_ipython", return_value=mock_ip):
            ComplexAnswer.set_source_code_from_notebook()

            source = ComplexAnswer.get_source_code()
            assert source is not None
            assert "class ComplexAnswer(BaseAnswer):" in source
            assert "DEFAULT_VALUE = 42" in source
            assert "def verify(self) -> bool:" in source
            # Should not include the print statement after the class
            assert 'print("Hello")' not in source

    def test_multiple_classes_in_history(self) -> None:
        """Test finding the right class when multiple classes exist in history."""

        class SecondAnswer(BaseAnswer):
            pass

        # Mock history with multiple classes
        mock_ip = MagicMock()
        mock_history = [
            (
                1,
                1,
                """class FirstAnswer(BaseAnswer):
    first: int = Field(description="First")
    def verify(self): return True""",
            ),
            (
                1,
                2,
                """class SecondAnswer(BaseAnswer):
    second: str = Field(description="Second")
    def verify(self): return True""",
            ),
            (
                1,
                3,
                """class ThirdAnswer(BaseAnswer):
    third: bool = Field(description="Third")
    def verify(self): return True""",
            ),
        ]
        mock_ip.history_manager.get_range.return_value = mock_history

        with patch("IPython.get_ipython", return_value=mock_ip):
            SecondAnswer.set_source_code_from_notebook()

            source = SecondAnswer.get_source_code()
            assert source is not None
            assert "class SecondAnswer(BaseAnswer):" in source
            assert "second: str = Field" in source
            # Should not contain other classes
            assert "first: int = Field" not in source
            assert "third: bool = Field" not in source
