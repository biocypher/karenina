"""Base answer class for Karenina.

This module defines the BaseAnswer class, which serves as the foundation for
all answer templates in the benchmark. It provides common functionality and
validation for answer structures.
"""

import inspect
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict


class BaseAnswer(BaseModel):
    """Base class for all answer templates in Karenina.

    This class provides common functionality and configuration for answer
    validation and processing.
    """

    model_config = ConfigDict(extra="allow")

    # Question ID will be set programmatically after class instantiation
    id: str | None = None

    # Source code storage (set automatically via __init_subclass__ or manually for exec-created classes)
    # Using ClassVar to prevent Pydantic from treating this as a model field
    _source_code: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically capture source code when Answer classes are defined.

        This hook is called whenever a class inherits from BaseAnswer.
        It attempts to capture the source code using inspect.getsource().
        For exec-created classes, this will fail and _source_code will be None,
        but can be set manually afterwards.
        """
        super().__init_subclass__(**kwargs)
        try:
            cls._source_code = inspect.getsource(cls)
        except (OSError, TypeError):
            # This happens for exec-created classes or when source isn't available
            # The source code can be set manually after class creation
            cls._source_code = None

    @classmethod
    def get_source_code(cls) -> str | None:
        """Get the source code of this Answer class.

        Returns:
            The source code string if available, None otherwise.

        For file-based classes, source code is captured automatically.
        For exec-created classes, source code must be set manually.
        """
        return cls._source_code

    @classmethod
    def set_source_code_from_notebook(cls) -> None:
        """Capture source code from notebook cell history (Jupyter/IPython only).

        This is a convenience method for interactive environments where
        inspect.getsource() doesn't work. It attempts to find the class
        definition in the recent cell execution history.

        Usage in notebook:
            class Answer(BaseAnswer):
                # your class definition
                pass
            Answer.set_source_code_from_notebook()
        """
        try:
            # Try to get IPython instance (works in Jupyter notebooks)
            from IPython import get_ipython

            ip = get_ipython()
            if ip is None:
                print("Warning: Not in an IPython/Jupyter environment")
                return

            # Get recent cell history
            history = list(ip.history_manager.get_range())

            # Look for the class definition in recent history (last 10 cells)
            class_name = cls.__name__
            for _, _, cell_content in reversed(history[-10:]):
                if f"class {class_name}(" in cell_content:
                    # Extract just the class definition part
                    lines = cell_content.strip().split("\n")
                    class_lines = []
                    in_class = False
                    base_indent = None

                    for line in lines:
                        if f"class {class_name}(" in line:
                            in_class = True
                            base_indent = len(line) - len(line.lstrip())
                            class_lines.append(line)
                        elif in_class:
                            if line.strip() == "" or (
                                base_indent is not None and len(line) - len(line.lstrip()) > base_indent
                            ):
                                class_lines.append(line)
                            else:
                                # End of class definition
                                break

                    if class_lines:
                        cls._source_code = "\n".join(class_lines)
                        print(f"✓ Source code captured for {class_name}")
                        return

            print(f"Warning: Could not find class definition for {class_name} in recent history")

        except ImportError:
            print("Warning: IPython not available. This method only works in Jupyter notebooks.")
        except Exception as e:
            print(f"Warning: Could not capture source code: {e}")

    def set_question_id(self, question_id: str) -> None:
        """Set the question ID programmatically.

        Args:
            question_id: The unique identifier for the question this answer relates to.
        """
        self.id = question_id


def capture_answer_source(answer_class: type) -> type:
    """Decorator/function to automatically capture source code for Answer classes in notebooks.

    Usage as decorator:
        @capture_answer_source
        class Answer(BaseAnswer):
            # your class definition
            pass

    Usage as function:
        class Answer(BaseAnswer):
            # your class definition
            pass
        Answer = capture_answer_source(Answer)

    Args:
        answer_class: The Answer class to capture source for

    Returns:
        The same class with source code captured
    """
    if hasattr(answer_class, "set_source_code_from_notebook"):
        answer_class.set_source_code_from_notebook()
    return answer_class
