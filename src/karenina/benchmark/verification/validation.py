"""Template validation logic for verification system."""

import inspect
from typing import Any, get_origin

from ...schemas.answer_class import BaseAnswer


def _is_dict_type(annotation: Any) -> bool:
    """Check if a type annotation represents a dictionary type."""
    # Direct dict type
    if annotation is dict:
        return True

    # typing.Dict or Dict from globals
    if hasattr(annotation, "__name__") and annotation.__name__ == "Dict":
        return True

    # Generic types like Dict[str, Any], dict[str, Any]
    origin = get_origin(annotation)
    if origin is dict:
        return True

    # Handle string annotations
    if isinstance(annotation, str):
        return annotation.lower().startswith(("dict", "Dict"))

    return False


def validate_answer_template(template_code: str) -> tuple[bool, str | None, type | None]:
    """
    Validate that template code defines a proper Answer class.

    Returns:
        (is_valid, error_message, Answer_class)
    """
    try:
        # Execute the template code to get the Answer class
        # Create a namespace with necessary imports
        global_ns = {
            "__builtins__": __builtins__,
            "BaseAnswer": BaseAnswer,
        }

        # Import commonly used pydantic and typing components
        try:
            from pydantic import Field

            global_ns["Field"] = Field
        except ImportError:
            pass

        try:
            from typing import Any, Literal, Optional, Union

            global_ns.update(
                {
                    "List": list,
                    "Dict": dict,
                    "Optional": Optional,
                    "Union": Union,
                    "Any": Any,
                    "Literal": Literal,
                }
            )
        except ImportError:
            pass

        local_ns: dict[str, Any] = {}

        # Execute the template code
        exec(template_code, global_ns, local_ns)

        # Check if Answer class was defined
        if "Answer" not in local_ns:
            return False, "No 'Answer' class found", None

        Answer = local_ns["Answer"]

        # Check if it's a class
        if not inspect.isclass(Answer):
            return False, "Answer is not a class", None

        # Check if it inherits from BaseAnswer
        if not issubclass(Answer, BaseAnswer):
            return False, "Answer class must inherit from BaseAnswer", None

        # Check if it has a verify method
        if not hasattr(Answer, "verify"):
            return False, "does not have a 'verify' method", None

        # Check if verify method is callable
        if not callable(getattr(Answer, "verify", None)):
            return False, "verify must be a callable method", None

        # Check if it has a 'correct' field annotation
        if not hasattr(Answer, "__annotations__") or "correct" not in Answer.__annotations__:
            return False, "Answer class must have a 'correct' field", None

        # Check if 'correct' is annotated as a dictionary type
        correct_annotation = Answer.__annotations__["correct"]
        # Handle various dictionary type annotations
        if not _is_dict_type(correct_annotation):
            return False, "Answer class 'correct' field must be annotated as dict type", None

        return True, None, Answer

    except SyntaxError as e:
        return False, f"Error executing template code: {e}", None
    except Exception as e:
        return False, f"Error executing template code: {e}", None
