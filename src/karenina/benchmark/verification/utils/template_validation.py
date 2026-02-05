"""Template validation logic for verification system."""

import inspect

from karenina.schemas.entities import BaseAnswer


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

        # Store the template code for exec-created classes
        # (since inspect.getsource() won't work for them)
        Answer._source_code = template_code

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

        # The 'correct' field is optional, but if present via model_post_init, it must be a dict
        if "model_post_init" in Answer.__dict__:
            try:
                from .template_parsing_helpers import create_test_instance_from_answer_class

                test_instance, ground_truth = create_test_instance_from_answer_class(Answer)
                if ground_truth is not None and not isinstance(ground_truth, dict):
                    return False, "model_post_init must assign 'self.correct' as a dictionary", None
            except Exception as e:
                return False, f"Error testing model_post_init: {e}", None

        return True, None, Answer

    except SyntaxError as e:
        return False, f"Error executing template code: {e}", None
    except Exception as e:
        return False, f"Error executing template code: {e}", None
