"""Template validation logic for verification system."""

import inspect
from typing import get_args, get_origin

from ...schemas.answer_class import BaseAnswer


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

        # The 'correct' field is optional, but if present via model_post_init, it must be a dict
        if "model_post_init" in Answer.__dict__:
            try:
                from typing import Any

                # Get required fields to create a valid test instance
                required_fields: dict[str, Any] = {}
                if hasattr(Answer, "__annotations__"):
                    for field_name, field_type in Answer.__annotations__.items():
                        if field_name != "correct":  # Skip correct field as it might be set by model_post_init
                            # Provide dummy values for required fields
                            if field_type is int or str(field_type) == "int":
                                required_fields[field_name] = 0
                            elif field_type is str or str(field_type) == "str":
                                required_fields[field_name] = ""
                            elif field_type is float or str(field_type) == "float":
                                required_fields[field_name] = 0.0
                            elif field_type is bool or str(field_type) == "bool":
                                required_fields[field_name] = False
                            else:
                                # Handle Literal and other complex types
                                origin = get_origin(field_type)
                                if origin is not None:
                                    # Handle Literal types
                                    if str(origin) == "typing.Literal":
                                        # Get the first literal value
                                        args = get_args(field_type)
                                        if args:
                                            required_fields[field_name] = args[0]
                                        else:
                                            required_fields[field_name] = ""
                                    else:
                                        # Default to empty string for unknown types
                                        required_fields[field_name] = ""
                                else:
                                    # Default to empty string for unknown types
                                    required_fields[field_name] = ""

                test_instance = Answer(**required_fields)
                if hasattr(test_instance, "correct") and not isinstance(test_instance.correct, dict):
                    return False, "model_post_init must assign 'self.correct' as a dictionary", None
            except Exception as e:
                return False, f"Error testing model_post_init: {e}", None

        return True, None, Answer

    except SyntaxError as e:
        return False, f"Error executing template code: {e}", None
    except Exception as e:
        return False, f"Error executing template code: {e}", None
