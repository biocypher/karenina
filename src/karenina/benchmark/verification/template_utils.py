"""Utility functions for working with Answer templates."""

from typing import Any, get_args, get_origin


def create_test_instance_from_answer_class(Answer: type) -> tuple[Any, dict[str, Any] | None]:
    """
    Create a test instance of an Answer class to extract ground truth and validate structure.

    This function provides dummy values for all required fields and then instantiates
    the Answer class to trigger model_post_init and extract the ground truth from
    the 'correct' field.

    Args:
        Answer: The Answer class to instantiate

    Returns:
        Tuple of (test_instance, ground_truth_dict)
        - test_instance: The instantiated Answer object
        - ground_truth_dict: The contents of the 'correct' field (None if not set)

    Raises:
        Exception: If the Answer class cannot be instantiated
    """
    # Get required fields to create a valid test instance
    required_fields: dict[str, Any] = {}
    if hasattr(Answer, "__annotations__"):
        for field_name, field_type in Answer.__annotations__.items():
            if field_name not in ("id", "correct"):  # Skip id and correct fields
                # Provide dummy values for required fields
                if field_type is int or str(field_type) == "int":
                    required_fields[field_name] = 0
                elif field_type is str or str(field_type) == "str":
                    required_fields[field_name] = ""
                elif field_type is float or str(field_type) == "float":
                    required_fields[field_name] = 0.0
                elif field_type is bool or str(field_type) == "bool":
                    required_fields[field_name] = False
                elif field_type is list or str(field_type) == "list":
                    required_fields[field_name] = []
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
                        # Handle List types
                        elif origin is list:
                            required_fields[field_name] = []
                        else:
                            # Default to empty string for unknown types
                            required_fields[field_name] = ""
                    else:
                        # Default to empty string for unknown types
                        required_fields[field_name] = ""

    # Create test instance to extract ground truth
    test_instance = Answer(**required_fields)

    # Extract ground truth if it exists
    ground_truth = None
    if hasattr(test_instance, "correct"):
        ground_truth = test_instance.correct

    return test_instance, ground_truth


def extract_ground_truth_from_template_code(template_code: str) -> dict[str, Any] | None:
    """
    Extract ground truth from Answer template code by creating a test instance.

    Args:
        template_code: The template code defining an Answer class

    Returns:
        The ground truth dictionary from the 'correct' field, or None if not available

    Raises:
        Exception: If the template code cannot be executed or Answer class cannot be instantiated
    """
    from ...schemas.answer_class import BaseAnswer

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
        raise ValueError("No 'Answer' class found in template code")

    Answer = local_ns["Answer"]

    # Store the template code for exec-created classes
    Answer._source_code = template_code

    # Create test instance and extract ground truth
    _, ground_truth = create_test_instance_from_answer_class(Answer)

    return ground_truth


def extract_rubric_traits_from_template(answer_template: str) -> list[Any]:
    """Extract rubric traits from answer template code.

    Args:
        answer_template: The answer template code string

    Returns:
        List of RubricTrait objects found in the template
    """
    try:
        # Prepare minimal execution environment similar to template validation
        from ...schemas.answer_class import BaseAnswer
        from ...schemas.rubric_class import Rubric, RubricTrait

        global_ns = {
            "__builtins__": __builtins__,
            "BaseAnswer": BaseAnswer,
            "Rubric": Rubric,
            "RubricTrait": RubricTrait,
        }
        try:
            from pydantic import Field

            global_ns["Field"] = Field
        except Exception:
            pass
        try:
            from typing import Any, ClassVar, Literal, Optional, Union

            global_ns.update(
                {
                    "List": list,
                    "Dict": dict,
                    "Optional": Optional,
                    "Union": Union,
                    "Any": Any,
                    "Literal": Literal,
                    "ClassVar": ClassVar,
                }
            )
        except Exception:
            pass

        local_ns: dict[str, Any] = {}
        exec(answer_template, global_ns, local_ns)

        # Store the template code for exec-created classes
        if "Answer" in local_ns:
            Answer = local_ns["Answer"]
            Answer._source_code = answer_template

        # Heuristics: check for rubric on Answer class or top-level var
        extracted_traits: list[RubricTrait] = []

        def _coerce_traits(obj: Any) -> list[RubricTrait]:
            traits_list: list[RubricTrait] = []
            if not obj:
                return traits_list
            # If wrapped in Rubric
            if isinstance(obj, Rubric):
                for t in obj.traits:
                    if isinstance(t, RubricTrait):
                        traits_list.append(t)
                return traits_list
            # If already list of RubricTrait
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, RubricTrait):
                        traits_list.append(item)
                    elif isinstance(item, dict) and "name" in item and "kind" in item:
                        try:
                            traits_list.append(RubricTrait(**item))
                        except Exception:
                            continue
            return traits_list

        AnswerCls = local_ns.get("Answer")
        if AnswerCls is not None:
            # Common attribute names that might store rubric traits
            for attr in ("question_rubric", "rubric_traits", "rubric"):
                if hasattr(AnswerCls, attr):
                    extracted_traits = _coerce_traits(getattr(AnswerCls, attr))
                    if extracted_traits:
                        break

        # Also allow a top-level constant like QUESTION_RUBRIC
        if not extracted_traits and "QUESTION_RUBRIC" in local_ns:
            extracted_traits = _coerce_traits(local_ns.get("QUESTION_RUBRIC"))

        return extracted_traits
    except Exception:
        # Silently ignore rubric extraction errors to keep TaskEval lightweight
        return []
