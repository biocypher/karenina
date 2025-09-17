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

    # Create test instance and extract ground truth
    _, ground_truth = create_test_instance_from_answer_class(Answer)

    return ground_truth
