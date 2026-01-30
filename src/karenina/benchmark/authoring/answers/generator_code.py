"""
Pydantic class code generation for answer templates.

This module contains functions for generating Python/Pydantic code
from structured ground truth specifications.
"""

from typing import Any


def python_type_to_annotation(type_str: str) -> str:
    """Convert a Python type string to proper type annotation.

    Args:
        type_str: Type string like "bool", "int", "List[str]", "Literal['a', 'b']"

    Returns:
        Properly formatted type annotation string
    """
    # Handle Literal types
    if type_str.startswith("Literal"):
        return type_str

    # Handle basic types
    type_mapping = {"bool": "bool", "int": "int", "float": "float", "str": "str"}

    # Handle List types
    if type_str.startswith("List["):
        inner_type = type_str[5:-1]  # Extract inner type from List[...]
        return f"List[{python_type_to_annotation(inner_type)}]"

    # Handle Dict types
    if type_str.startswith("Dict["):
        # Extract key, value types from Dict[key, value]
        inner = type_str[5:-1]
        key_type, value_type = inner.split(", ")
        return f"Dict[{python_type_to_annotation(key_type)}, {python_type_to_annotation(value_type)}]"

    return type_mapping.get(type_str, type_str)


def generate_verification_logic(attr_name: str, attr_type: str, tolerance: float = 0.001) -> str:
    """Generate verification logic for a specific attribute type.

    Args:
        attr_name: Name of the attribute
        attr_type: Type of the attribute
        tolerance: Tolerance for float comparisons

    Returns:
        Python expression string for verification
    """
    if attr_type == "float":
        return f'abs(self.{attr_name} - self.correct["{attr_name}"]) <= {tolerance}'
    elif attr_type in ["bool", "int", "str"] or attr_type.startswith("Literal"):
        return f'self.{attr_name} == self.correct["{attr_name}"]'
    elif attr_type.startswith("List["):
        # For lists, compare as sets to ignore order or do exact comparison
        return f'set(self.{attr_name}) == set(self.correct["{attr_name}"])'
    elif attr_type.startswith("Dict["):
        return f'self.{attr_name} == self.correct["{attr_name}"]'
    else:
        # Default to equality check
        return f'self.{attr_name} == self.correct["{attr_name}"]'


def format_ground_truth_value(value: Any) -> str:
    """Format ground truth value for Python code.

    Args:
        value: The ground truth value to format

    Returns:
        Python representation string
    """
    if isinstance(value, str):
        return repr(value)  # Handles quotes and escaping
    elif isinstance(value, bool | int | float):
        return str(value)
    elif isinstance(value, list | dict):
        return repr(value)
    else:
        return repr(value)


def generate_pydantic_class(
    spec_dict: dict[str, Any], class_name: str = "Answer", float_tolerance: float = 0.001
) -> str:
    """Generate a Pydantic class from structured generator output.

    Args:
        spec_dict: Dictionary with "attributes" and "field_descriptions" keys
        class_name: Name for the generated class (default: "Answer")
        float_tolerance: Tolerance for float comparisons in verify()

    Returns:
        Python source code string for the Pydantic class
    """
    attributes = spec_dict["attributes"]
    field_descriptions = spec_dict["field_descriptions"]

    # Start building the class
    lines = []

    # Class definition (no imports - they'll be added by existing system)
    lines.append(f"class {class_name}(BaseAnswer):")

    # Field definitions
    for attr in attributes:
        attr_name = attr["name"]
        attr_type = attr["type"]
        description = field_descriptions[attr_name]

        # Convert type to proper annotation
        type_annotation = python_type_to_annotation(attr_type)

        # Add field definition
        field_def = f'    {attr_name}: {type_annotation} = Field(description="{description}")'
        lines.append(field_def)

    lines.append("")

    # model_post_init method
    lines.append("    def model_post_init(self, __context):")

    # Build correct dictionary
    correct_dict_items = []
    for attr in attributes:
        attr_name = attr["name"]
        ground_truth_value = format_ground_truth_value(attr["ground_truth"])
        correct_dict_items.append(f'"{attr_name}": {ground_truth_value}')

    correct_dict = "{" + ", ".join(correct_dict_items) + "}"
    lines.append(f"        self.correct = {correct_dict}")
    lines.append("")

    # verify method
    lines.append("    def verify(self) -> bool:")
    if len(attributes) == 1:
        # Single attribute - simple check
        attr = attributes[0]
        verification_logic = generate_verification_logic(attr["name"], attr["type"], float_tolerance)
        lines.append(f"        return {verification_logic}")
    else:
        # Multiple attributes - all must pass
        lines.append("        return (")
        verification_conditions = []
        for attr in attributes:
            verification_logic = generate_verification_logic(attr["name"], attr["type"], float_tolerance)
            verification_conditions.append(f"            {verification_logic}")

        lines.append(" and\n".join(verification_conditions))
        lines.append("        )")

    lines.append("")

    # verify_granular method (only if multiple attributes)
    if len(attributes) > 1:
        lines.append("    def verify_granular(self) -> float:")
        lines.append("        correct_count = 0")
        lines.append("        total_count = " + str(len(attributes)))
        lines.append("")

        for attr in attributes:
            verification_logic = generate_verification_logic(attr["name"], attr["type"], float_tolerance)
            lines.append(f"        if {verification_logic}:")
            lines.append("            correct_count += 1")

        lines.append("")
        lines.append("        return correct_count / total_count")

    # Generate the class code
    return "\n".join(lines)
