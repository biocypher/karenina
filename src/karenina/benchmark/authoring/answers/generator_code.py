"""Pydantic class code generation for answer templates.

This module contains functions for generating Python/Pydantic code
from structured ground truth specifications. Generated code uses
VerifiedField-based templates where each field carries its own ground
truth and verification primitive, eliminating the need for manual
ground_truth(), verify(), and verify_granular() methods.
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
        return f"list[{python_type_to_annotation(inner_type)}]"

    # Handle list types (lowercase)
    if type_str.startswith("list["):
        inner_type = type_str[5:-1]
        return f"list[{python_type_to_annotation(inner_type)}]"

    # Handle Dict types
    if type_str.startswith("Dict["):
        # Extract key, value types from Dict[key, value]
        inner = type_str[5:-1]
        key_type, value_type = inner.split(", ")
        return f"dict[{python_type_to_annotation(key_type)}, {python_type_to_annotation(value_type)}]"

    # Handle dict types (lowercase)
    if type_str.startswith("dict["):
        inner = type_str[5:-1]
        key_type, value_type = inner.split(", ")
        return f"dict[{python_type_to_annotation(key_type)}, {python_type_to_annotation(value_type)}]"

    return type_mapping.get(type_str, type_str)


def _type_to_primitive(attr_type: str, float_tolerance: float = 0.001) -> tuple[str, str]:
    """Map a type annotation string to a verification primitive code snippet.

    Args:
        attr_type: Type string like "bool", "str", "int", "float", "List[str]", "Literal[...]"
        float_tolerance: Tolerance value used for NumericTolerance primitives.

    Returns:
        Tuple of (primitive_class_name, primitive_instantiation_code). The class
        name is used to track which primitives need to be imported.
    """
    if attr_type == "bool":
        return "BooleanMatch", "BooleanMatch()"
    elif attr_type == "str":
        return "ExactMatch", 'ExactMatch(normalize=["lowercase", "strip"])'
    elif attr_type == "int":
        return "NumericExact", "NumericExact()"
    elif attr_type == "float":
        return "NumericTolerance", f'NumericTolerance(tolerance={float_tolerance}, mode="relative")'
    elif attr_type in ("List[str]", "list[str]"):
        return "SetContainment", 'SetContainment(mode="exact")'
    elif attr_type.startswith("Literal"):
        return "LiteralMatch", "LiteralMatch()"
    else:
        # Default to ExactMatch for unknown types
        return "ExactMatch", 'ExactMatch(normalize=["lowercase", "strip"])'


def generate_verification_logic(attr_name: str, attr_type: str, tolerance: float = 0.001) -> str:
    """Generate verification logic for a specific attribute type.

    This function is retained for backward compatibility. New code generation
    uses VerifiedField primitives instead of manual verify() methods.

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

    This function is retained for backward compatibility and is used by
    builder.py for regex pattern formatting.

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
    """Generate a Pydantic class using VerifiedField-based verification.

    Generates a class where each field uses VerifiedField with an appropriate
    verification primitive. The resulting class does not need manual
    ground_truth(), verify(), or verify_granular() methods because
    BaseAnswer auto-generates them from the VerifiedField metadata.

    Args:
        spec_dict: Dictionary with "attributes" and "field_descriptions" keys.
            Each attribute entry must have "name", "type", and "ground_truth" keys.
        class_name: Name for the generated class (default: "Answer")
        float_tolerance: Tolerance for NumericTolerance primitives (default: 0.001)

    Returns:
        Python source code string for the Pydantic class, including the
        import line for all used primitives.
    """
    attributes = spec_dict["attributes"]
    field_descriptions = spec_dict["field_descriptions"]

    # Track which primitive class names are used so we can generate a minimal import
    used_primitives: list[str] = []

    # Build field lines and collect primitive names
    field_lines: list[str] = []
    for attr in attributes:
        attr_name = attr["name"]
        attr_type = attr["type"]
        description = field_descriptions[attr_name]
        ground_truth_repr = format_ground_truth_value(attr["ground_truth"])

        # Convert type to proper annotation
        type_annotation = python_type_to_annotation(attr_type)

        # Choose appropriate primitive
        primitive_cls_name, primitive_code = _type_to_primitive(attr_type, float_tolerance)
        if primitive_cls_name not in used_primitives:
            used_primitives.append(primitive_cls_name)

        # Generate VerifiedField call
        field_lines.append(f"    {attr_name}: {type_annotation} = VerifiedField(")
        field_lines.append(f'        description="{description}",')
        field_lines.append(f"        ground_truth={ground_truth_repr},")
        field_lines.append(f"        verify_with={primitive_code},")
        field_lines.append("    )")

    # Build import line: BaseAnswer and VerifiedField are always needed,
    # plus all used primitives in sorted order for determinism
    sorted_primitives = sorted(used_primitives)
    imports_parts = ["BaseAnswer", "VerifiedField"] + sorted_primitives
    import_line = f"from karenina.schemas.entities import {', '.join(imports_parts)}"

    # Assemble final class code
    lines: list[str] = []
    lines.append(import_line)
    lines.append("")
    lines.append("")
    lines.append(f"class {class_name}(BaseAnswer):")
    lines.extend(field_lines)

    return "\n".join(lines)
