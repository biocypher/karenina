"""Structured representation of VerifiedField answer templates.

This module defines the TemplateSpec model, a JSON-serializable representation
of a VerifiedField-based answer template. It serves as the interchange format
between the Python template code (used by the pipeline) and the visual template
builder GUI.

The bidirectional converter in template_converter.py translates between
Python source code and TemplateSpec JSON.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TemplateFieldSpec(BaseModel):
    """Specification for a single template field.

    Maps to a VerifiedField declaration in the Python template code.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Python identifier for the field.")
    type: str = Field(
        ...,
        description=("Field type: 'bool', 'str', 'int', 'float', 'list_str', 'literal', 'date'."),
    )
    description: str = Field(..., description="Field description for the judge LLM.")
    extraction_hint: str | None = Field(
        None,
        description="Optional hint about normalization or formatting for the judge.",
    )
    ground_truth: Any = Field(..., description="Expected correct value.")
    literal_values: list[str] | None = Field(
        None,
        description="Allowed values when type is 'literal'.",
    )
    verify_with: dict[str, Any] = Field(
        ...,
        description=(
            "Serialized verification primitive. Must include a 'type' key matching a registered primitive name."
        ),
    )
    weight: float = Field(1.0, description="Weight for verify_granular() scoring.")
    is_trace: bool = Field(
        False,
        description=("If True, this field uses a trace primitive and is excluded from judge parsing."),
    )


class VerifyStrategySpec(BaseModel):
    """Specification for a composition strategy node.

    Maps to AllOf, AnyOf, AtLeastN, or FieldCheck in the Python template.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        ...,
        description="Strategy type: 'all_of', 'any_of', 'at_least_n', 'field_check'.",
    )
    n: int | None = Field(None, description="Required count for 'at_least_n'.")
    field_name: str | None = Field(None, description="Field name for 'field_check' leaves.")
    conditions: list["VerifyStrategySpec"] = Field(
        default_factory=list,
        description="Child conditions for composite strategies.",
    )


class TemplateSpec(BaseModel):
    """Complete specification for a VerifiedField answer template.

    This is the JSON interchange format between the visual builder GUI
    and the Python template code. When verify_strategy is None, the
    default AllOf-all-fields strategy is used.
    """

    model_config = ConfigDict(extra="forbid")

    fields: list[TemplateFieldSpec] = Field(..., description="Ordered list of template fields.")
    verify_strategy: VerifyStrategySpec | None = Field(
        None,
        description=("Custom composition strategy. None means default: AllOf with all fields."),
    )
    class_name: str = Field(
        "Answer",
        description="Name for the generated Python class.",
    )


# Rebuild for forward references
VerifyStrategySpec.model_rebuild()
