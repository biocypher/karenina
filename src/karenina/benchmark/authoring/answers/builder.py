"""
AnswerBuilder class for programmatically creating Answer templates.

This module provides a fluent interface for building Answer templates that include
both classical Pydantic field attributes and regex validation patterns. The builder
compiles to an executable Answer class that can be used with QuestionManager.
"""

import re
from typing import Any, cast

from .generator import (
    GroundTruthField,
    _format_ground_truth_value,
    _generate_pydantic_class,
)


class AnswerBuilder:
    """Builder class for creating Answer templates programmatically.

    Provides a fluent interface for adding classical attributes (Pydantic fields)
    and regex validation patterns, then compiling to an executable Answer class.

    Example:
        builder = (AnswerBuilder()
            .add_attribute("mentions_drug", "bool", "Whether drug mentioned", True)
            .add_regex("citations", r"\\[\\d+\\]", expected=3, match_type="count"))
        Answer = builder.compile()
        benchmark.add_question("What is the dosage?", "500mg", answer_template=Answer)
    """

    def __init__(self) -> None:
        """Initialize an empty AnswerBuilder."""
        self.attributes: list[GroundTruthField] = []
        self.field_descriptions: dict[str, str] = {}
        self.regex_patterns: dict[str, dict[str, Any]] = {}
        self.regex_descriptions: dict[str, str] = {}

    def add_attribute(self, name: str, type_str: str, description: str, ground_truth: Any) -> "AnswerBuilder":
        """Add a classical Pydantic field attribute.

        Args:
            name: Field name (must be valid Python identifier)
            type_str: Type annotation string (e.g., "bool", "int", "List[str]")
            description: Human-readable description for the field
            ground_truth: Expected correct value for this attribute

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name already exists or is invalid
        """
        if not name.isidentifier():
            raise ValueError(f"Invalid field name '{name}': must be valid Python identifier")

        if any(attr.name == name for attr in self.attributes):
            raise ValueError(f"Attribute '{name}' already exists")

        if name in self.regex_patterns:
            raise ValueError(f"Name '{name}' already used for regex pattern")

        # Create GroundTruthField object
        attribute = GroundTruthField(name=name, type=type_str, ground_truth=ground_truth)

        self.attributes.append(attribute)
        self.field_descriptions[name] = description

        return self

    def remove_attribute(self, name: str) -> "AnswerBuilder":
        """Remove a classical attribute by name.

        Args:
            name: Name of the attribute to remove

        Returns:
            Self for method chaining

        Raises:
            ValueError: If attribute doesn't exist
        """
        # Find and remove attribute
        for i, attr in enumerate(self.attributes):
            if attr.name == name:
                self.attributes.pop(i)
                self.field_descriptions.pop(name, None)
                return self

        raise ValueError(f"Attribute '{name}' not found")

    def add_regex(
        self, name: str, pattern: str, expected: Any, match_type: str = "exact", description: str = ""
    ) -> "AnswerBuilder":
        """Add a regex validation pattern.

        Args:
            name: Pattern name (must be valid Python identifier)
            pattern: Regular expression pattern string
            expected: Expected result (varies by match_type)
            match_type: Type of matching - "exact", "contains", "count", "all"
            description: Human-readable description

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name already exists or pattern is invalid
        """
        if not name.isidentifier():
            raise ValueError(f"Invalid pattern name '{name}': must be valid Python identifier")

        if name in self.regex_patterns:
            raise ValueError(f"Regex pattern '{name}' already exists")

        if any(attr.name == name for attr in self.attributes):
            raise ValueError(f"Name '{name}' already used for attribute")

        # Validate regex pattern
        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

        # Validate match_type
        valid_match_types = {"exact", "contains", "count", "all"}
        if match_type not in valid_match_types:
            raise ValueError(f"Invalid match_type '{match_type}'. Must be one of: {valid_match_types}")

        self.regex_patterns[name] = {"pattern": pattern, "expected": expected, "match_type": match_type}

        if description:
            self.regex_descriptions[name] = description

        return self

    def remove_regex(self, name: str) -> "AnswerBuilder":
        """Remove a regex pattern by name.

        Args:
            name: Name of the regex pattern to remove

        Returns:
            Self for method chaining

        Raises:
            ValueError: If regex pattern doesn't exist
        """
        if name not in self.regex_patterns:
            raise ValueError(f"Regex pattern '{name}' not found")

        del self.regex_patterns[name]
        self.regex_descriptions.pop(name, None)

        return self

    def compile(self, class_name: str = "Answer") -> type[Any]:
        """Compile the builder into an executable Answer class.

        Args:
            class_name: Name for the generated class (default: "Answer")

        Returns:
            Executable Answer class ready for use with QuestionManager

        Raises:
            ValueError: If no attributes are defined or compilation fails
        """
        if not self.attributes and not self.regex_patterns:
            raise ValueError("Cannot compile empty AnswerBuilder: no attributes or regex patterns defined")

        # Build spec dictionary for _generate_pydantic_class
        spec = {
            "attributes": [attr.model_dump() for attr in self.attributes],
            "field_descriptions": self.field_descriptions.copy(),
        }

        # Generate base class code
        if self.attributes:
            base_code = _generate_pydantic_class(spec, class_name)
        else:
            # No classical attributes, create minimal class
            base_code = self._generate_minimal_class(class_name)

        # Modify code to include regex support if needed
        final_code = self._add_regex_support(base_code, class_name) if self.regex_patterns else base_code

        # Execute the code to create the class
        local_ns: dict[str, Any] = {}

        # Add imports to the code
        imports = "from karenina.schemas.entities import BaseAnswer\nfrom pydantic import Field\n\n"
        final_code = imports + final_code

        try:
            # Import the required modules
            from typing import Literal, Union

            from pydantic import Field

            from karenina.schemas.entities import BaseAnswer

            # Set up globals with necessary imports
            globals_dict = {
                "BaseAnswer": BaseAnswer,
                "Field": Field,
                "List": list,
                "Dict": dict,
                "Literal": Literal,
                "Union": Union,
                "__builtins__": __builtins__,
            }

            exec(final_code, globals_dict, local_ns)
            Answer = local_ns[class_name]
        except Exception as e:
            raise ValueError(f"Failed to compile Answer class: {e}") from e

        # Store source code on the class (critical for QuestionManager)
        Answer._source_code = final_code

        return cast(type[Any], Answer)

    def _generate_minimal_class(self, class_name: str) -> str:
        """Generate minimal Answer class when only regex patterns exist."""
        regex_dict = self._build_regex_dict()

        return f"""class {class_name}(BaseAnswer):
    regex: dict = Field(default_factory=dict, description="Regex validation patterns")

    def model_post_init(self, __context):
        self.correct = {{}}
        self.regex = {regex_dict}

    def verify(self) -> bool:
        return True  # Only regex validation
"""

    def _add_regex_support(self, base_code: str, class_name: str) -> str:
        """Add regex field and initialization to existing class code."""
        lines = base_code.split("\n")
        modified_lines = []
        in_class = False
        in_model_post_init = False
        model_post_init_indent = ""
        added_regex_field = False

        regex_dict = self._build_regex_dict()

        for line in lines:
            if f"class {class_name}(BaseAnswer):" in line:
                in_class = True
                modified_lines.append(line)
                continue

            # Add regex field after other Field definitions
            if (
                in_class
                and not added_regex_field
                and line.strip()
                and not line.startswith("    ")
                and not line.strip().startswith("#")
            ):
                # We've left the field definitions area
                modified_lines.append(
                    '    regex: dict = Field(default_factory=dict, description="Regex validation patterns")'
                )
                modified_lines.append("")
                added_regex_field = True

            if "def model_post_init(self, __context):" in line:
                in_model_post_init = True
                model_post_init_indent = line[: len(line) - len(line.lstrip())]
                modified_lines.append(line)
                continue

            if in_model_post_init and line.strip() == "" and len(modified_lines) > 0:
                # Add regex initialization before empty line in model_post_init
                modified_lines.append(line)
                modified_lines.append(f"{model_post_init_indent}    self.regex = {regex_dict}")
                in_model_post_init = False
                continue

            modified_lines.append(line)

        # If we never added the regex field, add it before the def model_post_init
        if not added_regex_field:
            for i, line in enumerate(modified_lines):
                if "def model_post_init(self, __context):" in line:
                    modified_lines.insert(
                        i, '    regex: dict = Field(default_factory=dict, description="Regex validation patterns")'
                    )
                    modified_lines.insert(i + 1, "")
                    break

        return "\n".join(modified_lines)

    def _build_regex_dict(self) -> str:
        """Build the regex dictionary string for code generation."""
        if not self.regex_patterns:
            return "{}"

        items = []
        for name, spec in self.regex_patterns.items():
            # Use repr for pattern to properly escape it for code generation
            pattern = repr(spec["pattern"])
            expected = _format_ground_truth_value(spec["expected"])
            match_type = repr(spec["match_type"])

            items.append(f'"{name}": {{"pattern": {pattern}, "expected": {expected}, "match_type": {match_type}}}')

        return "{" + ", ".join(items) + "}"

    def __repr__(self) -> str:
        """Display current builder state with attributes and regex patterns."""
        lines = ["AnswerBuilder:"]

        # Classical attributes
        lines.append(f"  Classical Attributes ({len(self.attributes)}):")
        if not self.attributes:
            lines.append("    (none)")
        else:
            for attr in self.attributes:
                ground_truth_str = _format_ground_truth_value(attr.ground_truth)
                lines.append(f"    - {attr.name}: {attr.type} = {ground_truth_str}")
                description = self.field_descriptions.get(attr.name, "")
                if description:
                    lines.append(f"      {description}")

        # Regex patterns
        lines.append(f"  Regex Patterns ({len(self.regex_patterns)}):")
        if not self.regex_patterns:
            lines.append("    (none)")
        else:
            for name, spec in self.regex_patterns.items():
                expected_str = _format_ground_truth_value(spec["expected"])
                lines.append(f"    - {name}: {spec['pattern']} ({spec['match_type']}, expected={expected_str})")
                description = self.regex_descriptions.get(name, "")
                if description:
                    lines.append(f"      {description}")

        return "\n".join(lines)
