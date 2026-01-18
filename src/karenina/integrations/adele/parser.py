"""
Parser for ADeLe (Assessment Dimensions for Language Evaluation) rubric text files.

ADeLe rubrics define 6 levels (0-5) for evaluating various cognitive and processing dimensions.
Each level has a label, description, and examples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class AdeleLevel:
    """A single level in an ADeLe rubric (0-5)."""

    index: int
    label: str
    description: str
    examples: list[str] = field(default_factory=list)

    def to_class_description(self) -> str:
        """Format level as a single class description string for LLMRubricTrait.

        Format: "Level N: Label. Description\nExamples:\n* example1\n* example2"
        """
        parts = [f"Level {self.index}: {self.label}. {self.description}"]
        if self.examples:
            parts.append("Examples:")
            for example in self.examples:
                parts.append(f"* {example}")
        return "\n".join(parts)


@dataclass
class AdeleRubric:
    """A parsed ADeLe rubric with optional header and 6 levels."""

    code: str
    header: str | None
    levels: list[AdeleLevel]

    def __post_init__(self) -> None:
        """Validate rubric structure."""
        if len(self.levels) != 6:
            raise ValueError(f"ADeLe rubric must have exactly 6 levels, got {len(self.levels)}")

        for i, level in enumerate(self.levels):
            if level.index != i:
                raise ValueError(f"Level at position {i} has index {level.index}, expected {i}")


# Regex to match level headers in two formats:
# Format 1: "Level 0: None. Description..." (AS, AT, CEc, etc.)
# Format 2: "Level 0. None: Description..." (KNa, KNc, KNf, etc.)
LEVEL_PATTERN = re.compile(
    r"^Level\s+(\d+)[.:]\s+([^.:]+)[.:]\s*(.*)$",
    re.MULTILINE,
)


def parse_adele_file(content: str, code: str) -> AdeleRubric:
    """Parse ADeLe rubric text content into structured format.

    Args:
        content: Raw text content of the rubric file
        code: File code/identifier (e.g., "AS", "AT", "CEc")

    Returns:
        Parsed AdeleRubric with header (if present) and 6 levels

    Raises:
        ValueError: If parsing fails or rubric structure is invalid
    """
    # Normalize line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Find all level markers
    level_matches = list(LEVEL_PATTERN.finditer(content))

    if len(level_matches) != 6:
        raise ValueError(f"Expected 6 levels in rubric {code}, found {len(level_matches)}")

    # Extract header (content before Level 0)
    first_level_start = level_matches[0].start()
    header_content = content[:first_level_start].strip()
    header = header_content if header_content else None

    # Parse each level
    levels: list[AdeleLevel] = []

    for i, match in enumerate(level_matches):
        level_index = int(match.group(1))
        label = match.group(2).strip()
        first_line_desc = match.group(3).strip()

        # Determine where this level's content ends
        level_end = level_matches[i + 1].start() if i + 1 < len(level_matches) else len(content)

        # Extract full level content (after the matched line)
        level_content = content[match.end() : level_end].strip()

        # Combine first line description with continuation
        full_description, examples = _parse_level_content(first_line_desc, level_content)

        levels.append(
            AdeleLevel(
                index=level_index,
                label=label,
                description=full_description,
                examples=examples,
            )
        )

    return AdeleRubric(code=code, header=header, levels=levels)


def _parse_level_content(first_line_desc: str, continuation: str) -> tuple[str, list[str]]:
    """Parse level content into description and examples.

    Args:
        first_line_desc: Description text from the level header line
        continuation: Remaining content after the level header line

    Returns:
        Tuple of (full_description, list_of_examples)
    """
    if not continuation:
        return first_line_desc, []

    lines = continuation.split("\n")
    description_parts = [first_line_desc] if first_line_desc else []
    examples: list[str] = []
    in_examples = False
    current_example_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            # Empty line - might end current example if in examples section
            if current_example_lines:
                examples.append(" ".join(current_example_lines))
                current_example_lines = []
            continue

        # Check if this line starts an example (bullet point)
        if stripped.startswith("* "):
            in_examples = True
            # Save previous example if exists
            if current_example_lines:
                examples.append(" ".join(current_example_lines))
                current_example_lines = []
            # Start new example (remove the "* " prefix)
            current_example_lines = [stripped[2:]]
        elif stripped == "Examples:":
            # Explicit examples marker, switch to examples mode
            in_examples = True
        elif in_examples:
            # Continuation of current example (multi-line example)
            if current_example_lines:
                current_example_lines.append(stripped)
            else:
                # Orphan continuation line, treat as new example
                current_example_lines = [stripped]
        else:
            # Still in description section
            description_parts.append(stripped)

    # Don't forget last example
    if current_example_lines:
        examples.append(" ".join(current_example_lines))

    full_description = " ".join(description_parts)
    return full_description, examples
