"""Unit tests for ADeLe rubric parser."""

import pytest

from karenina.integrations.adele.parser import (
    AdeleLevel,
    AdeleRubric,
    parse_adele_file,
)


class TestAdeleLevel:
    """Tests for AdeleLevel dataclass."""

    def test_to_class_description_with_examples(self) -> None:
        """Test formatting level with examples."""
        level = AdeleLevel(
            index=0,
            label="None",
            description="No attention required.",
            examples=["Example 1", "Example 2"],
        )
        result = level.to_class_description()

        assert "Level 0: None. No attention required." in result
        assert "Examples:" in result
        assert "* Example 1" in result
        assert "* Example 2" in result

    def test_to_class_description_without_examples(self) -> None:
        """Test formatting level without examples."""
        level = AdeleLevel(
            index=3,
            label="Intermediate",
            description="Moderate attention needed.",
            examples=[],
        )
        result = level.to_class_description()

        assert result == "Level 3: Intermediate. Moderate attention needed."
        assert "Examples:" not in result


class TestAdeleRubric:
    """Tests for AdeleRubric dataclass."""

    def test_valid_rubric(self) -> None:
        """Test creating a valid rubric with 6 levels."""
        levels = [AdeleLevel(index=i, label=f"Level{i}", description=f"Desc {i}") for i in range(6)]
        rubric = AdeleRubric(code="TEST", header="Test header", levels=levels)

        assert rubric.code == "TEST"
        assert rubric.header == "Test header"
        assert len(rubric.levels) == 6

    def test_invalid_level_count(self) -> None:
        """Test that rubric requires exactly 6 levels."""
        levels = [
            AdeleLevel(index=i, label=f"Level{i}", description=f"Desc {i}")
            for i in range(5)  # Only 5 levels
        ]

        with pytest.raises(ValueError, match="exactly 6 levels"):
            AdeleRubric(code="TEST", header=None, levels=levels)

    def test_invalid_level_indices(self) -> None:
        """Test that level indices must match their position."""
        levels = [AdeleLevel(index=i, label=f"Level{i}", description=f"Desc {i}") for i in range(6)]
        levels[3].index = 4  # Wrong index

        with pytest.raises(ValueError, match="index 4, expected 3"):
            AdeleRubric(code="TEST", header=None, levels=levels)


class TestParseAdeleFile:
    """Tests for parse_adele_file function."""

    def test_parse_rubric_with_header(self) -> None:
        """Test parsing rubric with header paragraph."""
        content = """This is the header paragraph describing the rubric.

Level 0: None. No requirement.
* Example for level 0

Level 1: Very Low. Minimal requirement.
* Example for level 1

Level 2: Low. Some requirement.
* Example for level 2

Level 3: Intermediate. Moderate requirement.
* Example for level 3

Level 4: High. Significant requirement.
* Example for level 4

Level 5: Very High. Maximum requirement.
* Example for level 5
"""
        rubric = parse_adele_file(content, "TEST")

        assert rubric.code == "TEST"
        assert rubric.header == "This is the header paragraph describing the rubric."
        assert len(rubric.levels) == 6

        # Check first level
        assert rubric.levels[0].index == 0
        assert rubric.levels[0].label == "None"
        assert rubric.levels[0].description == "No requirement."
        assert rubric.levels[0].examples == ["Example for level 0"]

        # Check last level
        assert rubric.levels[5].index == 5
        assert rubric.levels[5].label == "Very High"
        assert rubric.levels[5].description == "Maximum requirement."

    def test_parse_rubric_without_header(self) -> None:
        """Test parsing rubric without header (like AT.txt)."""
        content = """Level 0: None. The task is a staple one.
* Example 1

Level 1: Very Low. The task is common.
* Example 2

Level 2: Low. The task is moderately common.
* Example 3

Level 3: Intermediate. The task is somewhat common.
* Example 4

Level 4: High. The task is not common.
* Example 5

Level 5: Very High. The task is rare.
* Example 6
"""
        rubric = parse_adele_file(content, "AT")

        assert rubric.code == "AT"
        assert rubric.header is None
        assert len(rubric.levels) == 6

    def test_parse_multiline_examples(self) -> None:
        """Test parsing examples that span multiple lines."""
        content = """Level 0: None. Simple.
* Single line example

Level 1: Very Low. Still simple.
* Multi line example
that continues here

Level 2: Low. Getting there.
* Another example

Level 3: Intermediate. Moderate.
* Third example

Level 4: High. Complex.
* Fourth example

Level 5: Very High. Maximum.
* Fifth example
"""
        rubric = parse_adele_file(content, "TEST")

        # Level 1 should have the multi-line example joined
        assert "Multi line example that continues here" in rubric.levels[1].examples[0]

    def test_parse_multiple_examples_per_level(self) -> None:
        """Test parsing levels with multiple examples."""
        content = """Level 0: None. Simple.
* First example
* Second example
* Third example

Level 1: Very Low. Basic.
* Single example

Level 2: Low. Some.
* Example

Level 3: Intermediate. Moderate.
* Example

Level 4: High. Complex.
* Example

Level 5: Very High. Maximum.
* Example
"""
        rubric = parse_adele_file(content, "TEST")

        assert len(rubric.levels[0].examples) == 3
        assert rubric.levels[0].examples[0] == "First example"
        assert rubric.levels[0].examples[1] == "Second example"
        assert rubric.levels[0].examples[2] == "Third example"

    def test_parse_description_continuation(self) -> None:
        """Test parsing descriptions that continue after the header line."""
        content = """Level 0: None. No attention or scan is required.
The target information is immediately obvious.

Level 1: Very Low. Minimal.
Examples:
* Example

Level 2: Low. Some.
* Example

Level 3: Intermediate. Moderate.
* Example

Level 4: High. Significant.
* Example

Level 5: Very High. Maximum.
* Example
"""
        rubric = parse_adele_file(content, "TEST")

        # Description should include continuation text
        assert "No attention or scan is required." in rubric.levels[0].description
        assert "immediately obvious" in rubric.levels[0].description

    def test_parse_invalid_level_count(self) -> None:
        """Test that parsing fails with wrong number of levels."""
        content = """Level 0: None. Simple.
Level 1: Very Low. Basic.
Level 2: Low. Some.
"""
        with pytest.raises(ValueError, match="Expected 6 levels"):
            parse_adele_file(content, "TEST")

    def test_parse_handles_crlf_line_endings(self) -> None:
        """Test that parser handles Windows-style line endings."""
        content = "Level 0: None. Simple.\r\n* Example\r\n\r\nLevel 1: Very Low. Basic.\r\n* Example\r\n\r\nLevel 2: Low. Some.\r\n* Example\r\n\r\nLevel 3: Intermediate. Moderate.\r\n* Example\r\n\r\nLevel 4: High. Complex.\r\n* Example\r\n\r\nLevel 5: Very High. Maximum.\r\n* Example\r\n"
        rubric = parse_adele_file(content, "TEST")

        assert len(rubric.levels) == 6


class TestParseRealAdeleFiles:
    """Integration tests using actual bundled ADeLe files."""

    def test_parse_as_file(self) -> None:
        """Test parsing the AS (Attention and Scan) rubric file."""
        from importlib import resources

        ref = resources.files("karenina.integrations.adele.data").joinpath("AS.txt")
        content = ref.read_text(encoding="utf-8")

        rubric = parse_adele_file(content, "AS")

        assert rubric.code == "AS"
        assert rubric.header is not None
        assert "attention and scan" in rubric.header.lower()
        assert len(rubric.levels) == 6

        # Verify level labels
        assert rubric.levels[0].label == "None"
        assert rubric.levels[1].label == "Very low"
        assert rubric.levels[2].label == "Low"
        assert rubric.levels[3].label == "Intermediate"
        assert rubric.levels[4].label == "High"
        assert rubric.levels[5].label == "Very high"

    def test_parse_at_file_no_header(self) -> None:
        """Test parsing the AT (Atypicality) rubric file which has no header."""
        from importlib import resources

        ref = resources.files("karenina.integrations.adele.data").joinpath("AT.txt")
        content = ref.read_text(encoding="utf-8")

        rubric = parse_adele_file(content, "AT")

        assert rubric.code == "AT"
        assert rubric.header is None  # AT has no header paragraph
        assert len(rubric.levels) == 6

    def test_parse_all_bundled_files(self) -> None:
        """Test that all 18 bundled ADeLe files parse successfully."""
        from importlib import resources

        codes = [
            "AS",
            "AT",
            "CEc",
            "CEe",
            "CL",
            "KNa",
            "KNc",
            "KNf",
            "KNn",
            "KNs",
            "MCr",
            "MCt",
            "MCu",
            "MS",
            "QLl",
            "QLq",
            "SNs",
            "VO",
        ]

        for code in codes:
            ref = resources.files("karenina.integrations.adele.data").joinpath(f"{code}.txt")
            content = ref.read_text(encoding="utf-8")
            rubric = parse_adele_file(content, code)

            assert rubric.code == code, f"Failed for {code}"
            assert len(rubric.levels) == 6, f"Wrong level count for {code}"

            # Each level should have description and at least some examples
            for i, level in enumerate(rubric.levels):
                assert level.index == i, f"Wrong index for {code} level {i}"
                assert level.label, f"Missing label for {code} level {i}"
                assert level.description, f"Missing description for {code} level {i}"
