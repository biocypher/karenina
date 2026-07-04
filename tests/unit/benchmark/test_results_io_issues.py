"""Tests for ResultsIOManager issues.

Covers:
- Issue 111: _escape_csv_field() double-escapes CSV values
"""

import csv
from io import StringIO

import pytest

from karenina.benchmark.core.results_io import ResultsIOManager
from karenina.schemas.verification.model_identity import ModelIdentity
from karenina.schemas.verification.result import VerificationResult
from karenina.schemas.verification.result_components import (
    VerificationResultMetadata,
    VerificationResultTemplate,
)


def _make_result(
    question_id: str = "q1",
    question_text: str = "What is 2+2?",
    raw_llm_response: str = "4",
) -> VerificationResult:
    """Create a minimal VerificationResult for testing CSV export."""
    answering = ModelIdentity(interface="langchain", model_name="gpt-4")
    parsing = ModelIdentity(interface="langchain", model_name="gpt-4")
    metadata = VerificationResultMetadata(
        question_id=question_id,
        template_id="tmpl_abc",
        failure=None,
        caveats=[],
        question_text=question_text,
        answering=answering,
        parsing=parsing,
        execution_time=1.0,
        timestamp="2026-01-01T00:00:00",
        result_id="abcdef1234567890",
    )
    template = VerificationResultTemplate(
        raw_llm_response=raw_llm_response,
        template_verification_performed=True,
        verify_result=True,
    )
    return VerificationResult(metadata=metadata, template=template)


@pytest.mark.unit
class TestIssue111DoubleEscapedCsvValues:
    """Issue 111: _escape_csv_field() manually escapes values before passing them
    to csv.writer, which also escapes values per RFC 4180. This causes
    double-escaping: a field containing a comma gets wrapped in quotes twice."""

    def test_csv_field_with_comma_not_double_escaped(self) -> None:
        """A question_text with a comma should not be double-quoted in CSV output."""
        result = _make_result(question_text='Hello, "world"')
        results = {"q1": result}

        csv_output = ResultsIOManager.export_to_csv(results)

        # Parse the CSV output back
        reader = csv.reader(StringIO(csv_output))
        rows = list(reader)

        # Row 0 is headers, row 1 is data
        assert len(rows) == 2
        header = rows[0]
        data = rows[1]

        # Find the question_text column
        qt_index = header.index("question_text")
        parsed_value = data[qt_index]

        # The parsed value should be exactly the original string,
        # not a double-escaped version
        assert parsed_value == 'Hello, "world"', (
            f"Expected 'Hello, \"world\"' but got '{parsed_value}'. "
            "The value was likely double-escaped by _escape_csv_field + csv.writer."
        )

    def test_csv_field_with_newline_not_double_escaped(self) -> None:
        """A field with a newline should not be double-escaped."""
        result = _make_result(raw_llm_response="line1\nline2")
        results = {"q1": result}

        csv_output = ResultsIOManager.export_to_csv(results)

        reader = csv.reader(StringIO(csv_output))
        rows = list(reader)

        header = rows[0]
        data = rows[1]

        raw_index = header.index("raw_llm_response")
        parsed_value = data[raw_index]

        assert parsed_value == "line1\nline2", (
            f"Expected 'line1\\nline2' but got '{parsed_value}'. Newlines were likely double-escaped."
        )

    def test_csv_field_with_quotes_not_double_escaped(self) -> None:
        """A field with double quotes should have them properly escaped once."""
        result = _make_result(question_text='She said "hello"')
        results = {"q1": result}

        csv_output = ResultsIOManager.export_to_csv(results)

        reader = csv.reader(StringIO(csv_output))
        rows = list(reader)

        header = rows[0]
        data = rows[1]

        qt_index = header.index("question_text")
        parsed_value = data[qt_index]

        assert parsed_value == 'She said "hello"', (
            f"Expected 'She said \"hello\"' but got '{parsed_value}'. Quotes were likely double-escaped."
        )

    def test_csv_plain_field_unchanged(self) -> None:
        """A plain text field without special characters should pass through cleanly."""
        result = _make_result(question_text="Simple question")
        results = {"q1": result}

        csv_output = ResultsIOManager.export_to_csv(results)

        reader = csv.reader(StringIO(csv_output))
        rows = list(reader)

        header = rows[0]
        data = rows[1]

        qt_index = header.index("question_text")
        parsed_value = data[qt_index]

        assert parsed_value == "Simple question"
