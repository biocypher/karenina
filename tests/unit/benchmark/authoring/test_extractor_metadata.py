"""Tests for extract_questions_from_file metadata population."""

import pytest

from karenina.benchmark.authoring.questions.extractor import extract_questions_from_file


@pytest.fixture
def csv_with_metadata(tmp_path):
    """CSV with author, url, keywords, answer_notes, and custom columns."""
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text(
        "Question,Answer,AuthorName,AuthorEmail,Affiliation,URL,Area,Notes,Complexity\n"
        "What is X?,X is Y,Alice,alice@example.com,MIT,http://example.com,Biology,Tricky,1.0\n"
        "What is Z?,Z is W,Bob,,Harvard,,Chemistry,,2.0\n"
        "What is Q?,Q is R,,,,,Physics,,\n"
    )
    return str(csv_path)


@pytest.mark.unit
class TestAuthorExtraction:
    def test_author_populated_from_columns(self, csv_with_metadata):
        questions = extract_questions_from_file(
            file_path=csv_with_metadata,
            question_column="Question",
            answer_column="Answer",
            author_name_column="AuthorName",
            author_email_column="AuthorEmail",
            author_affiliation_column="Affiliation",
        )
        assert len(questions) == 3

        # Full author
        assert questions[0].author == {
            "@type": "Person",
            "name": "Alice",
            "email": "alice@example.com",
            "affiliation": "MIT",
        }

        # Partial author (no email)
        assert questions[1].author == {
            "@type": "Person",
            "name": "Bob",
            "affiliation": "Harvard",
        }

        # No author data at all
        assert questions[2].author is None

    def test_author_none_when_no_columns_specified(self, csv_with_metadata):
        questions = extract_questions_from_file(
            file_path=csv_with_metadata,
            question_column="Question",
            answer_column="Answer",
        )
        for q in questions:
            assert q.author is None


@pytest.mark.unit
class TestCustomMetadataExtraction:
    def test_custom_metadata_from_columns(self, csv_with_metadata):
        questions = extract_questions_from_file(
            file_path=csv_with_metadata,
            question_column="Question",
            answer_column="Answer",
            custom_metadata_columns=["Complexity"],
        )
        assert questions[0].custom_metadata == {"Complexity": 1.0}
        assert questions[1].custom_metadata == {"Complexity": 2.0}
        # Row 3 has no Complexity value - NaN should be excluded
        assert questions[2].custom_metadata is None

    def test_custom_metadata_preserves_pandas_types(self, csv_with_metadata):
        questions = extract_questions_from_file(
            file_path=csv_with_metadata,
            question_column="Question",
            answer_column="Answer",
            custom_metadata_columns=["Complexity", "Area"],
        )
        # Complexity is float (pandas inferred), Area is string
        assert questions[0].custom_metadata == {"Complexity": 1.0, "Area": "Biology"}
        assert isinstance(questions[0].custom_metadata["Complexity"], float)
        assert isinstance(questions[0].custom_metadata["Area"], str)

    def test_url_column_goes_to_custom_metadata(self, csv_with_metadata):
        questions = extract_questions_from_file(
            file_path=csv_with_metadata,
            question_column="Question",
            answer_column="Answer",
            url_column="URL",
        )
        assert questions[0].custom_metadata == {"url": "http://example.com"}
        # Row 2 has no URL
        assert questions[1].custom_metadata is None

    def test_url_and_custom_metadata_merge(self, csv_with_metadata):
        questions = extract_questions_from_file(
            file_path=csv_with_metadata,
            question_column="Question",
            answer_column="Answer",
            url_column="URL",
            custom_metadata_columns=["Complexity"],
        )
        assert questions[0].custom_metadata == {
            "url": "http://example.com",
            "Complexity": 1.0,
        }

    def test_no_custom_metadata_when_not_specified(self, csv_with_metadata):
        questions = extract_questions_from_file(
            file_path=csv_with_metadata,
            question_column="Question",
            answer_column="Answer",
        )
        for q in questions:
            assert q.custom_metadata is None
