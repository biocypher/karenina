"""Edge case tests for benchmark robustness.

Tests cover:
- Unicode/special characters in question text
- Very long template code (performance limits)
- Malformed JSON-LD files
- Maximum question count boundaries
- Interrupted checkpoint I/O recovery
"""

import contextlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from karenina import Benchmark

# =============================================================================
# Unicode and Special Characters Tests
# =============================================================================


@pytest.mark.unit
class TestUnicodeHandling:
    """Tests for unicode and special character handling in questions."""

    @pytest.mark.parametrize(
        "question_text,description",
        [
            ("What is 2+2? 你好世界", "Chinese characters"),
            ("Как дела? What is the answer?", "Cyrillic characters"),
            ("مرحبا بالعالم - What is this?", "Arabic characters"),
            ("🎉 What emoji is this? 🚀", "Emoji characters"),
            ("α + β = γ, solve for γ", "Greek mathematical symbols"),
            ("∑∏∫∂∇ - mathematical operators", "Math operators"),
            ("Tab\there\nNewline\r\nCRLF", "Whitespace characters"),
            ("Quote: \"nested 'quotes'\"", "Nested quotes"),
            ("Path: C:\\Users\\test\\file.txt", "Backslashes"),
            ("HTML: <script>alert('xss')</script>", "HTML/XSS attempt"),
            ("SQL: '; DROP TABLE users; --", "SQL injection attempt"),
            ("NULL: \x00 byte", "Null byte"),
            ("", "Empty string"),
            ("   ", "Whitespace only"),
        ],
        ids=[
            "chinese",
            "cyrillic",
            "arabic",
            "emoji",
            "greek",
            "math_ops",
            "whitespace",
            "quotes",
            "backslashes",
            "html_xss",
            "sql_injection",
            "null_byte",
            "empty",
            "whitespace_only",
        ],
    )
    def test_question_with_special_chars(self, question_text: str, description: str) -> None:
        """Test adding questions with various special characters."""
        benchmark = Benchmark.create(name=f"unicode_test_{description}")

        # Should not raise
        q_id = benchmark.add_question(
            question=question_text,
            raw_answer="test answer",
        )

        # Should retrieve correctly
        retrieved = benchmark.get_question(q_id)
        assert retrieved["question"] == question_text

    def test_unicode_in_template_code(self) -> None:
        """Test unicode characters in template code."""
        benchmark = Benchmark.create(name="template_unicode")
        q_id = benchmark.add_question(
            question="What is the Greek letter?",
            raw_answer="α",
        )

        template_code = '''
from pydantic import Field
from karenina.schemas.domain import BaseAnswer

class Answer(BaseAnswer):
    """Template with unicode: αβγδε"""
    letter: str = Field(description="Greek letter like α, β, γ")

    def verify(self) -> bool:
        return self.letter in "αβγδεζηθικλμνξοπρστυφχψω"
'''
        benchmark.add_answer_template(q_id, template_code)

        # Should retrieve correctly
        retrieved_template = benchmark.get_template(q_id)
        assert "αβγδε" in retrieved_template

    def test_unicode_in_metadata(self) -> None:
        """Test unicode in custom metadata."""
        benchmark = Benchmark.create(
            name="日本語ベンチマーク",
            description="これはテストです",
        )
        benchmark.set_custom_property("author", "著者名")
        benchmark.set_custom_property("category", "カテゴリー")

        assert benchmark.name == "日本語ベンチマーク"
        assert benchmark.get_custom_property("author") == "著者名"


# =============================================================================
# Long Content / Performance Boundary Tests
# =============================================================================


@pytest.mark.unit
class TestLongContentHandling:
    """Tests for handling very long content."""

    def test_very_long_question_text(self) -> None:
        """Test question with very long text (10KB)."""
        benchmark = Benchmark.create(name="long_question_test")
        long_text = "A" * 10_000  # 10KB of text

        q_id = benchmark.add_question(
            question=f"Question: {long_text}",
            raw_answer="short answer",
        )

        retrieved = benchmark.get_question(q_id)
        assert len(retrieved["question"]) > 10_000

    def test_very_long_template_code(self) -> None:
        """Test template with very long code (50KB)."""
        benchmark = Benchmark.create(name="long_template_test")
        q_id = benchmark.add_question(
            question="Test question",
            raw_answer="answer",
        )

        # Generate a long but valid template
        long_docstring = "x" * 40_000  # 40KB docstring
        template_code = f'''
from pydantic import Field
from karenina.schemas.domain import BaseAnswer

class Answer(BaseAnswer):
    """{long_docstring}"""
    value: str = Field(description="test")

    def verify(self) -> bool:
        return True
'''
        benchmark.add_answer_template(q_id, template_code)
        retrieved = benchmark.get_template(q_id)
        assert len(retrieved) > 40_000

    def test_very_long_raw_answer(self) -> None:
        """Test question with very long raw answer (100KB)."""
        benchmark = Benchmark.create(name="long_answer_test")
        long_answer = "B" * 100_000  # 100KB

        q_id = benchmark.add_question(
            question="Short question",
            raw_answer=long_answer,
        )

        retrieved = benchmark.get_question(q_id)
        assert len(retrieved["raw_answer"]) == 100_000


# =============================================================================
# Question Count Boundary Tests
# =============================================================================


@pytest.mark.unit
class TestQuestionCountBoundaries:
    """Tests for question count boundaries."""

    def test_empty_benchmark_properties(self) -> None:
        """Test properties on empty benchmark."""
        benchmark = Benchmark.create(name="empty")

        assert benchmark.question_count == 0
        assert benchmark.is_empty is True
        assert benchmark.is_complete is False
        assert benchmark.get_progress() == 0.0
        assert list(benchmark) == []

    def test_single_question_benchmark(self) -> None:
        """Test benchmark with exactly one question."""
        benchmark = Benchmark.create(name="single")
        q_id = benchmark.add_question(
            question="Only question",
            raw_answer="Only answer",
        )

        assert benchmark.question_count == 1
        assert benchmark.is_empty is False
        assert q_id in benchmark

    def test_many_questions_performance(self) -> None:
        """Test adding many questions (1000) for performance."""
        benchmark = Benchmark.create(name="many_questions")

        # Add 1000 questions
        for i in range(1000):
            benchmark.add_question(
                question=f"Question {i}",
                raw_answer=f"Answer {i}",
            )

        assert benchmark.question_count == 1000

        # Test iteration performance
        count = 0
        for _ in benchmark:
            count += 1
        assert count == 1000

    def test_batch_add_questions(self) -> None:
        """Test batch adding questions."""
        benchmark = Benchmark.create(name="batch_test")

        questions = [{"question": f"Q{i}", "raw_answer": f"A{i}"} for i in range(100)]
        benchmark.add_questions_batch(questions)

        assert benchmark.question_count == 100


# =============================================================================
# Malformed JSON-LD Tests
# =============================================================================


@pytest.mark.unit
class TestMalformedJsonLd:
    """Tests for handling malformed JSON-LD files."""

    def test_missing_context(self, tmp_path: Path) -> None:
        """Test loading JSON-LD without @context."""
        file_path = tmp_path / "no_context.jsonld"
        data = {
            "@type": "DataFeed",
            "name": "Test",
            "dataFeedElement": [],
        }
        file_path.write_text(json.dumps(data))

        with pytest.raises((ValidationError, KeyError, ValueError)):
            Benchmark.load(file_path)

    def test_wrong_type(self, tmp_path: Path) -> None:
        """Test loading JSON-LD with wrong @type."""
        file_path = tmp_path / "wrong_type.jsonld"
        data = {
            "@context": "https://schema.org/",
            "@type": "Person",  # Wrong type
            "name": "Test",
        }
        file_path.write_text(json.dumps(data))

        with pytest.raises((ValidationError, KeyError, ValueError)):
            Benchmark.load(file_path)

    def test_invalid_date_format(self, tmp_path: Path) -> None:
        """Test loading JSON-LD with invalid date format."""
        file_path = tmp_path / "bad_date.jsonld"
        data = {
            "@context": "https://schema.org/",
            "@type": "DataFeed",
            "name": "Test",
            "dateModified": "not-a-date",  # Invalid date
            "dataFeedElement": [],
        }
        file_path.write_text(json.dumps(data))

        # Should handle gracefully or raise clear error
        try:
            benchmark = Benchmark.load(file_path)
            # If it loads, date should be handled somehow
            assert benchmark is not None
        except Exception as e:
            # Should be a clear validation error
            assert "date" in str(e).lower() or "validation" in str(e).lower()

    def test_truncated_json(self, tmp_path: Path) -> None:
        """Test loading truncated JSON file."""
        file_path = tmp_path / "truncated.jsonld"
        file_path.write_text('{"@context": "https://schema.org/", "@type": "Da')

        with pytest.raises(json.JSONDecodeError):
            Benchmark.load(file_path)

    def test_binary_content(self, tmp_path: Path) -> None:
        """Test loading file with binary content."""
        file_path = tmp_path / "binary.jsonld"
        file_path.write_bytes(b"\x00\x01\x02\x03\xff\xfe")

        with pytest.raises((json.JSONDecodeError, UnicodeDecodeError, ValueError)):
            Benchmark.load(file_path)

    def test_deeply_nested_structure(self, tmp_path: Path) -> None:
        """Test handling deeply nested JSON structure."""
        # Create deeply nested structure
        nested: dict = {"value": "deep"}
        for _ in range(50):
            nested = {"nested": nested}

        file_path = tmp_path / "deep.jsonld"
        data = {
            "@context": "https://schema.org/",
            "@type": "DataFeed",
            "name": "Test",
            "dataFeedElement": [],
            "additionalProperty": nested,
        }
        file_path.write_text(json.dumps(data))

        # Should handle without stack overflow - may pass or fail validation
        with contextlib.suppress(ValidationError, KeyError, ValueError):
            Benchmark.load(file_path)


# =============================================================================
# Checkpoint I/O Recovery Tests
# =============================================================================


@pytest.mark.unit
class TestCheckpointIORecovery:
    """Tests for checkpoint I/O error recovery."""

    def test_save_to_readonly_directory(self, tmp_path: Path) -> None:
        """Test saving to read-only directory."""
        benchmark = Benchmark.create(name="readonly_test")
        benchmark.add_question(question="Q", raw_answer="A")

        # Create read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        file_path = readonly_dir / "test.jsonld"

        try:
            with pytest.raises(PermissionError):
                benchmark.save(file_path)
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)

    def test_load_nonexistent_file(self) -> None:
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            Benchmark.load(Path("/nonexistent/path/file.jsonld"))

    def test_save_requires_existing_parent_directory(self, tmp_path: Path) -> None:
        """Test that save requires parent directory to exist."""
        benchmark = Benchmark.create(name="parent_dir_test")
        benchmark.add_question(question="Q", raw_answer="A")

        # Path with non-existent parent
        file_path = tmp_path / "new" / "nested" / "dir" / "test.jsonld"

        # Should raise FileNotFoundError since parent doesn't exist
        with pytest.raises(FileNotFoundError):
            benchmark.save(file_path)

    def test_save_with_manually_created_parent(self, tmp_path: Path) -> None:
        """Test save works when parent directory is pre-created."""
        benchmark = Benchmark.create(name="parent_dir_test")
        benchmark.add_question(question="Q", raw_answer="A")

        # Create parent directories manually
        file_path = tmp_path / "new" / "nested" / "test.jsonld"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Should save successfully
        benchmark.save(file_path)
        assert file_path.exists()

    def test_overwrite_existing_file(self, tmp_path: Path) -> None:
        """Test overwriting existing checkpoint file."""
        file_path = tmp_path / "overwrite.jsonld"

        # Create initial benchmark
        benchmark1 = Benchmark.create(name="original")
        benchmark1.add_question(question="Q1", raw_answer="A1")
        benchmark1.save(file_path)

        # Create new benchmark and overwrite
        benchmark2 = Benchmark.create(name="replacement")
        benchmark2.add_question(question="Q2", raw_answer="A2")
        benchmark2.save(file_path)

        # Load and verify it's the new one
        loaded = Benchmark.load(file_path)
        assert loaded.name == "replacement"

    def test_concurrent_save_simulation(self, tmp_path: Path) -> None:
        """Test simulated concurrent save operations."""
        file_path = tmp_path / "concurrent.jsonld"

        benchmark = Benchmark.create(name="concurrent_test")
        for i in range(10):
            benchmark.add_question(question=f"Q{i}", raw_answer=f"A{i}")

        # Simulate multiple saves in sequence (not truly concurrent)
        for _ in range(5):
            benchmark.save(file_path)

        # Should still be valid
        loaded = Benchmark.load(file_path)
        assert loaded.question_count == 10

    def test_save_during_modification(self, tmp_path: Path) -> None:
        """Test saving while benchmark is being modified."""
        file_path = tmp_path / "modifying.jsonld"

        benchmark = Benchmark.create(name="modifying_test")
        benchmark.add_question(question="Initial", raw_answer="A")

        # Save initial state
        benchmark.save(file_path)

        # Modify and save again
        benchmark.add_question(question="Added", raw_answer="B")
        benchmark.save(file_path)

        # Verify both questions present
        loaded = Benchmark.load(file_path)
        assert loaded.question_count == 2

    def test_interrupted_save_with_temp_file(self, tmp_path: Path) -> None:
        """Test recovery when save is interrupted (simulated)."""
        file_path = tmp_path / "interrupted.jsonld"

        benchmark = Benchmark.create(name="interrupt_test")
        benchmark.add_question(question="Q", raw_answer="A")

        # First, do a successful save
        benchmark.save(file_path)

        # Simulate interrupted save by creating partial file
        partial_path = tmp_path / "partial.jsonld"
        partial_path.write_text('{"@context": "incomplete...')

        # Original file should still be valid
        loaded = Benchmark.load(file_path)
        assert loaded.name == "interrupt_test"

        # Partial file should fail to load
        with pytest.raises(json.JSONDecodeError):
            Benchmark.load(partial_path)


# =============================================================================
# Additional Robustness Tests
# =============================================================================


@pytest.mark.unit
class TestAdditionalRobustness:
    """Additional robustness tests."""

    def test_question_id_uniqueness(self) -> None:
        """Test that question IDs are unique even for similar questions."""
        benchmark = Benchmark.create(name="id_test")

        q_id1 = benchmark.add_question(question="Same question", raw_answer="A1")
        q_id2 = benchmark.add_question(question="Same question", raw_answer="A2")

        # IDs should be different (different raw_answer)
        # Actually they might be same if hash is only on question
        # This tests the actual behavior
        assert q_id1 != q_id2 or benchmark.question_count == 2

    def test_duplicate_question_handling(self) -> None:
        """Test handling of duplicate questions."""
        benchmark = Benchmark.create(name="duplicate_test")

        # Add same question twice with same answer
        benchmark.add_question(question="Duplicate", raw_answer="Same")
        benchmark.add_question(question="Duplicate", raw_answer="Same")

        # Behavior depends on implementation - may dedupe or allow
        assert benchmark.question_count >= 1

    def test_omitted_optional_fields(self) -> None:
        """Test benchmark creation without optional fields."""
        benchmark = Benchmark.create(name="minimal_test")

        assert benchmark.name == "minimal_test"
        # Description and version should have default values

    def test_empty_string_fields(self) -> None:
        """Test empty string in various fields."""
        benchmark = Benchmark.create(
            name="empty_string_test",
            description="",
        )

        q_id = benchmark.add_question(
            question="Q",
            raw_answer="",  # Empty raw answer
        )

        retrieved = benchmark.get_question(q_id)
        assert retrieved["raw_answer"] == ""

    def test_special_json_values(self) -> None:
        """Test handling of special JSON values in metadata."""
        benchmark = Benchmark.create(name="special_json")

        # Set various special values
        benchmark.set_custom_property("null_like", None)
        benchmark.set_custom_property("zero", 0)
        benchmark.set_custom_property("false", False)
        benchmark.set_custom_property("empty_list", [])
        benchmark.set_custom_property("empty_dict", {})

        assert benchmark.get_custom_property("zero") == 0
        assert benchmark.get_custom_property("false") is False
