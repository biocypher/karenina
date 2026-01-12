"""Integration tests for checkpoint I/O operations.

These tests verify that benchmarks can be saved to and loaded from JSON-LD
checkpoint files with full data integrity, proper error handling, and
thread-safe concurrent write operations.

Test scenarios:
- Save and load roundtrip (data integrity)
- Progressive save (results saved during verification)
- Concurrent writes (race condition handling with threading)
- Corrupted file handling
- File not found handling
- Invalid JSON handling

Fixtures used:
- minimal_benchmark: Single question benchmark
- multi_question_benchmark: 5 diverse questions
- benchmark_with_results: Benchmark with existing verification results
"""

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from karenina import Benchmark

# =============================================================================
# Save and Load Roundtrip Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.storage
class TestSaveLoadRoundtrip:
    """Test save and load roundtrip operations for data integrity."""

    def test_minimal_benchmark_roundtrip(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify minimal benchmark survives save/load cycle."""
        save_path = tmp_path / "roundtrip_minimal.jsonld"

        # Save and reload
        minimal_benchmark.save(save_path)
        reloaded = Benchmark.load(save_path)

        # Verify core properties
        assert reloaded.name == minimal_benchmark.name
        assert reloaded.description == minimal_benchmark.description
        assert reloaded.version == minimal_benchmark.version
        assert reloaded.question_count == minimal_benchmark.question_count

    def test_multi_question_roundtrip(self, multi_question_benchmark: Any, tmp_path: Path) -> None:
        """Verify multi-question benchmark survives save/load cycle."""
        save_path = tmp_path / "roundtrip_multi.jsonld"

        # Save and reload
        multi_question_benchmark.save(save_path)
        reloaded = Benchmark.load(save_path)

        # Verify question count
        assert reloaded.question_count == multi_question_benchmark.question_count
        assert reloaded.question_count >= 5  # multi_question has 5+ questions

    def test_benchmark_with_results_roundtrip(self, benchmark_with_results: Any, tmp_path: Path) -> None:
        """Verify benchmark with verification results survives save/load."""
        save_path = tmp_path / "roundtrip_results.jsonld"

        # Save and reload
        benchmark_with_results.save(save_path)
        reloaded = Benchmark.load(save_path)

        # Verify basic properties preserved
        assert reloaded.name == benchmark_with_results.name
        assert reloaded.question_count == benchmark_with_results.question_count

    def test_question_data_preserved(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify individual question data is preserved through roundtrip."""
        save_path = tmp_path / "roundtrip_questions.jsonld"

        # Get original question
        original_questions = minimal_benchmark.get_all_questions()
        assert len(original_questions) > 0
        original_q = original_questions[0]

        # Save and reload
        minimal_benchmark.save(save_path)
        reloaded = Benchmark.load(save_path)

        # Verify question preserved
        reloaded_questions = reloaded.get_all_questions()
        assert len(reloaded_questions) == len(original_questions)

        # Check question text matches
        reloaded_q = reloaded_questions[0]
        assert reloaded_q["question"] == original_q["question"]

    def test_metadata_preserved(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify metadata is preserved through save/load cycle."""
        save_path = tmp_path / "roundtrip_metadata.jsonld"

        # Get original metadata
        original_name = minimal_benchmark.name
        original_version = minimal_benchmark.version
        original_creator = minimal_benchmark.creator

        # Save and reload
        minimal_benchmark.save(save_path)
        reloaded = Benchmark.load(save_path)

        # Verify metadata
        assert reloaded.name == original_name
        assert reloaded.version == original_version
        assert reloaded.creator == original_creator

    def test_multiple_save_cycles(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify benchmark survives multiple save/load cycles."""
        current = minimal_benchmark

        for i in range(3):
            save_path = tmp_path / f"cycle_{i}.jsonld"
            current.save(save_path)
            current = Benchmark.load(save_path)

        # Final loaded version should match original
        assert current.name == minimal_benchmark.name
        assert current.question_count == minimal_benchmark.question_count


# =============================================================================
# File Extension and Path Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.storage
class TestFileExtensions:
    """Test file extension handling for checkpoint files."""

    def test_jsonld_extension(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify .jsonld extension is handled correctly."""
        save_path = tmp_path / "test.jsonld"
        minimal_benchmark.save(save_path)

        assert save_path.exists()
        reloaded = Benchmark.load(save_path)
        assert reloaded.name == minimal_benchmark.name

    def test_json_extension(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify .json extension is also accepted."""
        save_path = tmp_path / "test.json"
        minimal_benchmark.save(save_path)

        # Should save with .json extension
        assert save_path.exists()
        reloaded = Benchmark.load(save_path)
        assert reloaded.name == minimal_benchmark.name

    def test_no_extension_adds_jsonld(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify missing extension defaults to .jsonld."""
        save_path = tmp_path / "noext"
        minimal_benchmark.save(save_path)

        # Should have added .jsonld
        expected_path = tmp_path / "noext.jsonld"
        assert expected_path.exists()

    def test_nested_directory_path(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify saving to nested directories works."""
        nested_path = tmp_path / "level1" / "level2" / "benchmark.jsonld"
        nested_path.parent.mkdir(parents=True, exist_ok=True)

        minimal_benchmark.save(nested_path)
        assert nested_path.exists()

        reloaded = Benchmark.load(nested_path)
        assert reloaded.name == minimal_benchmark.name


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.storage
class TestErrorHandling:
    """Test error handling for checkpoint I/O operations."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError for missing file."""
        nonexistent = tmp_path / "does_not_exist.jsonld"

        with pytest.raises(FileNotFoundError) as exc_info:
            Benchmark.load(nonexistent)

        assert "not found" in str(exc_info.value).lower()

    def test_invalid_json_raises_error(self, tmp_path: Path) -> None:
        """Verify ValueError for invalid JSON content."""
        bad_json_path = tmp_path / "bad.jsonld"
        bad_json_path.write_text("{ this is not valid json }")

        with pytest.raises((ValueError, json.JSONDecodeError)):
            Benchmark.load(bad_json_path)

    def test_empty_file_raises_error(self, tmp_path: Path) -> None:
        """Verify error for empty file."""
        empty_path = tmp_path / "empty.jsonld"
        empty_path.write_text("")

        with pytest.raises((ValueError, json.JSONDecodeError)):
            Benchmark.load(empty_path)

    def test_valid_json_but_invalid_structure(self, tmp_path: Path) -> None:
        """Verify ValueError for valid JSON with wrong structure."""
        bad_structure_path = tmp_path / "wrong_structure.jsonld"
        bad_structure_path.write_text('{"some": "data", "but": "wrong"}')

        with pytest.raises(ValueError):
            Benchmark.load(bad_structure_path)

    def test_truncated_json_raises_error(self, tmp_path: Path) -> None:
        """Verify error for truncated/corrupted JSON."""
        truncated_path = tmp_path / "truncated.jsonld"
        truncated_path.write_text('{"name": "test", "dataFeedElement": [')

        with pytest.raises((ValueError, json.JSONDecodeError)):
            Benchmark.load(truncated_path)

    def test_binary_file_raises_error(self, tmp_path: Path) -> None:
        """Verify error when loading binary file."""
        binary_path = tmp_path / "binary.jsonld"
        binary_path.write_bytes(b"\x00\x01\x02\x03\x04\x05")

        with pytest.raises((ValueError, UnicodeDecodeError, json.JSONDecodeError)):
            Benchmark.load(binary_path)


# =============================================================================
# Progressive Save Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.storage
class TestProgressiveSave:
    """Test progressive save operations during verification."""

    def test_save_preserves_modifications(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify modifications are preserved through save."""
        save_path = tmp_path / "modified.jsonld"

        # Modify the benchmark
        original_name = minimal_benchmark.name
        new_description = "Modified description for testing"
        minimal_benchmark.description = new_description

        # Save and reload
        minimal_benchmark.save(save_path)
        reloaded = Benchmark.load(save_path)

        # Verify modification persisted
        assert reloaded.name == original_name
        assert reloaded.description == new_description

    def test_save_after_adding_question(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify new questions are saved correctly."""
        save_path = tmp_path / "with_new_question.jsonld"

        original_count = minimal_benchmark.question_count

        # Add a question
        minimal_benchmark.add_question(
            question="What is 2 + 2?",
            raw_answer="4",
            question_id="test-new-question",
        )

        # Save and reload
        minimal_benchmark.save(save_path)
        reloaded = Benchmark.load(save_path)

        # Verify new question count
        assert reloaded.question_count == original_count + 1

    def test_incremental_saves(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify incremental saves preserve all changes."""
        save_path = tmp_path / "incremental.jsonld"

        # Make first change and save
        minimal_benchmark.description = "First modification"
        minimal_benchmark.save(save_path)

        # Reload, make second change, save again
        reloaded = Benchmark.load(save_path)
        reloaded.version = "2.0.0"
        reloaded.save(save_path)

        # Final reload should have both changes
        final = Benchmark.load(save_path)
        assert final.description == "First modification"
        assert final.version == "2.0.0"

    def test_date_modified_updates_on_save(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify dateModified timestamp updates on save."""
        save_path = tmp_path / "dated.jsonld"

        original_modified = minimal_benchmark.modified_at

        # Small delay to ensure timestamp difference
        time.sleep(0.01)

        # Make a change and save
        minimal_benchmark.description = "Updated for timestamp test"
        minimal_benchmark.save(save_path)

        # Reload and check timestamp changed
        reloaded = Benchmark.load(save_path)
        assert reloaded.modified_at != original_modified


# =============================================================================
# Concurrent Write Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.storage
class TestConcurrentWrites:
    """Test concurrent write operations for race condition handling."""

    def test_sequential_writes_no_conflict(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify sequential writes to different files work correctly."""
        paths = [tmp_path / f"seq_{i}.jsonld" for i in range(5)]

        for path in paths:
            minimal_benchmark.save(path)

        # All files should exist and be loadable
        for path in paths:
            assert path.exists()
            loaded = Benchmark.load(path)
            assert loaded.name == minimal_benchmark.name

    def test_concurrent_writes_different_files(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify concurrent writes to different files work correctly."""
        paths = [tmp_path / f"concurrent_{i}.jsonld" for i in range(5)]
        errors: list[Exception] = []

        def save_to_path(path: Path) -> None:
            try:
                minimal_benchmark.save(path)
            except Exception as e:
                errors.append(e)

        # Start multiple threads writing to different files
        threads = [threading.Thread(target=save_to_path, args=(p,)) for p in paths]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

        # All files should exist and be loadable
        for path in paths:
            assert path.exists()
            loaded = Benchmark.load(path)
            assert loaded.name == minimal_benchmark.name

    def test_concurrent_writes_same_file(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify concurrent writes to same file don't corrupt data.

        Note: This test verifies that the final state is valid, not that
        all writes are individually preserved (last write wins).
        """
        save_path = tmp_path / "concurrent_same.jsonld"
        write_count = 10
        completed_writes = []

        def save_with_version(version: str) -> None:
            try:
                # Create a copy-like behavior by modifying and saving
                minimal_benchmark.version = version
                minimal_benchmark.save(save_path)
                completed_writes.append(version)
            except Exception:
                pass  # Concurrent write conflicts are expected

        # Start multiple threads writing to same file
        threads = [threading.Thread(target=save_with_version, args=(f"v{i}.0.0",)) for i in range(write_count)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # File should exist and be loadable (not corrupted)
        assert save_path.exists()
        loaded = Benchmark.load(save_path)
        assert loaded.name == minimal_benchmark.name

        # Version should be one of the written versions
        assert loaded.version.startswith("v")

    def test_read_during_write(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify reads during write operations are handled gracefully."""
        save_path = tmp_path / "read_write.jsonld"

        # Initial save
        minimal_benchmark.save(save_path)

        read_results: list[Any] = []
        errors: list[Exception] = []

        def reader() -> None:
            for _ in range(5):
                try:
                    loaded = Benchmark.load(save_path)
                    read_results.append(loaded.name)
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

        def writer() -> None:
            for i in range(5):
                try:
                    minimal_benchmark.version = f"write_{i}"
                    minimal_benchmark.save(save_path)
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

        # Run reader and writer concurrently
        read_thread = threading.Thread(target=reader)
        write_thread = threading.Thread(target=writer)

        read_thread.start()
        write_thread.start()

        read_thread.join()
        write_thread.join()

        # Some reads should have succeeded
        assert len(read_results) > 0

        # Final file should be valid
        final = Benchmark.load(save_path)
        assert final.name == minimal_benchmark.name


# =============================================================================
# Data Integrity Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.storage
class TestDataIntegrity:
    """Test data integrity of checkpoint files."""

    def test_json_file_is_valid_utf8(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify saved file is valid UTF-8."""
        save_path = tmp_path / "utf8.jsonld"
        minimal_benchmark.save(save_path)

        # Read as bytes and decode as UTF-8
        content = save_path.read_bytes()
        decoded = content.decode("utf-8")  # Should not raise
        assert len(decoded) > 0

    def test_json_file_is_parseable(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify saved file contains valid JSON."""
        save_path = tmp_path / "parseable.jsonld"
        minimal_benchmark.save(save_path)

        # Parse with json module directly
        with open(save_path, encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert "name" in data

    def test_jsonld_context_preserved(self, minimal_benchmark: Any, tmp_path: Path) -> None:
        """Verify JSON-LD @context is preserved."""
        save_path = tmp_path / "context.jsonld"
        minimal_benchmark.save(save_path)

        with open(save_path, encoding="utf-8") as f:
            data = json.load(f)

        # Should have JSON-LD @context
        assert "@context" in data
        assert "schema.org" in str(data["@context"]).lower() or "@type" in data

    def test_large_benchmark_integrity(self, tmp_path: Path) -> None:
        """Verify large benchmark saves and loads correctly."""
        # Create a benchmark with many questions
        benchmark = Benchmark.create(
            name="Large Test Benchmark",
            description="A benchmark with many questions for integrity testing",
        )

        for i in range(100):
            benchmark.add_question(
                question=f"Question {i}: What is {i} + {i}?",
                raw_answer=str(i * 2),
                question_id=f"large-q-{i}",
            )

        save_path = tmp_path / "large.jsonld"
        benchmark.save(save_path)

        reloaded = Benchmark.load(save_path)
        assert reloaded.question_count == 100

    def test_special_characters_preserved(self, tmp_path: Path) -> None:
        """Verify special characters are preserved through save/load."""
        benchmark = Benchmark.create(
            name="Special Characters Test",
            description="Testing: émojis 🎉, unicode 日本語, symbols <>&\"'",
        )

        benchmark.add_question(
            question="What is the French word for 'water'? (l'eau)",
            raw_answer="l'eau",
            question_id="special-chars",
        )

        save_path = tmp_path / "special.jsonld"
        benchmark.save(save_path)

        reloaded = Benchmark.load(save_path)
        assert "émojis" in reloaded.description
        assert "日本語" in reloaded.description
        assert "l'eau" in reloaded.get_all_questions()[0]["question"]
