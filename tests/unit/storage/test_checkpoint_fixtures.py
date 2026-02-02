"""Test that sample checkpoint fixtures can be loaded and used.

This module verifies the checkpoint fixtures are valid JSON-LD and can be
loaded via Benchmark.load().

Fixtures tested:
- tests/fixtures/checkpoints/minimal.jsonld
- tests/fixtures/checkpoints/with_results.jsonld
- tests/fixtures/checkpoints/multi_question.jsonld
"""

from pathlib import Path

import pytest

from karenina import Benchmark


@pytest.mark.unit
def test_minimal_checkpoint_loads(fixtures_dir: Path) -> None:
    """Test that minimal.jsonld can be loaded via Benchmark.load()."""
    checkpoint_path = fixtures_dir / "checkpoints" / "minimal.jsonld"

    benchmark = Benchmark.load(checkpoint_path)

    assert benchmark.name == "minimal-test"
    assert benchmark.description == "A minimal benchmark for testing with one simple question"
    assert benchmark.version == "0.1.0"
    assert benchmark.question_count == 1


@pytest.mark.unit
def test_minimal_checkpoint_question_content(fixtures_dir: Path) -> None:
    """Test that minimal checkpoint has correct question content."""
    checkpoint_path = fixtures_dir / "checkpoints" / "minimal.jsonld"

    benchmark = Benchmark.load(checkpoint_path)

    questions = benchmark.get_all_questions()
    assert len(questions) == 1

    q = questions[0]
    assert "What is 2+2?" in q["question"]
    assert q["raw_answer"] == "4"
    # Note: minimal.jsonld doesn't have finished property, defaults to False
    assert q["finished"] is False


@pytest.mark.unit
def test_with_results_checkpoint_loads(fixtures_dir: Path) -> None:
    """Test that with_results.jsonld can be loaded via Benchmark.load()."""
    checkpoint_path = fixtures_dir / "checkpoints" / "with_results.jsonld"

    benchmark = Benchmark.load(checkpoint_path)

    assert benchmark.name == "with-results-test"
    assert benchmark.question_count == 1


@pytest.mark.unit
def test_with_results_checkpoint_has_rubrics(fixtures_dir: Path) -> None:
    """Test that with_results checkpoint has global and question rubrics."""
    checkpoint_path = fixtures_dir / "checkpoints" / "with_results.jsonld"

    benchmark = Benchmark.load(checkpoint_path)

    # Check global rubric exists
    global_rubric = benchmark.get_global_rubric()
    assert global_rubric is not None
    assert len(global_rubric.callable_traits) == 1

    # Check question-specific rubric exists
    q_rubric = benchmark._rubric_manager.get_question_rubric("q002")
    assert q_rubric is not None


@pytest.mark.unit
def test_with_results_checkpoint_metadata(fixtures_dir: Path) -> None:
    """Test that with_results checkpoint has verification result metadata.

    Note: In JSON-LD, custom metadata properties must have "custom_" prefix.
    The extraction function strips this prefix when returning custom_metadata.
    """
    checkpoint_path = fixtures_dir / "checkpoints" / "with_results.jsonld"

    benchmark = Benchmark.load(checkpoint_path)

    # Check custom metadata with verification results
    # The "custom_" prefix is stripped during extraction
    q = benchmark.get_question("q002")
    custom_metadata = q.get("custom_metadata", {})
    assert "verification_results" in custom_metadata
    assert custom_metadata["verification_results"]["template_result"]["passed"] is True


@pytest.mark.unit
def test_multi_question_checkpoint_loads(fixtures_dir: Path) -> None:
    """Test that multi_question.jsonld can be loaded via Benchmark.load()."""
    checkpoint_path = fixtures_dir / "checkpoints" / "multi_question.jsonld"

    benchmark = Benchmark.load(checkpoint_path)

    assert benchmark.name == "multi-question-test"
    assert benchmark.question_count == 5


@pytest.mark.unit
def test_multi_question_checkpoint_all_questions(fixtures_dir: Path) -> None:
    """Test that multi_question checkpoint has all 5 questions with correct content."""
    checkpoint_path = fixtures_dir / "checkpoints" / "multi_question.jsonld"

    benchmark = Benchmark.load(checkpoint_path)

    question_ids = benchmark.get_question_ids()
    assert len(question_ids) == 5
    assert "mq001" in question_ids
    assert "mq005" in question_ids

    # Check specific questions
    mq001 = benchmark.get_question("mq001")
    assert "boiling point" in mq001["question"].lower()

    mq005 = benchmark.get_question("mq005")
    assert mq005["finished"] is False  # This one is marked unfinished


@pytest.mark.unit
def test_multi_question_checkpoint_keywords(fixtures_dir: Path) -> None:
    """Test that multi_question checkpoint questions have proper keywords."""
    checkpoint_path = fixtures_dir / "checkpoints" / "multi_question.jsonld"

    benchmark = Benchmark.load(checkpoint_path)

    # Each question should have keywords
    for q_id in benchmark.get_question_ids():
        q = benchmark.get_question(q_id)
        assert "keywords" in q
        assert len(q["keywords"]) > 0


@pytest.mark.unit
def test_multi_question_checkpoint_global_rubric(fixtures_dir: Path) -> None:
    """Test that multi_question checkpoint has global rubric."""
    checkpoint_path = fixtures_dir / "checkpoints" / "multi_question.jsonld"

    benchmark = Benchmark.load(checkpoint_path)

    global_rubric = benchmark.get_global_rubric()
    assert global_rubric is not None
    assert len(global_rubric.callable_traits) == 1


@pytest.mark.unit
def test_checkpoint_roundtrip(tmp_path: Path, fixtures_dir: Path) -> None:
    """Test that loading and saving a checkpoint preserves data."""
    # Load original
    original_path = fixtures_dir / "checkpoints" / "minimal.jsonld"
    original = Benchmark.load(original_path)

    # Save to temp location
    temp_path = tmp_path / "roundtrip.jsonld"
    original.save(temp_path)

    # Load saved version
    loaded = Benchmark.load(temp_path)

    # Verify key attributes preserved
    assert loaded.name == original.name
    assert loaded.question_count == original.question_count
    assert loaded.get_question_ids() == original.get_question_ids()
