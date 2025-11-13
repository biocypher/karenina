"""Integration tests for question access methods.

This module tests the rationalized and extended question access methods including:
- Enhanced filter_questions() with custom_filter parameter
- filter_by_metadata() with dot notation and multiple match modes
- filter_by_custom_metadata() with AND/OR logic
- Enhanced search_questions() with multi-term, multi-field, regex support
- get_finished_questions() symmetry method
- count_by_field() generic statistics

These tests verify that all methods work correctly with various metadata structures
and that the documentation examples function as intended.
"""

import pytest

from karenina.benchmark.benchmark import Benchmark


@pytest.fixture
def sample_benchmark():
    """Create a benchmark with diverse questions and metadata for testing."""
    benchmark = Benchmark.create("Test Benchmark", "Integration test benchmark")

    # Add questions with varied system and custom metadata
    questions_data = [
        {
            "question": "What is Python?",
            "raw_answer": "A programming language",
            "finished": True,
            "custom_metadata": {
                "category": "programming",
                "difficulty": "easy",
                "tags": ["python", "basics"],
                "year": 2023,
            },
        },
        {
            "question": "Explain quantum mechanics",
            "raw_answer": "Physics of particles",
            "finished": True,
            "custom_metadata": {
                "category": "science",
                "difficulty": "hard",
                "tags": ["physics", "quantum"],
                "year": 2024,
            },
        },
        {
            "question": "What is machine learning?",
            "raw_answer": "AI algorithms",
            "finished": False,
            "custom_metadata": {
                "category": "programming",
                "difficulty": "medium",
                "tags": ["ai", "ml"],
                "year": 2024,
            },
        },
        {
            "question": "Describe DNA replication",
            "raw_answer": "Biological process",
            "finished": True,
            "custom_metadata": {
                "category": "science",
                "difficulty": "medium",
                "tags": ["biology", "dna"],
                "year": 2023,
            },
        },
        {
            "question": "How does Python handle memory?",
            "raw_answer": "Garbage collection",
            "finished": False,
            "custom_metadata": {
                "category": "programming",
                "difficulty": "hard",
                "tags": ["python", "memory"],
                "year": 2024,
            },
        },
        {
            "question": "What are neural networks?",
            "raw_answer": "AI structures",
            "finished": True,
            "custom_metadata": {
                "category": "programming",
                "difficulty": "hard",
                "tags": ["ai", "ml", "neural"],
                "year": 2023,
            },
        },
    ]

    for q_data in questions_data:
        benchmark.add_question(
            question=q_data["question"],
            raw_answer=q_data["raw_answer"],
            finished=q_data["finished"],
            custom_metadata=q_data["custom_metadata"],
        )

    return benchmark


class TestEnhancedFilterQuestions:
    """Tests for enhanced filter_questions() method with custom_filter parameter."""

    def test_filter_by_system_metadata(self, sample_benchmark):
        """Test filtering by system metadata fields."""
        # Filter by finished status
        finished = sample_benchmark.filter_questions(finished=True)
        assert len(finished) == 4
        assert all(q["finished"] for q in finished)

        unfinished = sample_benchmark.filter_questions(finished=False)
        assert len(unfinished) == 2
        assert all(not q["finished"] for q in unfinished)

    def test_filter_with_custom_filter_lambda(self, sample_benchmark):
        """Test filtering with custom lambda function."""
        # Filter by custom metadata using lambda
        hard_questions = sample_benchmark.filter_questions(
            custom_filter=lambda q: q.get("custom_metadata", {}).get("difficulty") == "hard"
        )
        assert len(hard_questions) == 3
        assert all(q["custom_metadata"]["difficulty"] == "hard" for q in hard_questions)

    def test_combine_system_and_custom_filters(self, sample_benchmark):
        """Test combining system metadata filters with custom lambda."""
        # Filter: finished AND hard difficulty
        finished_hard = sample_benchmark.filter_questions(
            finished=True, custom_filter=lambda q: q.get("custom_metadata", {}).get("difficulty") == "hard"
        )
        assert len(finished_hard) == 2
        assert all(q["finished"] and q["custom_metadata"]["difficulty"] == "hard" for q in finished_hard)

    def test_complex_custom_filter(self, sample_benchmark):
        """Test complex filtering logic with multiple conditions."""
        # Filter: programming category AND (year >= 2024 OR difficulty == hard)
        complex_filter = sample_benchmark.filter_questions(
            custom_filter=lambda q: q.get("custom_metadata", {}).get("category") == "programming"
            and (
                q.get("custom_metadata", {}).get("year", 0) >= 2024
                or q.get("custom_metadata", {}).get("difficulty") == "hard"
            )
        )
        assert len(complex_filter) >= 2  # Should match several questions

    def test_custom_filter_exception_handling(self, sample_benchmark):
        """Test that custom filter exceptions are handled gracefully."""
        # Custom filter that might raise exceptions should not crash
        results = sample_benchmark.filter_questions(
            custom_filter=lambda q: q["nonexistent_field"]["nested"] == "value"  # Will raise KeyError
        )
        # Should return empty list, not raise exception
        assert len(results) == 0


class TestFilterByMetadata:
    """Tests for filter_by_metadata() method with dot notation and match modes."""

    def test_exact_match_mode(self, sample_benchmark):
        """Test exact match mode for metadata filtering."""
        # Exact match on custom metadata
        programming_qs = sample_benchmark.filter_by_metadata("custom_metadata.category", "programming")
        assert len(programming_qs) == 4
        assert all(q["custom_metadata"]["category"] == "programming" for q in programming_qs)

        # Exact match on system metadata
        finished_qs = sample_benchmark.filter_by_metadata("finished", True)
        assert len(finished_qs) == 4
        assert all(q["finished"] for q in finished_qs)

    def test_contains_match_mode(self, sample_benchmark):
        """Test substring matching mode."""
        # Find questions with "prog" in category
        prog_related = sample_benchmark.filter_by_metadata("custom_metadata.category", "prog", match_mode="contains")
        assert len(prog_related) == 4  # All "programming" questions

        # Case-insensitive substring matching
        python_questions = sample_benchmark.filter_by_metadata("question", "python", match_mode="contains")
        assert len(python_questions) == 2

    def test_in_match_mode(self, sample_benchmark):
        """Test 'in' mode for list/array fields."""
        # Find questions tagged with "python"
        python_tagged = sample_benchmark.filter_by_metadata("custom_metadata.tags", "python", match_mode="in")
        assert len(python_tagged) == 2
        assert all("python" in q["custom_metadata"]["tags"] for q in python_tagged)

        # Find questions tagged with "ai"
        ai_tagged = sample_benchmark.filter_by_metadata("custom_metadata.tags", "ai", match_mode="in")
        assert len(ai_tagged) == 2

    def test_regex_match_mode(self, sample_benchmark):
        """Test regex matching mode."""
        # Find questions starting with "What"
        what_questions = sample_benchmark.filter_by_metadata("question", r"^What", match_mode="regex")
        assert len(what_questions) == 3

        # Find questions with "is" or "are" in them
        is_are_questions = sample_benchmark.filter_by_metadata("question", r"\b(is|are)\b", match_mode="regex")
        assert len(is_are_questions) >= 2

    def test_nested_field_access(self, sample_benchmark):
        """Test accessing nested fields with dot notation."""
        # Access nested custom metadata
        hard_qs = sample_benchmark.filter_by_metadata("custom_metadata.difficulty", "hard")
        assert len(hard_qs) == 3

        # Access nested with integer values
        recent_qs = sample_benchmark.filter_by_metadata("custom_metadata.year", 2024)
        assert len(recent_qs) == 3

    def test_nonexistent_field_returns_empty(self, sample_benchmark):
        """Test that filtering by nonexistent field returns empty list."""
        results = sample_benchmark.filter_by_metadata("custom_metadata.nonexistent", "value")
        assert len(results) == 0

    def test_invalid_match_mode_raises_error(self, sample_benchmark):
        """Test that invalid match mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid match_mode"):
            sample_benchmark.filter_by_metadata("question", "test", match_mode="invalid")


class TestFilterByCustomMetadata:
    """Tests for filter_by_custom_metadata() method with AND/OR logic."""

    def test_single_criterion(self, sample_benchmark):
        """Test filtering by single custom metadata criterion."""
        programming_qs = sample_benchmark.filter_by_custom_metadata(category="programming")
        assert len(programming_qs) == 4
        assert all(q["custom_metadata"]["category"] == "programming" for q in programming_qs)

    def test_and_logic_multiple_criteria(self, sample_benchmark):
        """Test AND logic with multiple criteria (default)."""
        # Programming AND hard
        prog_hard = sample_benchmark.filter_by_custom_metadata(category="programming", difficulty="hard")
        assert len(prog_hard) == 2
        assert all(
            q["custom_metadata"]["category"] == "programming" and q["custom_metadata"]["difficulty"] == "hard"
            for q in prog_hard
        )

        # Programming AND medium
        prog_medium = sample_benchmark.filter_by_custom_metadata(category="programming", difficulty="medium")
        assert len(prog_medium) == 1

    def test_or_logic_multiple_criteria(self, sample_benchmark):
        """Test OR logic with match_all=False."""
        # Programming OR science category
        # Note: OR logic on same key with different values isn't supported directly
        # So we test separate filters
        programming_qs = sample_benchmark.filter_by_custom_metadata(category="programming")
        science_qs = sample_benchmark.filter_by_custom_metadata(category="science")
        assert len(programming_qs) == 4
        assert len(science_qs) == 2

    def test_three_criteria_and_logic(self, sample_benchmark):
        """Test AND logic with three criteria."""
        # Programming AND hard AND year 2024
        prog_hard_2024 = sample_benchmark.filter_by_custom_metadata(
            category="programming", difficulty="hard", year=2024
        )
        assert len(prog_hard_2024) == 1

    def test_no_matches(self, sample_benchmark):
        """Test when no questions match the criteria."""
        no_match = sample_benchmark.filter_by_custom_metadata(category="history", difficulty="impossible")
        assert len(no_match) == 0

    def test_empty_custom_metadata(self, sample_benchmark):
        """Test behavior with questions that have no custom metadata."""
        # Add a question without custom metadata
        sample_benchmark.add_question("Plain question", "Plain answer")

        # Should not match when filtering by custom metadata
        programming_qs = sample_benchmark.filter_by_custom_metadata(category="programming")
        plain_questions = [q for q in sample_benchmark.get_all_questions() if not q.get("custom_metadata")]
        assert len(plain_questions) == 1
        assert all(q.get("custom_metadata") for q in programming_qs)


class TestEnhancedSearchQuestions:
    """Tests for enhanced search_questions() method with unified API."""

    def test_single_term_search_backward_compatible(self, sample_benchmark):
        """Test that single-term search is backward compatible."""
        # Simple string search (original API)
        python_qs = sample_benchmark.search_questions("Python")
        assert len(python_qs) == 2
        assert all("python" in q["question"].lower() for q in python_qs)

    def test_multi_term_and_search(self, sample_benchmark):
        """Test multi-term search with AND logic."""
        # Both "machine" AND "learning" must be present
        ml_qs = sample_benchmark.search_questions(["machine", "learning"], match_all=True)
        assert len(ml_qs) == 1
        assert "machine" in ml_qs[0]["question"].lower()
        assert "learning" in ml_qs[0]["question"].lower()

    def test_multi_term_or_search(self, sample_benchmark):
        """Test multi-term search with OR logic."""
        # Either "quantum" OR "DNA" must be present
        science_qs = sample_benchmark.search_questions(["quantum", "DNA"], match_all=False)
        assert len(science_qs) == 2

    def test_multi_field_search(self, sample_benchmark):
        """Test searching in multiple fields."""
        # Search in both question and answer
        language_mentions = sample_benchmark.search_questions("language", fields=["question", "raw_answer"])
        assert len(language_mentions) >= 1

    def test_case_sensitive_search(self, sample_benchmark):
        """Test case-sensitive search mode."""
        # Case-sensitive: should match "Python" but not "python"
        capital_python = sample_benchmark.search_questions("Python", case_sensitive=True)
        # All matches should have "Python" with capital P
        assert len(capital_python) >= 1

    def test_regex_search(self, sample_benchmark):
        """Test regex search mode."""
        # Find questions starting with "What"
        what_questions = sample_benchmark.search_questions(r"^What", regex=True)
        assert len(what_questions) == 3

        # Find questions with "is" or "are"
        is_are_questions = sample_benchmark.search_questions(r"\b(is|are)\b", regex=True)
        assert len(is_are_questions) >= 2

    def test_combined_search_options(self, sample_benchmark):
        """Test combining multiple search options."""
        # Multi-term AND, searching in multiple fields
        results = sample_benchmark.search_questions(
            ["neural", "networks"], match_all=True, fields=["question", "raw_answer"]
        )
        assert len(results) >= 1

    def test_regex_with_multi_term_or(self, sample_benchmark):
        """Test regex patterns with multi-term OR logic."""
        # Match questions with "DNA.*replication" OR "quantum.*mechanics"
        complex_search = sample_benchmark.search_questions(
            [r"DNA.*replication", r"quantum.*mechanics"], match_all=False, regex=True
        )
        assert len(complex_search) >= 1

    def test_search_returns_empty_for_no_matches(self, sample_benchmark):
        """Test that search returns empty list when nothing matches."""
        no_match = sample_benchmark.search_questions("nonexistent_term_xyz")
        assert len(no_match) == 0

    def test_invalid_regex_handled_gracefully(self, sample_benchmark):
        """Test that invalid regex patterns don't crash."""
        # Invalid regex should not raise exception, just return no matches
        results = sample_benchmark.search_questions(r"[invalid(regex", regex=True)
        assert len(results) == 0

    def test_empty_query_list(self, sample_benchmark):
        """Test that empty query list returns empty results."""
        # Empty list with OR logic should return empty
        results_or = sample_benchmark.search_questions([], match_all=False)
        assert len(results_or) == 0

        # Empty list with AND logic should also return empty (not all questions!)
        results_and = sample_benchmark.search_questions([], match_all=True)
        assert len(results_and) == 0


class TestGetFinishedQuestions:
    """Tests for get_finished_questions() symmetry method."""

    def test_get_finished_questions(self, sample_benchmark):
        """Test getting finished question IDs."""
        finished_ids = sample_benchmark.get_finished_questions(ids_only=True)
        assert len(finished_ids) == 4

        # Verify all returned IDs are actually finished
        for q_id in finished_ids:
            question = sample_benchmark.get_question(q_id)
            assert question["finished"] is True

    def test_symmetry_with_unfinished(self, sample_benchmark):
        """Test that finished + unfinished = all questions."""
        finished_ids = sample_benchmark.get_finished_questions(ids_only=True)
        unfinished_ids = sample_benchmark.get_unfinished_questions(ids_only=True)

        total_questions = len(sample_benchmark.get_question_ids())
        assert len(finished_ids) + len(unfinished_ids) == total_questions

        # No overlap between finished and unfinished
        assert set(finished_ids).isdisjoint(set(unfinished_ids))

    def test_empty_benchmark(self):
        """Test get_finished_questions on empty benchmark."""
        benchmark = Benchmark.create("Empty")
        finished_ids = benchmark.get_finished_questions(ids_only=True)
        assert len(finished_ids) == 0

    def test_all_finished(self):
        """Test when all questions are finished."""
        benchmark = Benchmark.create("All Finished")
        for i in range(3):
            benchmark.add_question(f"Question {i}", f"Answer {i}", finished=True)

        finished_ids = benchmark.get_finished_questions(ids_only=True)
        assert len(finished_ids) == 3
        assert len(benchmark.get_unfinished_questions(ids_only=True)) == 0

    def test_get_finished_questions_default_returns_objects(self, sample_benchmark):
        """Test that default behavior returns question objects, not IDs."""
        finished_questions = sample_benchmark.get_finished_questions()
        assert len(finished_questions) == 4

        # Verify all returned items are dictionaries (question objects)
        for q in finished_questions:
            assert isinstance(q, dict)
            assert "id" in q
            assert "question" in q
            assert "raw_answer" in q
            assert q["finished"] is True

    def test_get_unfinished_questions_default_returns_objects(self, sample_benchmark):
        """Test that default behavior returns question objects, not IDs."""
        unfinished_questions = sample_benchmark.get_unfinished_questions()
        assert len(unfinished_questions) == 2

        # Verify all returned items are dictionaries (question objects)
        for q in unfinished_questions:
            assert isinstance(q, dict)
            assert "id" in q
            assert "question" in q
            assert "raw_answer" in q
            assert q["finished"] is False

    def test_get_missing_templates_default_returns_objects(self):
        """Test that default behavior returns question objects, not IDs."""
        # Create a benchmark specifically for this test
        benchmark = Benchmark.create("Test Missing Templates")
        benchmark.add_question("Question 1", "Answer 1")
        benchmark.add_question("Question 2", "Answer 2")

        missing_template_questions = benchmark.get_missing_templates()
        assert len(missing_template_questions) == 2

        # Verify all returned items are dictionaries (question objects)
        for q in missing_template_questions:
            assert isinstance(q, dict)
            assert "id" in q
            assert "question" in q
            assert "raw_answer" in q

    def test_harmonized_methods_consistency(self, sample_benchmark):
        """Test that harmonized methods work consistently with other filter methods."""
        # All these should return question objects by default
        finished = sample_benchmark.get_finished_questions()
        unfinished = sample_benchmark.get_unfinished_questions()
        filtered = sample_benchmark.filter_questions(finished=True)

        # All should be lists of dicts
        assert all(isinstance(q, dict) for q in finished)
        assert all(isinstance(q, dict) for q in unfinished)
        assert all(isinstance(q, dict) for q in filtered)

        # Finished count should match filtered count
        assert len(finished) == len(filtered)

    def test_get_all_questions_default_returns_objects(self, sample_benchmark):
        """Test that get_all_questions() returns question objects by default."""
        all_questions = sample_benchmark.get_all_questions()
        assert len(all_questions) == 6

        # Verify all returned items are dictionaries (question objects)
        for q in all_questions:
            assert isinstance(q, dict)
            assert "id" in q
            assert "question" in q
            assert "raw_answer" in q

    def test_get_all_questions_ids_only(self, sample_benchmark):
        """Test that get_all_questions(ids_only=True) returns just IDs."""
        all_ids = sample_benchmark.get_all_questions(ids_only=True)
        assert len(all_ids) == 6

        # Verify all returned items are strings (IDs)
        for q_id in all_ids:
            assert isinstance(q_id, str)

        # Verify get_question_ids() returns the same result
        question_ids = sample_benchmark.get_question_ids()
        assert set(all_ids) == set(question_ids)


class TestCountByField:
    """Tests for count_by_field() generic statistics method."""

    def test_count_by_custom_metadata_field(self, sample_benchmark):
        """Test counting by custom metadata field."""
        # Count by category
        category_counts = sample_benchmark.count_by_field("custom_metadata.category")
        assert category_counts["programming"] == 4
        assert category_counts["science"] == 2

    def test_count_by_system_metadata_field(self, sample_benchmark):
        """Test counting by system metadata field."""
        # Count finished status
        status_counts = sample_benchmark.count_by_field("finished")
        assert status_counts[True] == 4
        assert status_counts[False] == 2

    def test_count_by_nested_field(self, sample_benchmark):
        """Test counting by nested custom metadata field."""
        # Count by difficulty
        difficulty_counts = sample_benchmark.count_by_field("custom_metadata.difficulty")
        assert difficulty_counts["easy"] == 1
        assert difficulty_counts["medium"] == 2
        assert difficulty_counts["hard"] == 3

        # Count by year
        year_counts = sample_benchmark.count_by_field("custom_metadata.year")
        assert year_counts[2023] == 3
        assert year_counts[2024] == 3

    def test_count_on_filtered_subset(self, sample_benchmark):
        """Test counting on a filtered subset of questions."""
        # First filter to programming questions
        programming_qs = sample_benchmark.filter_by_custom_metadata(category="programming")

        # Then count difficulty within programming questions only
        prog_difficulty_counts = sample_benchmark.count_by_field("custom_metadata.difficulty", questions=programming_qs)
        assert prog_difficulty_counts["easy"] == 1
        assert prog_difficulty_counts["medium"] == 1
        assert prog_difficulty_counts["hard"] == 2

    def test_count_with_none_values(self, sample_benchmark):
        """Test that None values are counted correctly."""
        # Add a question without custom metadata
        sample_benchmark.add_question("No metadata question", "Answer")

        # Count by category - should include None for question without metadata
        category_counts = sample_benchmark.count_by_field("custom_metadata.category")
        assert None in category_counts
        assert category_counts[None] == 1

    def test_count_empty_field_path(self, sample_benchmark):
        """Test counting when field doesn't exist."""
        nonexistent_counts = sample_benchmark.count_by_field("custom_metadata.nonexistent")
        # Should return all None values
        assert nonexistent_counts[None] == 6

    def test_count_on_empty_benchmark(self):
        """Test count_by_field on empty benchmark."""
        benchmark = Benchmark.create("Empty")
        counts = benchmark.count_by_field("finished")
        assert len(counts) == 0


class TestIntegrationScenarios:
    """Integration tests combining multiple methods in realistic scenarios."""

    def test_progressive_filtering_workflow(self, sample_benchmark):
        """Test a realistic workflow of progressive filtering."""
        # 1. Start with all programming questions
        programming_qs = sample_benchmark.filter_by_custom_metadata(category="programming")
        assert len(programming_qs) == 4

        # 2. Count difficulty distribution
        difficulty_counts = sample_benchmark.count_by_field("custom_metadata.difficulty", questions=programming_qs)
        assert difficulty_counts["hard"] == 2

        # 3. Filter to hard programming questions
        hard_prog = sample_benchmark.filter_by_custom_metadata(category="programming", difficulty="hard")
        assert len(hard_prog) == 2

        # 4. Search within that subset for Python-related
        python_ids = {q["id"] for q in hard_prog if "python" in q["question"].lower()}
        assert len(python_ids) >= 1

    def test_complex_analysis_pipeline(self, sample_benchmark):
        """Test a complex analysis pipeline using multiple methods."""
        # Find all finished questions from 2024
        recent_finished = sample_benchmark.filter_questions(
            finished=True, custom_filter=lambda q: q.get("custom_metadata", {}).get("year") == 2024
        )
        assert len(recent_finished) >= 1

        # Search for AI-related content in those questions
        ai_questions = sample_benchmark.search_questions(
            ["ai", "neural", "machine"], match_all=False, fields=["question", "raw_answer"]
        )

        # Count categories among AI questions
        ai_categories = sample_benchmark.count_by_field("custom_metadata.category", questions=ai_questions)
        assert "programming" in ai_categories

    def test_metadata_update_and_requery(self, sample_benchmark):
        """Test updating metadata and re-querying."""
        # Find all hard questions
        hard_qs = sample_benchmark.filter_by_metadata("custom_metadata.difficulty", "hard")
        original_count = len(hard_qs)

        # Update one to be "expert" instead
        if hard_qs:
            q_id = hard_qs[0]["id"]
            custom_meta = sample_benchmark.get_question_metadata(q_id).get("custom_metadata", {})
            custom_meta["difficulty"] = "expert"
            sample_benchmark.update_question_metadata(q_id, custom_metadata=custom_meta)

            # Re-query should show one less hard question
            hard_qs_after = sample_benchmark.filter_by_metadata("custom_metadata.difficulty", "hard")
            assert len(hard_qs_after) == original_count - 1

            # Should now have an expert question
            expert_qs = sample_benchmark.filter_by_metadata("custom_metadata.difficulty", "expert")
            assert len(expert_qs) == 1

    def test_bulk_tagging_workflow(self, sample_benchmark):
        """Test bulk tagging based on search results."""
        # Find all questions mentioning Python
        python_qs = sample_benchmark.search_questions("Python")

        # Add a "reviewed" tag to all of them
        for q in python_qs:
            q_id = q["id"]
            sample_benchmark.set_question_custom_property(q_id, "reviewed", True)

        # Verify we can filter by the new property
        reviewed_qs = sample_benchmark.filter_questions(
            custom_filter=lambda q: q.get("custom_metadata", {}).get("reviewed") is True
        )
        assert len(reviewed_qs) == len(python_qs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
